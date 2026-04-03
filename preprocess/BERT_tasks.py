from __future__ import annotations
import os
import argparse
from typing import Optional, List
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from core import settings
from core.config import BERTClassificationConfig
from core.data_utils import set_seed, read_csv_safely, save_csv_safely, process_csv_format

# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
class TextLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        t = str(self.texts[idx])
        enc = self.tokenizer(
            t,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def _resolve_model_name(cfg: BERTClassificationConfig) -> str:
    return cfg.model_name_or_path or settings.BERT_MODEL_PATH


def _label_to_id_map(label_names: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(label_names)}


def _validate_labels(labels: list[int], num_labels: int) -> list[int]:
    """Clip labels to valid range [0, num_labels-1]"""
    validated = []
    clipped_count = 0
    for label in labels:
        if label < 0 or label >= num_labels:
            validated.append(max(0, min(label, num_labels - 1)))
            clipped_count += 1
        else:
            validated.append(label)
    
    if clipped_count > 0:
        print(f"[warning] clipped {clipped_count} labels to valid range [0, {num_labels-1}]")
    
    return validated


def evaluate_results(actual_labels, predicted_labels, output_dir: str | None = None, task_type: str = "classification"):
    cm = confusion_matrix(actual_labels, predicted_labels)
    print("\nConfusion Matrix:")
    print(cm)
    cr = classification_report(actual_labels, predicted_labels, digits=3)
    print("\nClassification Report:")
    print(cr)

    # Save evaluation to file
    if output_dir:
        eval_path = os.path.join(output_dir, f"{task_type}_evaluation.txt")
        with open(eval_path, 'w', encoding='utf-8') as f:
            f.write(f"BERT Classification Evaluation Results ({task_type})\n")
            f.write("="*60 + "\n")
            f.write(f"Confusion Matrix:\n{cm}\n\n")
            f.write(f"Classification Report:\n{cr}\n")
        print(f"[bert] evaluation saved: {eval_path}")


# --------------------------------------------------------------------------------------
# Train & Evaluate (with train/test split)
# --------------------------------------------------------------------------------------
def train_and_evaluate(cfg: BERTClassificationConfig) -> None:
    """
    Train BERT classifier with stratified train/test split.
    Each run uses a different random split for robustness.
    Split ratio is configured via cfg.test_size (default 0.2 = 80:20).
    """
    set_seed(cfg.seed)
    model_name = _resolve_model_name(cfg)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=cfg.num_labels)

    # Load and preprocess data
    data_csv = os.path.join(settings.DATA_DIR, cfg.input_filename)
    df = read_csv_safely(data_csv)
    df = process_csv_format(df, cfg.classification_type, ["text", "label"])

    label2id = _label_to_id_map(cfg.label_names)
    id2label = {i: n for i, n in enumerate(cfg.label_names)}

    # Convert string labels to int
    y = df["label"].str.strip().str.lower().map(
        {k.lower(): v for k, v in label2id.items()}
    ).fillna(-1).astype(int).tolist()

    if any(v < 0 for v in y):
        bad = df.loc[[i for i, v in enumerate(y) if v < 0], "label"].unique().tolist()
        raise ValueError(f"Unknown labels found: {bad} (expected {cfg.label_names})")

    y = _validate_labels(y, cfg.num_labels)
    texts = df["text"].tolist()

    # Train/test split (stratified, random each run)
    import random
    random_state = random.randint(0, 10000)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, y, test_size=cfg.test_size, stratify=y, random_state=random_state
    )
    print(f"[bert] split data: train={len(train_texts)}, test={len(test_texts)} (random_state={random_state})")

    # Create train dataloader
    train_ds = TextLabelDataset(train_texts, train_labels, tokenizer, cfg.max_length)
    sampler = RandomSampler(train_ds, generator=torch.Generator().manual_seed(cfg.seed))
    train_dl = DataLoader(train_ds, sampler=sampler, batch_size=cfg.batch_size)

    device = torch.device(f"cuda:{settings.CUDA_DEVICE}" if (settings.USE_GPU and torch.cuda.is_available()) else "cpu")
    model.to(device)
    print(f"[bert] device: {device}")

    # Training
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    num_training_steps = len(train_dl) * cfg.epochs
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=num_training_steps)

    model.train()
    step = 0
    for epoch in range(cfg.epochs):
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optim.step()
            sched.step()
            optim.zero_grad()
            step += 1
            if step % 20 == 0:
                print(f"[train] epoch={epoch+1}/{cfg.epochs} step={step} loss={loss.item():.4f}")

    # Save model
    out_dir = os.path.join(settings.MODELS_DIR, cfg.output_subdir)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[train] saved model to: {out_dir}")

    # Evaluate on test set
    model.eval()
    preds_idx = []
    batch_size = cfg.batch_size

    with torch.no_grad():
        for i in range(0, len(test_texts), batch_size):
            batch_texts = test_texts[i:i+batch_size]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=cfg.max_length,
                return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            preds = out.logits.argmax(dim=-1).cpu().tolist()
            preds_idx.extend(preds)

    pred_labels = [id2label[p] for p in preds_idx]
    actual_labels = [id2label[l] for l in test_labels]

    # Save results
    from core.data_utils import create_timestamped_output_dir
    result_dir = create_timestamped_output_dir(settings.RESULTS_DIR, f"bert_{cfg.classification_type}")

    result_df = pd.DataFrame({
        "text": test_texts,
        "label": actual_labels,
        "prediction": pred_labels
    })
    result_path = os.path.join(result_dir, "prediction.csv")
    save_csv_safely(result_df, result_path)
    print(f"[bert] saved predictions: {result_path}")

    # Evaluation
    evaluate_results(
        actual_labels=actual_labels,
        predicted_labels=pred_labels,
        output_dir=result_dir,
        task_type=cfg.classification_type
    )


# --------------------------------------------------------------------------------------
# Train (legacy - for external train data)
# --------------------------------------------------------------------------------------
def train_bert_classifier(cfg: BERTClassificationConfig) -> None:
    """Train BERT classifier for paragraph/synthesis (uses full dataset)"""
    set_seed(cfg.seed)
    model_name = _resolve_model_name(cfg)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=cfg.num_labels)

    train_csv = os.path.join(settings.DATA_DIR, cfg.input_filename)
    df = read_csv_safely(train_csv)
    df = process_csv_format(df, cfg.classification_type, ["text", "label"])
    label2id = _label_to_id_map(cfg.label_names)
    y = df["label"].str.strip().str.lower().map(
        {k.lower(): v for k, v in label2id.items()}
    ).fillna(-1).astype(int).tolist()

    if any(v < 0 for v in y):
        bad = df.loc[[i for i, v in enumerate(y) if v < 0], "label"].unique().tolist()
        raise ValueError(f"Unknown labels found: {bad} (expected {cfg.label_names})")

    y = _validate_labels(y, cfg.num_labels)

    ds = TextLabelDataset(df["text"].tolist(), y, tokenizer, cfg.max_length)
    sampler = RandomSampler(ds, generator=torch.Generator().manual_seed(cfg.seed))
    dl = DataLoader(ds, sampler=sampler, batch_size=cfg.batch_size)

    device = torch.device(f"cuda:{settings.CUDA_DEVICE}" if (settings.USE_GPU and torch.cuda.is_available()) else "cpu")
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    num_training_steps = len(dl) * cfg.epochs
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=num_training_steps)

    model.train()
    step = 0
    for epoch in range(cfg.epochs):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optim.step()
            sched.step()
            optim.zero_grad()
            step += 1
            if step % 20 == 0:
                print(f"[train] epoch={epoch+1}/{cfg.epochs} step={step} loss={loss.item():.4f}")

    out_dir = os.path.join(settings.MODELS_DIR, cfg.output_subdir)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[train] saved model to: {out_dir}")


# --------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------
def classify_text(cfg: BERTClassificationConfig, filter_labels: Optional[List[str]] = None) -> None:
    """Classify text CSV with optional label filtering and save predictions"""
    set_seed(cfg.seed)
    model_dir = os.path.join(settings.MODELS_DIR, cfg.output_subdir)
    model_name = model_dir if os.path.isdir(model_dir) else _resolve_model_name(cfg)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=cfg.num_labels)

    infer_csv = os.path.join(settings.DATA_DIR, cfg.input_filename)
    df = read_csv_safely(infer_csv)

    has_gold = ("label" in df.columns)
    if has_gold:
        df = process_csv_format(df, cfg.classification_type, ["id", "text", "label"])
    else:
        df = process_csv_format(df, "classify", ["id", "text"])

    # Filter by labels if specified (for synthesis workflow)
    if filter_labels and "prediction" in df.columns:
        print(f"[filter] applying label filter: {filter_labels}")
        original_len = len(df)
        df = df[df["prediction"].isin(filter_labels)].reset_index(drop=True)
        print(f"[filter] filtered {original_len} -> {len(df)} samples")
    
    if len(df) == 0:
        print("[warning] no samples after filtering")
        return

    device = torch.device(f"cuda:{settings.CUDA_DEVICE}" if (settings.USE_GPU and torch.cuda.is_available()) else "cpu")
    model.to(device)
    model.eval()

    texts = df["text"].tolist()
    preds_idx: list[int] = []
    batch_size = cfg.batch_size

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=cfg.max_length,
                return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            preds = out.logits.argmax(dim=-1).cpu().tolist()
            preds_idx.extend(preds)
            
            if ((i // batch_size + 1) % 10 == 0) or (i + len(batch_texts) >= len(texts)):
                progress = (i + len(batch_texts)) / len(texts) * 100
                print(f"[infer] processed {i+len(batch_texts)}/{len(texts)} ({progress:.1f}%)")

    id2label = {i: n for i, n in enumerate(cfg.label_names)}
    pred_labels = [id2label[p] for p in preds_idx]

    from core.data_utils import create_timestamped_output_dir
    out_dir = create_timestamped_output_dir(settings.RESULTS_DIR, f"bert_{cfg.classification_type}_classify")
    out_path = os.path.join(out_dir, "prediction.csv")
    out_df = pd.DataFrame({"id": df["id"], "text": df["text"], "prediction": pred_labels})
    save_csv_safely(out_df, out_path)
    print(f"[infer] saved predictions: {out_path}")

    if has_gold:
        evaluate_results(
            actual_labels=df["label"].astype(str).str.strip().str.lower().tolist(),
            predicted_labels=pd.Series(pred_labels).astype(str).str.strip().str.lower().tolist(),
            output_dir=out_dir,
            task_type=cfg.classification_type
        )
    else:
        print("[infer] gold labels not provided -> evaluation skipped")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT-based text classification (paragraph/synthesis)")
    parser.add_argument("--mode", required=True, choices=["train", "classify"], 
                       help="operation mode")
    parser.add_argument("--type", required=True, choices=["paragraph", "synthesis"],
                       help="classification type")
    parser.add_argument("--filter_labels", default=None,
                       help="comma-separated labels to filter (for synthesis workflow)")
    args = parser.parse_args()

    cfg = BERTClassificationConfig.create(args.type)
    
    if args.mode == "train":
        train_bert_classifier(cfg)
    
    elif args.mode == "classify":
        filter_list = None
        if args.filter_labels:
            filter_list = [s.strip() for s in args.filter_labels.split(",")]
            print(f"[cli] filter labels: {filter_list}")
        classify_text(cfg, filter_labels=filter_list)