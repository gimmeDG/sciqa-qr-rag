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


def evaluate_results(actual_labels, predicted_labels):
    cm = confusion_matrix(actual_labels, predicted_labels)
    print("\nConfusion Matrix:")
    print(cm)
    cr = classification_report(actual_labels, predicted_labels, digits=3)
    print("\nClassification Report:")
    print(cr)


# --------------------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------------------
def train_bert_classifier(cfg: BERTClassificationConfig) -> None:
    """Train BERT classifier for paragraph/synthesis"""
    set_seed(cfg.seed)
    model_name = _resolve_model_name(cfg)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=cfg.num_labels)

    train_csv = os.path.join(settings.DATA_DIR, cfg.input_filename)
    df = read_csv_safely(train_csv)
    df = process_csv_format(df, cfg.classification_type, ["text", "label"])
    label2id = _label_to_id_map(cfg.label_names)
    y = df["label"].map(label2id).fillna(-1).astype(int).tolist()
    
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

    out_dir = os.path.join(settings.RESULTS_DIR, cfg.output_subdir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "prediction.csv")
    out_df = pd.DataFrame({"id": df["id"], "text": df["text"], "prediction": pred_labels})
    save_csv_safely(out_df, out_path)
    print(f"[infer] saved predictions: {out_path}")

    if has_gold:
        evaluate_results(
            actual_labels=df["label"].astype(str).str.strip().str.lower().tolist(),
            predicted_labels=pd.Series(pred_labels).astype(str).str.strip().str.lower().tolist()
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