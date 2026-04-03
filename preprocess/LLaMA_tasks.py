"""
Llama 3.3 70B tasks (via Vertex AI Model Garden MaaS)
Uses OpenAI-compatible endpoint with Google Auth ADC.
"""
from __future__ import annotations
import os, re, json, time, argparse
from typing import Literal, List, Dict, Any
from collections import defaultdict
import pandas as pd

import google.auth
import google.auth.transport.requests
from openai import OpenAI
from sklearn.metrics import confusion_matrix, classification_report

from core import settings
from core.config import LlamaConfig
from core.data_utils import read_csv_safely, save_csv_safely

TaskType = Literal["paragraph", "synthesis-method", "ner"]


# --------------------------------------------------------------------------------------
# NER Evaluation Helpers
# --------------------------------------------------------------------------------------
def _clean_json_string(json_str: str) -> str:
    json_str = json_str.replace("\\xa0", " ").replace("\\", "\\\\")
    json_str = json_str.replace("'", '"')
    json_str = json_str.replace("\n", " ").replace("\r", " ")
    json_str = re.sub(r'\s+', ' ', json_str)
    return json_str


def _extract_ner_entities(content: str) -> List[Dict[str, Any]]:
    entity_pattern = r"\{'(?P<type>[^']+)': '(?P<entity>[^']+)', '(?P<formula>[^']+)', '(?P<metal>[^']+)'\}"
    matches = re.finditer(entity_pattern, content)
    extracted = []
    for match in matches:
        metals = [m.strip() for m in match.group("metal").split(",") if m.strip()] if match.group("metal") else []
        extracted.append({
            "type": match.group("type"),
            "entity": match.group("entity"),
            "formula": match.group("formula"),
            "metals": metals,
        })
    return extracted


def _filter_ner_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]] | None:
    targets = [e for e in entities if e['type'].upper() == 'TARGET']
    if len(targets) != 1 or any("MOF" in t['entity'].upper() for t in targets):
        return None
    target_metals = set(targets[0]['metals'])
    filtered = []
    for entity in entities:
        if entity['type'].upper() == 'TARGET' and not entity['metals']:
            continue
        if entity['type'].upper() == 'PRECURSOR':
            if not entity['metals'] or not set(entity['metals']).intersection(target_metals):
                continue
        filtered.append(entity)
    return filtered


def _parse_filtered_to_labels(entities: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    valid_keys = ['target', 'precursor', 'substrate', 'solvent', 'additive']
    return [{e['type'].lower(): e['entity']} for e in entities if e['type'].lower() in valid_keys]


def _extract_actual_ner_labels(label_str: str) -> List[Dict[str, str]]:
    cleaned = _clean_json_string(str(label_str).strip())
    if not cleaned.startswith("[") and not cleaned.endswith("]"):
        cleaned = "[" + cleaned + "]"
    try:
        return json.loads(cleaned)
    except:
        return []


def _evaluate_ner_performance(actual_list: List, predicted_list: List, match_type: str = 'r') -> Dict[str, Any]:
    keys = ['target', 'precursor', 'substrate', 'solvent', 'additive']
    performance = {k: {'TP': 0, 'FP': 0, 'FN': 0, 'Total': 0} for k in keys}

    for actual, predicted in zip(actual_list, predicted_list):
        if actual is None or predicted is None:
            continue
        actual_dict = defaultdict(set)
        predicted_dict = defaultdict(set)

        for item in actual:
            for key, value in item.items():
                nk = key.lower()
                if nk in keys:
                    actual_dict[nk].add(value.lower())
                    performance[nk]['Total'] += 1

        for item in predicted:
            for key, value in item.items():
                nk = key.lower()
                if nk in keys:
                    predicted_dict[nk].add(value.lower())

        for key in keys:
            matched_actual, matched_predicted = set(), set()
            tp = 0
            if match_type == 'e':
                tp = len(actual_dict[key] & predicted_dict[key])
                matched_actual = matched_predicted = actual_dict[key] & predicted_dict[key]
            else:
                for act in actual_dict[key]:
                    for pred in predicted_dict[key]:
                        if pred in act or act in pred:
                            tp += 1
                            matched_actual.add(act)
                            matched_predicted.add(pred)
            performance[key]['TP'] += tp
            performance[key]['FP'] += len(predicted_dict[key] - matched_predicted)
            performance[key]['FN'] += len(actual_dict[key] - matched_actual)

    total_f1, weighted_f1, correct, total_pred = 0, 0, 0, 0
    total_labels = sum(performance[k]['Total'] for k in keys)

    print("\n" + "="*60)
    print("NER Evaluation Results (Relaxed Match)" if match_type == 'r' else "NER Evaluation Results (Exact Match)")
    print("="*60)

    for key in keys:
        tp, fp, fn = performance[key]['TP'], performance[key]['FP'], performance[key]['FN']
        prec = tp / (tp + fp) if tp + fp > 0 else 0
        rec = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
        correct += tp
        total_pred += tp + fp + fn
        total_f1 += f1 / 5
        if total_labels > 0:
            weighted_f1 += f1 * performance[key]['Total'] / total_labels
        print(f"{key}: Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}, Total={performance[key]['Total']}")

    accuracy = correct / total_pred if total_pred > 0 else 0
    print("-"*60)
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Macro F1-Score: {total_f1:.3f}")
    print(f"Weighted F1-Score: {weighted_f1:.3f}")
    print("="*60)

    return {'accuracy': accuracy, 'macro_f1': total_f1, 'weighted_f1': weighted_f1, 'per_class': performance}


# --------------------------------------------------------------------------------------
# Llama Client (OpenAI-compatible with Google Auth)
# --------------------------------------------------------------------------------------
class LlamaClient:
    """OpenAI-compatible client for Llama via Vertex AI MaaS endpoints."""

    def __init__(self, cfg: LlamaConfig):
        self.cfg = cfg
        self.creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self.base_url = (
            f"https://{cfg.location}-aiplatform.googleapis.com/v1beta1/"
            f"projects/{cfg.project_id}/locations/{cfg.location}/endpoints/openapi"
        )
        self._refresh_client()

    def _refresh_client(self):
        """Refresh Google Auth token and recreate OpenAI client."""
        self.creds.refresh(google.auth.transport.requests.Request())
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.creds.token,
        )

    def chat_completion(self, messages: list, **kwargs) -> str:
        """Send chat completion request with auto token refresh."""
        # Check if token needs refresh (expires within 5 minutes)
        if self.creds.expired or (self.creds.expiry and
            (self.creds.expiry.timestamp() - time.time()) < 300):
            self._refresh_client()

        response = self.client.chat.completions.create(
            model=self.cfg.model_id,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.cfg.max_tokens),
            temperature=kwargs.get("temperature", self.cfg.temperature),
        )
        return response.choices[0].message.content


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def _extract_answer_token(text: str) -> str:
    """
    Extract the <label> from phrases like:
    - "Therefore, the answer is <label>."
    - "the answer is <label>"
    Fallback: last token.
    """
    m = re.search(r"the answer is\s+([A-Za-z0-9\-\(\)\/]+)", text or "", re.IGNORECASE)
    if m:
        return m.group(1).strip().lower()
    tok = (text or "").strip().split()
    return (tok[-1] if tok else "").strip().lower()


# --------------------------------------------------------------------------------------
# System prompts (same as GPT_tasks.py)
# --------------------------------------------------------------------------------------
def system_prompt(task: TaskType) -> str:
    if task == "paragraph":
        return (
            "As a catalyst expert in materials science, classify each paragraph into one of the following "
            "four paragraph types: 'synthesis', 'system', 'performance' and 'others'. "
            "Each paragraph should belong to only one category. "
            "Always include a sentence in the format of 'the answer is paragraph type' at the end of the answer."
        )

    elif task == "synthesis-method":
        return (
            "As a catalyst synthesis expert in materials science, classify each paragraph into one of the following "
            "seven synthesis methods: 'electrodeposition', 'sol-gel', 'solid-phase', 'hydro-solvothermal', "
            "'precipitation', 'vapor-phase' and 'others'. "
            "Each paragraph should belong to only one category. "
            "Always include a sentence in the format of 'the answer is synthesis method' at the end of the answer."
        )

    elif task == "ner":
        return (
            "As a catalyst synthesis expert in materials science, label each synthesis entity related to "
            "hydrothermal or solvothermal reactions in the given paragraph as one of the following five entity types: "
            "'Solvent', 'Precursor', 'Target', 'Additive' and 'Substrate. \n"
            "First, identify and extract the primary 'TARGET' material described in a paragraph. Then, identify and extract "
            "the 'PRECURSOR', 'SOLVENT', 'Additive' and 'SUBSTRATE' used in the synthesis of the 'TARGET' material. \n"
            "The 'PRECURSOR' refers only to a metal precursor that provides a metal elemental source to the 'TARGET'. "
            "Each synthesis entity should belong to only one entity type. \n"
            "Always use the answer format: \"{'entity type': 'synthesis entity', 'molecular formula', 'metal components'}\". "
            "If 'molecular formula' is ambiguous or unknown, use 'synthesis entity' as the 'molecular formula'. \n"
            "The 'molecular formula' must represent the complete chemical composition showing ALL atoms in a compound, "
            "not using abbreviated notations such as \"acac\" (acetylacetonate) or any other chemical shorthand. "
            "It is essential to display the exact atomic composition without any abbreviated functional groups or ligands.\n"
            "If the paragraph does not include synthesis entity, the entity in the response should be 'None'."
        )
    else:
        raise ValueError(f"Unknown task type: {task}")


# --------------------------------------------------------------------------------------
# Few-shot examples (imported from GPT_tasks for consistency)
# --------------------------------------------------------------------------------------
def get_paragraph_examples() -> list:
    """Return 10-shot examples for paragraph classification."""
    from preprocess.GPT_tasks import get_paragraph_examples as gpt_examples
    return gpt_examples()


def get_synthesis_examples() -> list:
    """Return 5-shot examples for synthesis method classification."""
    from preprocess.GPT_tasks import get_synthesis_examples as gpt_examples
    return gpt_examples()


def get_ner_examples() -> list:
    """Return 10-shot examples for NER."""
    from preprocess.GPT_tasks import get_ner_examples as gpt_examples
    return gpt_examples()


def get_examples(task: TaskType) -> list:
    """Get examples for the given task type."""
    if task == "paragraph":
        return get_paragraph_examples()
    elif task == "synthesis-method":
        return get_synthesis_examples()
    elif task == "ner":
        return get_ner_examples()
    else:
        raise ValueError(f"Unknown task type: {task}")


# --------------------------------------------------------------------------------------
# Main processing
# --------------------------------------------------------------------------------------
def process_dataset(
    input_file: str,
    output_dir: str,
    task: TaskType,
    cfg: LlamaConfig,
) -> pd.DataFrame:
    """
    Process a CSV file using Llama 3.3 70B (via Vertex AI MaaS).

    Args:
        input_file: Path to input CSV (columns: id, text, label)
        output_dir: Directory to save results
        task: Task type (paragraph, synthesis-method, ner)
        cfg: LlamaConfig instance

    Returns:
        DataFrame with predictions
    """
    os.makedirs(output_dir, exist_ok=True)

    df = read_csv_safely(input_file, header=None, usecols=[0, 1, 2])
    df.columns = ["id", "label", "text"]
    print(f"[llama] Loaded {len(df)} rows from {input_file}")
    print(f"[llama] Task: {task}, Model: {cfg.model_id}")
    print(f"[llama] Project: {cfg.project_id}, Location: {cfg.location}")

    # Initialize client
    client = LlamaClient(cfg)

    # Build system message and examples
    sys_msg = system_prompt(task)
    examples = get_examples(task)

    results = []
    actual_labels_list = []
    predicted_labels_list = []
    t0 = time.time()

    for idx, row in df.iterrows():
        qid, text, label = str(row["id"]), str(row["text"]), str(row["label"])

        # Build messages
        messages = [{"role": "system", "content": sys_msg}]
        messages.extend(examples)
        messages.append({"role": "user", "content": text})

        try:
            response = client.chat_completion(messages)
        except Exception as e:
            print(f"[llama] Error on row {qid}: {e}")
            response = ""

        # Extract prediction
        if task == "ner":
            pred = response  # Keep raw output for NER
            entities = _extract_ner_entities(response)
            filtered = _filter_ner_entities(entities)
            pred_labels = _parse_filtered_to_labels(filtered) if filtered else []
            actual_labels = _extract_actual_ner_labels(label)
            actual_labels_list.append(actual_labels)
            predicted_labels_list.append(pred_labels)
        else:
            pred = _extract_answer_token(response)
            actual_labels_list.append(label.strip().lower())
            predicted_labels_list.append(pred.strip().lower() if pred else "")

        results.append({
            "id": qid,
            "text": text,
            "label": label,
            "prediction": pred,
        })

        if (len(results) % 10) == 0:
            elapsed = time.time() - t0
            avg_time = elapsed / len(results)
            remaining = avg_time * (len(df) - len(results))
            print(f"[llama] {len(results)}/{len(df)} ({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)")

    # Save results
    result_df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, f"llama_{task}_results.csv")
    save_csv_safely(result_df, output_file)
    print(f"[llama] Saved {len(result_df)} predictions to {output_file}")

    elapsed_total = time.time() - t0
    print(f"[llama] Total time: {elapsed_total:.1f}s ({elapsed_total/len(df):.2f}s per sample)")

    # Run evaluation and save to file
    eval_path = os.path.join(output_dir, f"llama_{task}_evaluation.txt")
    if task == "ner":
        eval_results = _evaluate_ner_performance(actual_labels_list, predicted_labels_list, match_type='r')
        with open(eval_path, 'w', encoding='utf-8') as f:
            f.write("NER Evaluation Results (Relaxed Match)\n")
            f.write("="*60 + "\n")
            keys = ['target', 'precursor', 'substrate', 'solvent', 'additive']
            for key in keys:
                perf = eval_results['per_class'][key]
                tp, fp, fn = perf['TP'], perf['FP'], perf['FN']
                prec = tp / (tp + fp) if tp + fp > 0 else 0
                rec = tp / (tp + fn) if tp + fn > 0 else 0
                f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
                f.write(f"{key}: Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}, Total={perf['Total']}\n")
            f.write("-"*60 + "\n")
            f.write(f"Accuracy: {eval_results['accuracy']:.3f}\n")
            f.write(f"Macro F1-Score: {eval_results['macro_f1']:.3f}\n")
            f.write(f"Weighted F1-Score: {eval_results['weighted_f1']:.3f}\n")
    else:
        cm = confusion_matrix(actual_labels_list, predicted_labels_list)
        cr = classification_report(actual_labels_list, predicted_labels_list, digits=3)
        print(f"\nConfusion Matrix:\n{cm}")
        print(f"\nClassification Report:\n{cr}")
        with open(eval_path, 'w', encoding='utf-8') as f:
            f.write(f"Llama Classification Evaluation Results ({task})\n")
            f.write("="*60 + "\n")
            f.write(f"Confusion Matrix:\n{cm}\n\n")
            f.write(f"Classification Report:\n{cr}\n")
    print(f"[llama] evaluation saved: {eval_path}")

    return result_df


# --------------------------------------------------------------------------------------
# CLI entrypoint
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama 3.3 70B Classification/NER (via Vertex AI MaaS)")
    parser.add_argument("--task", type=str, required=True,
                        choices=["paragraph", "synthesis-method", "ner"])
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    cfg = LlamaConfig.create(args.task)
    process_dataset(
        input_file=args.input_csv,
        output_dir=args.output_dir,
        task=args.task,
        cfg=cfg,
    )