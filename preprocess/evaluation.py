from __future__ import annotations
import os
import re
import json
import argparse
from typing import Optional, List, Dict, Any, Tuple, Set
import pandas as pd
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _normalize_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def _read_csv_any(path: str, encodings: Tuple[str, ...] = ("utf-8", "cp949", "euc-kr"), **kwargs) -> pd.DataFrame:
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read CSV: {path}\n{last_err}")


# --------------------------------------------------------------------------------------
# Classification evaluation
# --------------------------------------------------------------------------------------
def evaluate_classification(
    pred_csv: str,
    gold_csv: str,
    id_col: str = "id",
    pred_col: str = "prediction",
    gold_col: str = "label",
    label_order: Optional[List[str]] = None,
    save_json: Optional[str] = None,
) -> Dict[str, Any]:
    pred = pd.read_csv(pred_csv, encoding="utf-8")
    gold = pd.read_csv(gold_csv, encoding="utf-8")

    if id_col not in pred.columns or pred_col not in pred.columns:
        raise ValueError(f"[pred] required columns: {id_col}, {pred_col}; got {list(pred.columns)}")
    if id_col not in gold.columns or gold_col not in gold.columns:
        raise ValueError(f"[gold] required columns: {id_col}, {gold_col}; got {list(gold.columns)}")

    df = gold[[id_col, gold_col]].merge(pred[[id_col, pred_col]], on=id_col, how="left")
    y_true = _normalize_series(df[gold_col])
    y_pred = _normalize_series(df[pred_col].fillna(""))

    if label_order is None:
        labels = sorted(set(y_true.unique()) | set(y_pred.unique()))
    else:
        labels = [s.lower() for s in label_order]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\nConfusion Matrix:")
    print(cm)

    cr_text = classification_report(y_true, y_pred, labels=labels, zero_division=0, digits=3)
    print("\nClassification Report:")
    print(cr_text)

    acc = float(accuracy_score(y_true, y_pred))
    prfs = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0, average=None
    )
    macro_prfs = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0, average="macro"
    )
    micro_prfs = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0, average="micro"
    )

    out: Dict[str, Any] = {
        "num_samples": int(len(df)),
        "labels": labels,
        "accuracy": acc,
        "macro": {
            "precision": float(macro_prfs[0]),
            "recall": float(macro_prfs[1]),
            "f1": float(macro_prfs[2]),
        },
        "micro": {
            "precision": float(micro_prfs[0]),
            "recall": float(micro_prfs[1]),
            "f1": float(micro_prfs[2]),
        },
        "classification_report_text": cr_text
    }

    if save_json:
        os.makedirs(os.path.dirname(save_json), exist_ok=True)
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[eval] saved metrics: {save_json}")

    return out


# --------------------------------------------------------------------------------------
# NER evaluation
# --------------------------------------------------------------------------------------
_NER_KEYS = ["target", "precursor", "substrate", "solvent", "additive"]

def _clean_json_like(s: str) -> str:
    s = (s or "").replace("\\xa0", " ").replace("\\", "\\\\")
    s = s.replace("'", '"')
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _gold_labels_from_csv(gold_csv: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Supported formats:
      1) Headerless two columns: [id, labels_json_like]
      2) With headers: id, labels  (labels is a JSON-like list of dicts)
         Example: [{"target": "NiFe-LDH"}, {"precursor": "Ni(NO3)2·6H2O"}, ...]
    """
    try:
        df = _read_csv_any(gold_csv, header=None, usecols=[0, 1])
        df.columns = ["id", "labels"]
    except Exception:
        df = _read_csv_any(gold_csv)
        if not {"id", "labels"}.issubset(set(df.columns)):
            raise ValueError(f"[gold] expected columns: id, labels")

    gold: Dict[str, List[Dict[str, str]]] = {}
    for _, row in df.iterrows():
        ex_id = str(row["id"]).strip()
        raw = _clean_json_like(str(row["labels"]).strip())
        if not raw.startswith("["):
            raw = "[" + raw
        if not raw.endswith("]"):
            raw = raw + "]"
        try:
            items = json.loads(raw)
        except Exception:
            items = []
        norm_items: List[Dict[str, str]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            for k, v in it.items():
                nk = str(k).strip().lower()
                if nk in _NER_KEYS:
                    norm_items.append({nk: str(v).strip()})
        gold[ex_id] = norm_items
    return gold

def _parse_pred_ner_block(pred_text: str) -> List[Dict[str, Any]]:
    """
    Expected lines (GPT NER):
      {'entity type': 'Target', 'NiFe-LDH', 'NiFe(OH)2', 'Ni, Fe'}
      {'entity type': 'Precursor', 'Fe(NO3)3·9H2O', 'Fe(NO3)3·9H2O', 'Fe'}
    Returns list of dicts: {"type": str, "entity": str, "formula": str, "metals": Set[str]}
    """
    if pred_text is None:
        return []
    text = str(pred_text).strip()
    if text.lower() == "none":
        return []

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out = []
    pat = re.compile(
        r"\{'entity type'\s*:\s*'(?P<etype>[^']+)'\s*,\s*'(?P<entity>[^']*)'\s*,\s*'(?P<formula>[^']*)'(?:\s*,\s*'(?P<metals>[^']*)')?\}",
        re.IGNORECASE
    )
    for ln in lines:
        m = pat.search(ln)
        if not m:
            continue
        et = m.group("etype").strip().lower()
        ent = m.group("entity").strip()
        frm = m.group("formula").strip()
        mets = m.group("metals") or ""
        metals = set([t.strip() for t in mets.split(",") if t.strip()]) if mets else set()
        if et in _NER_KEYS:
            out.append({"type": et, "entity": ent, "formula": frm, "metals": metals})
    return out

def _pred_labels_from_csv(pred_csv: str,
                          id_col: str = "id",
                          pred_col: str = "prediction") -> Dict[str, List[Dict[str, str]]]:
    """
    Converts prediction CSV to a dict[id] -> list[{key:value}] aligned with gold format.
    """
    df = _read_csv_any(pred_csv)
    if id_col not in df.columns or pred_col not in df.columns:
        raise ValueError(f"[pred] required columns: {id_col}, {pred_col}; got {list(df.columns)}")

    pred: Dict[str, List[Dict[str, str]]] = {}
    for _, row in df.iterrows():
        ex_id = str(row[id_col]).strip()
        block = _parse_pred_ner_block(str(row[pred_col]))
        pairs: List[Dict[str, str]] = []
        for item in block:
            k = item["type"].lower()
            v = item["entity"].strip()
            if k in _NER_KEYS and v:
                pairs.append({k: v})
        pred[ex_id] = pairs
    return pred


# --------------------------------------------------------------------------------------
# NER evaluation
# --------------------------------------------------------------------------------------
def _entity_sets(items: List[Dict[str, str]]) -> Dict[str, Set[str]]:
    d: Dict[str, Set[str]] = defaultdict(set)
    for it in items:
        for k, v in it.items():
            nk = str(k).strip().lower()
            if nk in _NER_KEYS:
                d[nk].add(str(v).strip().lower())
    return d

def _match_counts(act_set: Set[str], pred_set: Set[str], mode: str) -> Tuple[int, Set[str], Set[str]]:
    if mode == "e":
        tp_items = act_set & pred_set
        return len(tp_items), tp_items, tp_items
    # relaxed substring match
    tp = 0
    matched_a: Set[str] = set()
    matched_p: Set[str] = set()
    for a in act_set:
        for p in pred_set:
            if p in a or a in p:
                tp += 1
                matched_a.add(a)
                matched_p.add(p)
    return tp, matched_a, matched_p

def evaluate_ner(
    pred_csv: str,
    gold_csv: str,
    match_type: str = "r",   # exact match 'e' or relaxed match 'r'
    save_json: Optional[str] = None
) -> Dict[str, Any]:
    gold = _gold_labels_from_csv(gold_csv)
    pred = _pred_labels_from_csv(pred_csv)

    keys = _NER_KEYS
    perf = {k: {"TP": 0, "FP": 0, "FN": 0, "Total": 0} for k in keys}
    common_ids = set(gold.keys()) & set(pred.keys())

    for idx, ex_id in enumerate(sorted(common_ids), start=1):
        gset = _entity_sets(gold[ex_id])
        pset = _entity_sets(pred[ex_id])

        for k in keys:
            perf[k]["Total"] += len(gset.get(k, set()))

            tp, matched_a, matched_p = _match_counts(gset.get(k, set()), pset.get(k, set()), match_type)
            fp_items = pset.get(k, set()) - matched_p
            fn_items = gset.get(k, set()) - matched_a

            perf[k]["TP"] += tp
            perf[k]["FP"] += len(fp_items)
            perf[k]["FN"] += len(fn_items)

            print(f"{idx} - {ex_id} - {k} - FP: {sorted(fp_items)}")
            print(f"{idx} - {ex_id} - {k} - FN: {sorted(fn_items)}")

    total_labels = sum(perf[k]["Total"] for k in keys) or 1
    correct = sum(perf[k]["TP"] for k in keys)
    total_pred_units = sum(perf[k]["TP"] + perf[k]["FP"] + perf[k]["FN"] for k in keys) or 1

    out = {
        "per_type": {},
        "accuracy": correct / total_pred_units,
        "macro_f1": 0.0,
        "weighted_f1": 0.0,
        "match_type": match_type
    }

    macro_f1 = 0.0
    weighted_f1 = 0.0
    for k in keys:
        tp, fp, fn, tot = perf[k]["TP"], perf[k]["FP"], perf[k]["FN"], perf[k]["Total"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        out["per_type"][k] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "total": tot,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
        macro_f1 += f1 / len(keys)
        weighted_f1 += f1 * (tot / total_labels)

        print(f"{k}: Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}, Total={tot}")

    out["macro_f1"] = macro_f1
    out["weighted_f1"] = weighted_f1

    print(f"Accuracy: {out['accuracy']:.3f}")
    print(f"Macro F1: {out['macro_f1']:.3f}")
    print(f"Weighted F1: {out['weighted_f1']:.3f}")

    if save_json:
        os.makedirs(os.path.dirname(save_json), exist_ok=True)
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[ner-eval] saved metrics: {save_json}")

    return out


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Evaluate predictions")
    sub = p.add_subparsers(dest="mode", required=True)

    # classification
    pc = sub.add_parser("cls")
    pc.add_argument("--pred_csv", required=True)
    pc.add_argument("--gold_csv", required=True)
    pc.add_argument("--id_col", default="id")
    pc.add_argument("--pred_col", default="prediction")
    pc.add_argument("--gold_col", default="label")
    pc.add_argument("--labels", nargs="*", default=None)
    pc.add_argument("--save_json", default=None)

    # ner
    pn = sub.add_parser("ner")
    pn.add_argument("--pred_csv", required=True)
    pn.add_argument("--gold_csv", required=True)
    pn.add_argument("--match_type", choices=["e", "r"], default="r")
    pn.add_argument("--save_json", default=None)

    args = p.parse_args()

    if args.mode == "cls":
        evaluate_classification(
            pred_csv=args.pred_csv,
            gold_csv=args.gold_csv,
            id_col=args.id_col,
            pred_col=args.pred_col,
            gold_col=args.gold_col,
            label_order=args.labels,
            save_json=args.save_json
        )
    else:
        evaluate_ner(
            pred_csv=args.pred_csv,
            gold_csv=args.gold_csv,
            match_type=args.match_type,
            save_json=args.save_json
        )

if __name__ == "__main__":
    main()
