from __future__ import annotations
import os
import re
import json
import pandas as pd
from typing import List, Dict, Any, Tuple

_CANON_TYPES = {
    "target": "Target",
    "precursor": "Precursor",
    "solvent": "Solvent",
    "additive": "Additive",
    "substrate": "Substrate",
}

# Expected raw line format (one entity per line), e.g.:
# {'entity type': 'Target', 'NiFe-LDH', 'NiFe(OH)2', 'Ni, Fe'}
_LINE_RE = re.compile(
    r"""\{\s*['‘"]entity\s*type['’"]\s*:\s*['‘"](?P<etype>[^'’"]+)['’"]\s*,\s*
        ['‘"](?P<entity>[^'’"]*)['’"]\s*,\s*
        ['‘"](?P<formula>[^'’"]*)['’"]\s*,\s*
        ['‘"](?P<metals>[^'’"]*)['’"]\s*\}""",
    re.IGNORECASE | re.VERBOSE,
)

def _canon_entity_type(x: str) -> str:
    if not x:
        return ""
    k = x.strip().lower()
    return _CANON_TYPES.get(k, "")

def _normalize_metals(s: str) -> str:
    if not s:
        return ""
    toks = [t.strip() for t in re.split(r"[;,]", s) if t.strip()]
    seen = set()
    out: List[str] = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return ", ".join(out)

def _normalize_text(s: str) -> str:
    return (s or "").strip()

def _parse_prediction_blob(blob: str) -> List[Dict[str, Any]]:
    """
    Parse multi-line GPT NER output into structured records.
    Lines not matching the entity pattern are ignored.
    'None' (case-insensitive) lines are ignored.
    """
    records: List[Dict[str, Any]] = []
    if not blob or str(blob).strip().lower() == "none":
        return records

    for raw in str(blob).splitlines():
        line = raw.strip()
        if not line or line.lower() == "none":
            continue
        m = _LINE_RE.search(line)
        if not m:
            continue
        et_raw = _canon_entity_type(m.group("etype"))
        if not et_raw:
            # skip unknown entity types
            continue

        entity  = _normalize_text(m.group("entity"))
        formula = _normalize_text(m.group("formula"))
        metals  = _normalize_metals(m.group("metals"))

        # Drop completely empty rows
        if not entity and not formula and not metals:
            continue

        records.append({
            "entity_type": et_raw,
            "entity": entity,
            "formula": formula or entity,  # fall back to entity when formula empty
            "metals": metals,
            "raw_line": line,
        })
    return records

def _dedupe(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[Tuple[str, str, str]] = set()
    out: List[Dict[str, Any]] = []
    for r in records:
        key = (r["entity_type"], r["entity"], r["formula"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

def filter_ner_results(input_csv: str, output_csv: str) -> None:
    """
    Clean, normalize, and de-duplicate raw GPT NER outputs.

    Input CSV requirements:
      - Must contain: 'id' and a prediction column (one of)
        ['prediction', 'pred', 'output', 'gpt_output'].
      - The prediction column can contain multi-line, line-per-entity blocks.

    Output CSV:
      - Long format with one row per (id, entity) record:
        columns = ['id', 'entity_type', 'entity', 'formula', 'metals', 'raw_line', 'aggregated_json']
      - 'aggregated_json' holds all parsed records for the same id (identical JSON per id).

    Behavior:
      1) Parse each line with {_LINE_RE} pattern.
      2) Canonicalize entity_type to one of: Target/Precursor/Solvent/Additive/Substrate.
      3) Normalize whitespaces and metals list; drop 'None' and empty rows.
      4) De-duplicate by (entity_type, entity, formula) within the same id.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.read_csv(input_csv, encoding="utf-8")

    # Detect prediction column
    cand_cols = ["prediction", "pred", "output", "gpt_output"]
    pred_col = next((c for c in cand_cols if c in df.columns), None)
    if pred_col is None:
        raise ValueError(f"Missing prediction column. Expected one of {cand_cols}")

    if "id" not in df.columns:
        raise ValueError("Input CSV must contain an 'id' column.")

    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        _id = row["id"]
        blob = row[pred_col]
        recs = _parse_prediction_blob(blob)
        recs = _dedupe(recs)
        agg_json = json.dumps(recs, ensure_ascii=False)
        if not recs:
            # still emit one row to keep id visible
            rows.append({
                "id": _id,
                "entity_type": "",
                "entity": "",
                "formula": "",
                "metals": "",
                "raw_line": "",
                "aggregated_json": agg_json,
            })
            continue
        for r in recs:
            rows.append({
                "id": _id,
                **r,
                "aggregated_json": agg_json,
            })

    out_df = pd.DataFrame(rows, columns=[
        "id", "entity_type", "entity", "formula", "metals", "raw_line", "aggregated_json"
    ])
    out_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[ner-filter] saved: {output_csv}  (rows={len(out_df)})")
