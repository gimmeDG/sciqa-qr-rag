from __future__ import annotations
import os
import random
import numpy as np
import pandas as pd
import torch

REQUIRED_COLUMNS = {
    "paragraph": ["text", "label"],      # training format expectation
    "synthesis": ["text", "label"],      # training format expectation
    "classify":  ["id", "text"]          # inference format expectation
}

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"[seed] set to {seed}")

def read_csv_safely(path: str,
                    encoding_candidates=("utf-8", "cp949", "euc-kr"),
                    **kwargs) -> pd.DataFrame:
    last_err = None
    for enc in encoding_candidates:
        try:
            df = pd.read_csv(path, encoding=enc, **kwargs)
            print(f"[io] loaded CSV: {path} (encoding={enc})")
            return df
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read CSV: {path} (tried={encoding_candidates})\n{last_err}")

def save_csv_safely(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        df.to_csv(path, index=False, encoding="utf-8")
    except Exception:
        df.to_csv(path, index=False, encoding="cp949")
    print(f"[io] saved CSV: {path}")

def process_csv_format(df: pd.DataFrame,
                       mode: str,  # "paragraph" | "synthesis" | "classify"
                       require_columns: list[str] | None = None) -> pd.DataFrame:
    cols_needed = require_columns or REQUIRED_COLUMNS.get(mode, [])
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise ValueError(f"[format] missing columns {missing}; required={cols_needed}")
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()
    print(f"[format] mode={mode} columns_ok={cols_needed}")
    return df
