from __future__ import annotations
import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

# --------------------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------------------
def env_required(key: str, cast=lambda x: x, allow_empty: bool = False):
    val = os.getenv(key)
    if val is None or (not allow_empty and str(val).strip() == ""):
        raise RuntimeError(
            f"[CONFIG ERROR] Missing required environment variable: {key}\n"
            f"→ Set it in your environment or .env file.\n"
            f"Example: export {key}=your_value\n"
        )
    try:
        return cast(val)
    except Exception as e:
        raise RuntimeError(f"[CONFIG ERROR] Failed to cast {key}: {e}") from e

def _resolve_path(val: str) -> Path:
    """Resolve a path relative to PROJECT_ROOT (handles ./relative and absolute)."""
    p = Path(val)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


# --------------------------------------------------------------------------------------
# Directories
# --------------------------------------------------------------------------------------
DATA_DIR:    Path = _resolve_path(os.getenv("DATA_DIR")    or "data")
MODELS_DIR:  Path = _resolve_path(os.getenv("MODELS_DIR")  or "models")
RESULTS_DIR: Path = _resolve_path(os.getenv("RESULTS_DIR") or "results")
LOGS_DIR:    Path = _resolve_path(os.getenv("LOGS_DIR")    or "logs")


# --------------------------------------------------------------------------------------
# Device / Seed
# --------------------------------------------------------------------------------------
USE_GPU: bool     = True
CUDA_DEVICE: int  = 0
SEED_DEFAULT: int = 42


# --------------------------------------------------------------------------------------
# API Keys
# --------------------------------------------------------------------------------------
OPENAI_API_KEY: str  = env_required("OPENAI_API_KEY")
OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")


# --------------------------------------------------------------------------------------
# BERT defaults
# --------------------------------------------------------------------------------------
BERT_MODEL_PATH: str = os.getenv("BERT_MODEL_PATH", "matbert-base-uncased")
BERT_BATCH_SIZE: int = int(os.getenv("BERT_BATCH_SIZE", 32))
BERT_LR: float       = float(os.getenv("BERT_LR", 2e-5))
BERT_EPOCHS: int     = int(os.getenv("BERT_EPOCHS", 3))
BERT_MAX_LEN: int    = int(os.getenv("BERT_MAX_LEN", 256))


# --------------------------------------------------------------------------------------
# GPT defaults 
# --------------------------------------------------------------------------------------
GPT_MODEL_NAME: str    = os.getenv("GPT_MODEL_NAME", "gpt-4o")
GPT_TEMPERATURE: float = float(os.getenv("GPT_TEMPERATURE", 0.0))
GPT_MAX_TOKENS: int    = int(os.getenv("GPT_MAX_TOKENS", 1000))


# --------------------------------------------------------------------------------------
# RAG
# --------------------------------------------------------------------------------------
RAG_JSON_COLLECTION   = env_required("RAG_JSON_COLLECTION")
RAG_JSON_PERSIST_DIR  = env_required("RAG_JSON_PERSIST_DIR", cast=_resolve_path)
RAG_JSON_DATA_PATH    = env_required("RAG_JSON_DATA_PATH",   cast=_resolve_path)

RAG_HTML_COLLECTION   = env_required("RAG_HTML_COLLECTION")
RAG_HTML_PERSIST_DIR  = env_required("RAG_HTML_PERSIST_DIR", cast=_resolve_path)
RAG_HTML_DATA_PATH    = env_required("RAG_HTML_DATA_PATH",   cast=_resolve_path)
RAG_HTML_CHUNK_SIZE: int = int(os.getenv("RAG_HTML_CHUNK_SIZE", 300))

# QA evaluation CSV (shared by both JSON and HTML)
RAG_QA_CSV_PATH       = env_required("RAG_QA_CSV_PATH",      cast=_resolve_path)

RAG_EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "text-embedding-3-large")
RAG_EMBED_DIMENSIONS: int = int(os.getenv("RAG_EMBED_DIMENSIONS", 3072))
RAG_LLM_MODEL   = os.getenv("RAG_LLM_MODEL",   "gpt-4o")
RAG_INSUFFICIENCY_CHECK_MODEL = os.getenv("RAG_INSUFFICIENCY_CHECK_MODEL", RAG_LLM_MODEL)
