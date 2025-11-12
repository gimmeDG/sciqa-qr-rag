from __future__ import annotations
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

PROJECT_ROOT: Path = Path(__file__).resolve().parent

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


# --------------------------------------------------------------------------------------
# Directories
# --------------------------------------------------------------------------------------
DATA_DIR:    Path = Path(os.getenv("DATA_DIR")    or (PROJECT_ROOT / "data"))
MODELS_DIR:  Path = Path(os.getenv("MODELS_DIR")  or (PROJECT_ROOT / "models"))
RESULTS_DIR: Path = Path(os.getenv("RESULTS_DIR") or (PROJECT_ROOT / "results"))
LOGS_DIR:    Path = Path(os.getenv("LOGS_DIR")    or (PROJECT_ROOT / "logs"))

for _d in (DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


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
RAG_JSON_COLLECTION  = env_required("RAG_JSON_COLLECTION")
RAG_JSON_PERSIST_DIR = env_required("RAG_JSON_PERSIST_DIR", cast=Path)
RAG_JSON_DATA_PATH   = env_required("RAG_JSON_DATA_PATH",   cast=Path)

RAG_HTML_COLLECTION  = env_required("RAG_HTML_COLLECTION")
RAG_HTML_PERSIST_DIR = env_required("RAG_HTML_PERSIST_DIR", cast=Path)
RAG_HTML_DATA_PATH   = env_required("RAG_HTML_DATA_PATH",   cast=Path)

RAG_EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "text-embedding-3-large")
RAG_LLM_MODEL   = os.getenv("RAG_LLM_MODEL",   "gpt-4o")
RAG_INSUFFICIENCY_CHECK_MODEL = os.getenv("RAG_INSUFFICIENCY_CHECK_MODEL", RAG_LLM_MODEL)
