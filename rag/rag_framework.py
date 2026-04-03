from __future__ import annotations
import os
import re
import csv
import math
import time
import json
import threading
from typing import List, Dict, Any, Optional, Literal
from pathlib import Path
from collections import defaultdict

from openai import RateLimitError, APITimeoutError, APIConnectionError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from core import settings

_RETRY_EXCEPTIONS = (RateLimitError, APITimeoutError, APIConnectionError)

# ======================================================================================
# Llama Token Usage & Cost Tracking
# ======================================================================================
_LLAMA_USAGE = {"prompt_tokens": 0, "completion_tokens": 0}

# Llama 3.3 70B pricing on Vertex AI (per 1M tokens)
_LLAMA_PRICE_INPUT = 0.27   # $0.27 / 1M input tokens
_LLAMA_PRICE_OUTPUT = 0.28  # $0.28 / 1M output tokens

def reset_llama_usage():
    """Reset Llama token usage counters."""
    global _LLAMA_USAGE
    _LLAMA_USAGE = {"prompt_tokens": 0, "completion_tokens": 0}

def get_llama_usage():
    """Get Llama token usage and estimated cost."""
    p = _LLAMA_USAGE["prompt_tokens"]
    c = _LLAMA_USAGE["completion_tokens"]
    cost_in = (p / 1_000_000) * _LLAMA_PRICE_INPUT
    cost_out = (c / 1_000_000) * _LLAMA_PRICE_OUTPUT
    return {
        "prompt_tokens": p,
        "completion_tokens": c,
        "total_tokens": p + c,
        "cost_input_usd": round(cost_in, 6),
        "cost_output_usd": round(cost_out, 6),
        "cost_total_usd": round(cost_in + cost_out, 6),
    }

def _track_llama_usage(response):
    """Track token usage from Llama API response."""
    global _LLAMA_USAGE
    if hasattr(response, 'usage') and response.usage:
        _LLAMA_USAGE["prompt_tokens"] += getattr(response.usage, 'prompt_tokens', 0) or 0
        _LLAMA_USAGE["completion_tokens"] += getattr(response.usage, 'completion_tokens', 0) or 0

# ======================================================================================
# Optional dependencies for RAGAS evaluation (used only in descriptive mode)
# - Compatible with both ragas 0.2.x (function-based) and 0.3.x (class-based)
# ======================================================================================
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    from ragas.run_config import RunConfig
    try:
        # ragas >= 0.3.x
        from ragas.metrics import ContextRelevance as _ContextRelevanceClass
        _CONTEXT_METRIC = _ContextRelevanceClass()
    except Exception:
        # ragas 0.2.x
        from ragas.metrics import context_relevancy as _context_relevancy_fn
        _CONTEXT_METRIC = _context_relevancy_fn
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    _RAGAS_AVAILABLE = True
except Exception:
    ragas_evaluate = None
    faithfulness = None
    answer_relevancy = None
    _CONTEXT_METRIC = None
    Dataset = None
    _RAGAS_AVAILABLE = False


# ======================================================================================
# Optional OpenAI client (used for GPT-based decomposition / reformulation / answering / re-check)
# ======================================================================================
_OPENAI_OK = False
try:
    from openai import OpenAI
    from core import settings as _settings_for_openai

    _OPENAI_API_KEY = _settings_for_openai.OPENAI_API_KEY
    _OPENAI_OK = bool(_OPENAI_API_KEY)
    _OPENAI_CLIENT = OpenAI(api_key=_OPENAI_API_KEY) if _OPENAI_OK else None
except Exception:
    _OPENAI_OK = False
    _OPENAI_CLIENT = None

# ======================================================================================
# Optional Llama client (via Vertex AI MaaS with Google Auth)
# Always initialize if possible (lazy init on first use)
# ======================================================================================
_LLAMA_OK = False
_LLAMA_CLIENT = None

class _LlamaClientForRAG:
    """Llama client for RAG (OpenAI-compatible with Google Auth)."""
    def __init__(self):
        import google.auth
        import google.auth.transport.requests
        from core.config import LlamaConfig

        self.cfg = LlamaConfig.create("paragraph")
        self.creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self.base_url = (
            f"https://{self.cfg.location}-aiplatform.googleapis.com/v1beta1/"
            f"projects/{self.cfg.project_id}/locations/{self.cfg.location}/endpoints/openapi"
        )
        self._refresh_client()

    def _refresh_client(self):
        import google.auth.transport.requests
        self.creds.refresh(google.auth.transport.requests.Request())
        self.client = OpenAI(base_url=self.base_url, api_key=self.creds.token)

    def chat_completion(self, **kwargs):
        import time as _time
        if self.creds.expired or (self.creds.expiry and
            (self.creds.expiry.timestamp() - _time.time()) < 300):
            self._refresh_client()
        kwargs["model"] = self.cfg.model_id
        return self.client.chat.completions.create(**kwargs)

def _get_llama_client():
    """Lazy initialization of Llama client."""
    global _LLAMA_CLIENT, _LLAMA_OK
    if _LLAMA_CLIENT is None:
        try:
            _LLAMA_CLIENT = _LlamaClientForRAG()
            _LLAMA_OK = True
        except Exception as e:
            print(f"[RAG] Llama client init failed: {e}")
            _LLAMA_OK = False
    return _LLAMA_CLIENT if _LLAMA_OK else None

_RAG_ANSWER_MODEL = os.getenv("RAG_LLM_MODEL", settings.GPT_MODEL_NAME)
_RAG_CHECK_MODEL  = os.getenv("RAG_INSUFFICIENCY_CHECK_MODEL", _RAG_ANSWER_MODEL)

# Global backend selector (can be changed at runtime)
_ACTIVE_BACKEND = "openai"  # "openai" or "llama"

def set_llm_backend(backend: str):
    """Set active LLM backend: 'openai' or 'llama'"""
    global _ACTIVE_BACKEND
    _ACTIVE_BACKEND = backend.lower()
    print(f"[RAG] LLM backend set to: {_ACTIVE_BACKEND}")


# ======================================================================================
# Unified LLM call function (supports OpenAI and Llama backends)
# ======================================================================================
@retry(
    retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
    wait=wait_exponential(multiplier=1, min=1, max=16),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _call_openai_chat(**kwargs):
    # Use Llama backend if configured
    if _ACTIVE_BACKEND == "llama":
        client = _get_llama_client()
        if client is not None:
            # Remove response_format for Llama (not supported)
            kwargs.pop("response_format", None)
            response = client.chat_completion(**kwargs)
            _track_llama_usage(response)
            return response
        print("[RAG] Llama unavailable, falling back to OpenAI")
    return _OPENAI_CLIENT.chat.completions.create(**kwargs)


def _strip_doi_prefix(query: str) -> str:
    """Strip everything up to the first 'that' or 'where' for retrieval."""
    lower = query.lower()
    positions = []
    for kw in (" that ", " where "):
        idx = lower.find(kw)
        if idx != -1:
            positions.append((idx, len(kw)))
    if positions:
        idx, kw_len = min(positions, key=lambda x: x[0])
        return query[idx + kw_len:].strip() or query
    return query


def _normalize_doi(doi_str: str) -> str:
    """Normalize DOI to bare form (e.g. '10.1038/s41467-019-13519-1')."""
    s = doi_str.strip()
    if not s:
        return ""
    s = re.sub(r"^(?:DOI\s*:?\s*)+", "", s, flags=re.IGNORECASE).strip()
    s_lower = s.lower()
    for prefix in ("https://doi.org/", "http://doi.org/",
                    "https://dx.doi.org/", "http://dx.doi.org/",
                    "doi.org/", "doi:"):
        if s_lower.startswith(prefix):
            s = s[len(prefix):]
            break
    m = re.search(r"nature\.com/articles/(.+?)(?:\?.*)?$", s, re.IGNORECASE)
    if m:
        return "10.1038/" + m.group(1).lower()
    return s.lower().strip()


def _normalize_units(text: str) -> str:
    """Normalize unit symbols for consistent matching.

    Handles variations like:
    - dec-1, dec−1, dec–1, dec⁻¹ → dec-1
    - mV/dec, mV dec-1 → mV dec-1
    - °C, ℃ → C
    - cm-2, cm−2, cm⁻² → cm-2
    """
    s = text.strip().lower()
    # Normalize various minus signs to standard hyphen
    s = re.sub(r'[−–—⁻]', '-', s)
    # Normalize superscript numbers
    s = s.replace('⁻¹', '-1').replace('⁻²', '-2')
    s = s.replace('¹', '1').replace('²', '2')
    # Normalize degree symbols
    s = s.replace('°c', 'c').replace('℃', 'c').replace('° c', 'c')
    # Remove extra spaces
    s = re.sub(r'\s+', ' ', s)
    return s


def _doi_match(response: str, expected: str) -> bool:
    """Check if expected answer is found in the response.

    Supports both DOI matching and general value matching with unit normalization.
    """
    if not expected or not response:
        return False

    # Normalize both for comparison
    norm_expected = _normalize_units(expected)
    norm_response = _normalize_units(response)

    # Direct substring match with normalized units
    if norm_expected in norm_response:
        return True

    # Extract numeric value for comparison (e.g., "120" from "120 °C")
    exp_nums = re.findall(r'[\d.]+', norm_expected)
    if exp_nums:
        # Check if the main numeric value appears in response
        main_num = exp_nums[0]
        # Look for the number with similar context (temperature, mV, etc.)
        if main_num in norm_response:
            # Additional validation: check unit context
            exp_has_temp = 'c' in norm_expected or 'k' in norm_expected
            exp_has_mv = 'mv' in norm_expected
            exp_has_dec = 'dec' in norm_expected

            resp_has_temp = 'c' in norm_response or 'k' in norm_response
            resp_has_mv = 'mv' in norm_response
            resp_has_dec = 'dec' in norm_response

            # If same unit type context, consider it a match
            if (exp_has_temp and resp_has_temp) or \
               (exp_has_mv and resp_has_mv) or \
               (exp_has_dec and resp_has_dec):
                # Verify the exact number appears as a standalone value
                if re.search(rf'\b{re.escape(main_num)}\b', norm_response):
                    return True

    # Legacy DOI matching
    norm_doi_expected = _normalize_doi(expected)
    if norm_doi_expected and norm_doi_expected in _normalize_doi(response):
        return True
    for doi in re.findall(r"10\.\d{4,}(?:\.\d+)*/[^\s,;)\]\"']+", response):
        if _normalize_doi(doi) == norm_doi_expected:
            return True
    for url in re.findall(r"nature\.com/articles/[^\s,;)\]\"']+", response):
        if _normalize_doi(url) == norm_doi_expected:
            return True

    return False


# ======================================================================================
# Data I/O helpers
# ======================================================================================
def _read_questions_from_csv(csv_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            q = r.get("question", "") or r.get("query", "")
            a = r.get("answer", "") or r.get("gold", "")
            rows.append({"question": q, "answer": a, **r})
    return rows


# ======================================================================================
# Descriptive-mode parsing & evaluation (RAGAS)
# ======================================================================================
def _extract_response_parts(response: str) -> Dict[str, str]:
    """Extract reference paragraph and answer from LLM response."""
    result = {"context": "", "answer": ""}
    if not response:
        return result

    context_match = re.search(r"References:\s*(.*?)(?=Answer:|$)", response, re.DOTALL)
    if context_match:
        result["context"] = context_match.group(1).strip()

    answer_match = re.search(r"Answer:\s*(.*?)$", response, re.DOTALL)
    if answer_match:
        result["answer"] = answer_match.group(1).strip()
    else:
        parts = response.split("References:")
        if len(parts) > 1:
            after_context = parts[1].strip()
            after_tags = re.split(r"\n(?:References:)", after_context)
            if after_tags:
                result["answer"] = after_tags[-1].strip()
        else:
            result["answer"] = response.strip()

    return result


# ======================================================================================
# Insufficient-information handling (matching reference code exactly)
# ======================================================================================
def _handle_insufficient_info(
    question: str,
    answer: str,
    context: Any,
) -> Optional[Dict[str, float]]:
    """Handle cases with insufficient information or empty answers.

    Sets faithfulness=1.0 for these cases (no hallucination when no info).
    """
    insufficient_keywords = ["insufficient information", "insufficient", "not enough", "정보 부족", "정보부족", "정보 없음"]

    if any(keyword in answer.lower() for keyword in insufficient_keywords) or not answer.strip():
        print(f"    [insufficient-check] Insufficient info detected: faithfulness=1.0")
        return {
            "faithfulness": 1.0,
            "answer_relevancy": 0.0,
            "ContextRelevance": 0.0
        }

    if not context or (isinstance(context, str) and not context.strip()):
        print(f"    [insufficient-check] No context: faithfulness=1.0")
        return {
            "faithfulness": 1.0,
            "answer_relevancy": 0.0,
            "ContextRelevance": 0.0
        }

    return None


# ======================================================================================
# GPT helpers (decomposition / reformulation / answering)
# ======================================================================================
def _gpt_decompose_query(complex_query: str) -> List[str]:
    if not _OPENAI_OK:
        return [complex_query]
    prompt = f"""
You decompose complex scientific questions into multiple concise, retrieval-friendly sub-questions.

Original question:
"{complex_query}"

Decompose into up to 3 independent sub-questions, each targeting a specific aspect. Keep domain terms intact.

Return STRICT JSON:
{{
  "sub_questions": [
    "sub-question 1",
    "sub-question 2",
    "sub-question 3"
  ]
}}
""".strip()
    try:
        completion = _call_openai_chat(
            model=_RAG_ANSWER_MODEL,
            messages=[
                {"role": "system", "content": "Decompose complex scientific queries into concise sub-questions."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        content = completion.choices[0].message.content or "{}"
        data = json.loads(content)
        subs = [s for s in data.get("sub_questions", []) if isinstance(s, str) and s.strip()]
        return subs if subs else [complex_query]
    except Exception:
        return [complex_query]

def _gpt_rewrite_query(query: str) -> str:
    if not _OPENAI_OK:
        return query
    prompt = f"""
Extract the essential keywords and domain terms from the following question and rewrite it as a concise keyword-focused search query.

Question:
"{query}"

Keep core technical terms, remove function words, and make it retrieval-friendly.

Return STRICT JSON:
{{
  "rewritten_query": "..."
}}
""".strip()
    try:
        completion = _call_openai_chat(
            model=_RAG_ANSWER_MODEL,
            messages=[
                {"role": "system", "content": "Rewrite questions into keyword-focused search queries."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        data = json.loads(completion.choices[0].message.content or "{}")
        rq = data.get("rewritten_query", "") or query
        return rq
    except Exception:
        return query

def _gpt_generate_basic_answer(query: str, context_blocks: List[str]) -> str:
    """Generate answer using the same prompt format as reference code."""
    joined = "\n\n====================\n\n".join(context_blocks or [])
    # DEBUG: Check if context is empty
    print(f"    [DEBUG] context_blocks count: {len(context_blocks or [])}, joined length: {len(joined)}")
    if context_blocks:
        print(f"    [DEBUG] First block preview: {(context_blocks[0][:200] + '...') if len(context_blocks[0]) > 200 else context_blocks[0]}")
    if not _OPENAI_OK:
        return "Insufficient information"
    # System prompt matching reference code
    system = "You are an expert assistant analyzing scientific literature to answer questions. Answer based ONLY on the provided information, do not guess if you are unsure."
    # User prompt matching reference code format
    user = f"""Below is document information related to the user's question:

{joined}

Based on the information above, please answer the following question.
If the information is sufficient, provide the reference paragraphs along with your answer. Only output the paragraphs that were directly used to formulate the answer.
If there is not enough information, honestly answer "Insufficient information".

"Answer format:
References:
Answer:
"

User question: {{query}}"""
    try:
        resp = _call_openai_chat(
            model=_RAG_ANSWER_MODEL,
            temperature=0,
            max_tokens=3000,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or "Insufficient information"
    except Exception:
        return "Insufficient information"

def _gpt_generate_advanced_answer(
    query: str,
    sub_queries: List[str],
    paragraphs: List[Dict[str, Any]],
    base_rag: Optional[_BaseRAG] = None,
) -> Tuple[str, List[str]]:
    """Returns (answer, context_parts_used_for_llm). Uses same prompt format as reference code."""
    processed_doc_ids = set()
    context_parts: List[str] = []
    fetch_full = getattr(base_rag, "_fetch_full_document", None) if base_rag else None
    for para in sorted(paragraphs, key=lambda x: x.get("score", 0), reverse=True):
        doc_id = para.get("doc_id")
        if not doc_id or doc_id in processed_doc_ids:
            continue
        processed_doc_ids.add(doc_id)
        # try full document first
        full_text = fetch_full(doc_id) if callable(fetch_full) else ""
        if full_text:
            chunk = full_text
        else:
            preview = para.get("text", "")
            section = para.get("section", "")
            pidx = para.get("paragraph_idx", "")
            header = f"[DocID: {doc_id}] [Section: {section}] [ParagraphIdx: {pidx}]"
            chunk = f"{header}\n{preview}"
        context_parts.append(chunk)
    joined = "\n\n====================\n\n".join(context_parts)
    if not _OPENAI_OK:
        return "Insufficient information", context_parts
    # System prompt matching reference code
    system = "You are an expert assistant analyzing scientific literature to answer questions. Answer based ONLY on the provided information, do not guess if you are unsure."
    # Subqueries text matching reference code
    subqueries_text = "This question was decomposed into the following sub-queries for retrieval:\n"
    for i, sq in enumerate(sub_queries, 1):
        subqueries_text += f"{i}. {sq}\n"
    subqueries_text += "\nSynthesize information relevant to each sub-query to provide the final answer.\n\n"
    # User prompt matching reference code format
    user = f"""Below is document information related to the user's question:

{joined}

{subqueries_text}Based on the information above, please answer the following question.
If the information is sufficient, provide the reference paragraphs along with your answer. Only output the paragraphs that were directly used to formulate the answer.
If there is not enough information, honestly answer "Insufficient information".

"Answer format:
References:
Answer:
"

User question: {{query}}"""
    try:
        resp = _call_openai_chat(
            model=_RAG_ANSWER_MODEL,
            temperature=0,
            max_tokens=3000,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or "Insufficient information", context_parts
    except Exception:
        return "Insufficient information", context_parts


# ======================================================================================
# RAGAS single-sample evaluation (version-compatible, matching reference code)
# ======================================================================================
def _ragas_eval(question: str, answer: str, context: Any) -> Dict[str, float]:
    """Perform RAGAS evaluation. Checks for insufficient info first."""
    print(f"    [RAGAS] Running evaluation...")

    insufficient_result = _handle_insufficient_info(question, answer, context)
    if insufficient_result is not None:
        return insufficient_result

    if not _RAGAS_AVAILABLE or ragas_evaluate is None or Dataset is None or _CONTEXT_METRIC is None:
        return {"error": "RAGAS dependencies are not available."}

    # Helper: contexts를 List[List[str]]로 표준화
    def _to_nested_list(ctx: Any) -> List[List[str]]:
        if isinstance(ctx, str):
            return [[ctx]]
        if isinstance(ctx, list):
            if not ctx:
                return [[]]
            if isinstance(ctx[0], str):
                return [ctx]
            if isinstance(ctx[0], list):
                return ctx
        return [[str(ctx)]]

    # Helper: 다양한 형식의 점수를 float로 추출 (NaN 처리)
    def _to_float(val: Any) -> float:
        try:
            if isinstance(val, (int, float)):
                return 0.0 if math.isnan(val) else float(val)
            if isinstance(val, list) and val:
                x = float(val[0])
                return 0.0 if math.isnan(x) else x
            if isinstance(val, dict) and "score" in val:
                x = float(val["score"])
                return 0.0 if math.isnan(x) else x
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    try:
        answer = answer or "Insufficient information"
        contexts_nested = _to_nested_list(context or "Insufficient information")

        ds = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": contexts_nested,
        })

        # Use explicit LLM and embeddings (consistent with reference code)
        llm = ChatOpenAI(model=_RAG_ANSWER_MODEL, temperature=0)
        embeddings = OpenAIEmbeddings()
        metrics = [faithfulness, answer_relevancy, _CONTEXT_METRIC]

        raw = ragas_evaluate(
            ds,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            run_config=RunConfig(timeout=90),
        )

        # Handle different RAGAS return types (matching reference code)
        if isinstance(raw, dict):
            raw_dict = raw
        elif hasattr(raw, "metric_results"):
            raw_dict = {mr.metric.name: mr.score for mr in raw.metric_results}
        elif hasattr(raw, "scores"):
            scores = raw.scores
            if isinstance(scores, dict):
                raw_dict = scores
            else:  # list
                raw_dict = {
                    metrics[i].name if hasattr(metrics[i], "name")
                    else metrics[i].__name__: scores[i]
                    for i in range(len(scores))
                }
        elif hasattr(raw, "to_pandas"):
            # RAGAS 0.2+ returns EvaluationResult with to_pandas()
            df = raw.to_pandas()
            raw_dict = df.iloc[0].to_dict() if len(df) > 0 else {}
        else:
            raise ValueError("Unknown RAGAS result format")

        # Metric name standardization (matching reference code)
        rename = {
            "context_relevance":    "ContextRelevance",
            "context_relevancy":    "ContextRelevance",
            "context_precision":    "ContextRelevance",
            "nv_context_relevance": "ContextRelevance",
            "ContextRelevance":     "ContextRelevance",
            "faithfulness":         "faithfulness",
            "Faithfulness":         "faithfulness",
            "answer_relevancy":     "answer_relevancy",
            "answer_relevance":     "answer_relevancy",
        }

        results = {}
        for k, v in raw_dict.items():
            # Handle sub-metric dict (no 'score' key)
            if isinstance(v, dict) and "score" not in v:
                for sub_k, sub_v in v.items():
                    std_k = rename.get(sub_k, sub_k)
                    results[std_k] = _to_float(sub_v)
            else:
                std_k = rename.get(k, k)
                results[std_k] = _to_float(v)

        # Fill missing required metrics with 0.0
        for req in ("faithfulness", "answer_relevancy", "ContextRelevance"):
            results.setdefault(req, 0.0)

        # Handle NaN faithfulness as 1.0 (no hallucination if no info)
        if math.isnan(results.get("faithfulness", 0.0)):
            print(f"    [RAGAS] faithfulness NaN -> 1.0")
            results["faithfulness"] = 1.0

        return results

    except Exception as e:
        print(f"    [RAGAS] Error: {e}")
        return {
            "faithfulness": 1.0,
            "answer_relevancy": 0.0,
            "ContextRelevance": 0.0,
            "error": str(e)
        }


# ======================================================================================
# Base RAG implementations
# ======================================================================================
class _BaseRAG:
    """Common interface for RAG backends."""

    def __init__(self):
        self._para_cache: Dict[str, str] = {}
        self._meta_cache: Dict[str, Dict[str, Any]] = {}

    def _dense_retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def _sparse_retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """BM25-based retrieval (optional)."""
        return []

    def hybrid_retrieve(self, query: str, top_k: int = 5, bm25_weight: float = 0.5) -> List[Dict[str, Any]]:
        """Dense + BM25 hybrid with normalized weighted fusion."""
        dense = self._dense_retrieve(query, top_k=max(10, top_k * 2))
        sparse = self._sparse_retrieve(query, top_k=max(10, top_k * 3))

        # Collect raw scores per doc_id
        by_doc: Dict[str, Dict[str, Any]] = {}

        for it in dense:
            did = it.get("doc_id")
            if not did:
                continue
            d = by_doc.setdefault(did, {
                "doc_id": did, "text": it.get("text", ""),
                "section": it.get("section", ""),
                "paragraph_idx": it.get("paragraph_idx", -1),
                "raw_vector": 0.0, "raw_bm25": 0.0,
            })
            d["raw_vector"] = max(d["raw_vector"], float(it.get("score", 0.0)))
            if not d.get("text") and it.get("text"):
                d["text"] = it["text"]
            self._meta_cache[did] = {
                "section": it.get("section", ""),
                "paragraph_idx": it.get("paragraph_idx", -1),
            }

        for it in sparse:
            did = it.get("doc_id")
            if not did:
                continue
            d = by_doc.setdefault(did, {
                "doc_id": did, "text": it.get("text", ""),
                "section": it.get("section", ""),
                "paragraph_idx": it.get("paragraph_idx", -1),
                "raw_vector": 0.0, "raw_bm25": 0.0,
            })
            d["raw_bm25"] = max(d["raw_bm25"], float(it.get("score", 0.0)))
            if not d.get("text") and it.get("text"):
                d["text"] = it["text"]
            self._meta_cache[did] = {
                "section": it.get("section", ""),
                "paragraph_idx": it.get("paragraph_idx", -1),
            }

        # Normalize by max score
        max_v = max((d["raw_vector"] for d in by_doc.values()), default=1.0) or 1.0
        max_b = max((d["raw_bm25"] for d in by_doc.values()), default=1.0) or 1.0

        fused: List[Dict[str, Any]] = []
        for did, d in by_doc.items():
            nv = (d["raw_vector"] / max_v) * (1 - bm25_weight)
            nb = (d["raw_bm25"] / max_b) * bm25_weight
            fused.append({
                "doc_id": did,
                "text": d.get("text", ""),
                "section": d.get("section", ""),
                "paragraph_idx": d.get("paragraph_idx", -1),
                "score": nv + nb,
            })

        fused.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return fused[:top_k]

    def _build_context_blocks(self, doc_ids: List[str]) -> List[str]:
        blocks: List[str] = []
        fetch_full = getattr(self, "_fetch_full_document", None)
        for did in doc_ids:
            if callable(fetch_full):
                full_text = fetch_full(did)
                if full_text:
                    blocks.append(full_text)
                    continue
            # fallback: use cached single chunk
            txt = self._para_cache.get(did)
            if txt:
                meta = self._meta_cache.get(did, {})
                header = f"[DocID: {did}] [Section: {meta.get('section','')}] [ParagraphIdx: {meta.get('paragraph_idx','')}]"
                blocks.append(f"{header}\n{txt}")
        return blocks

    def _llm_answer(self, query: str, blocks: List[str], strict_unknown: bool = True) -> str:
        return _gpt_generate_basic_answer(query, blocks)

    def answer(self, query: str, top_k: int = 5, rewritten: Optional[str] = None) -> tuple[str, List[Dict[str, Any]], List[str]]:
        """Returns (response, retrieved_paragraphs, context_blocks_used_for_llm)."""
        q = rewritten if rewritten else query
        paras = self._dense_retrieve(q, top_k=top_k)
        doc_ids = list({p.get("doc_id") for p in paras if p.get("doc_id")})
        blocks = self._build_context_blocks(doc_ids)
        resp = self._llm_answer(query, blocks, strict_unknown=True)
        return resp, paras, blocks


# ======================================================================================
# JSON / HTML backends
#   C-RAG: dense-only
#   QR-RAG: hybrid via BM25 + dense (used by wrapper through .hybrid_retrieve)
# ======================================================================================
class _ChromaMixin:
    """Shared ChromaDB helpers."""

    def _init_chroma(self, name: str, persist_dir: str, api_key: str, embed_model: str,
                     data_path: str = "", data_format: str = "json"):
        import chromadb
        from chromadb.utils import embedding_functions

        self._chroma = chromadb.PersistentClient(path=str(persist_dir))
        self._embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key, model_name=embed_model
        )
        self._collection = self._chroma.get_collection(
            name=name, embedding_function=self._embedding_fn
        )
        self._full_doc_cache: Dict[str, str] = {}
        self._data_path: str = str(Path(data_path).resolve()) if data_path else ""
        self._data_format: str = data_format

    def _collect_to_paragraphs(self, results, top_k: int) -> List[Dict[str, Any]]:
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        seen = set()
        out: List[Dict[str, Any]] = []
        for doc, meta, dist in zip(docs, metas, dists):
            did = meta.get("docId", "") or meta.get("id", "")
            if not did or did in seen:
                continue
            seen.add(did)
            score = 1 - float(dist)
            section = meta.get("section", "")
            pidx = meta.get("paragraphIdx", meta.get("paragraphId", ""))
            out.append({
                "score": score,
                "text": doc,
                "doc_id": did,
                "section": section,
                "paragraph_idx": pidx
            })
            self._para_cache[did] = doc
            self._meta_cache[did] = {"section": section, "paragraph_idx": pidx}
            if len(out) >= top_k:
                break
        return out

    def _ensure_bm25(self):
        if getattr(self, "_bm25_ready", False):
            return
        try:
            total = self._collection.count()
            batch_size = 5000
            all_docs: List[str] = []
            all_meta: List[Dict[str, Any]] = []
            for offset in range(0, total, batch_size):
                batch = self._collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=["documents", "metadatas"]
                )
                if batch and "documents" in batch and batch["documents"]:
                    all_docs.extend(batch["documents"])
                    all_meta.extend(batch["metadatas"])
            self._bm25_texts = all_docs
            self._bm25_ids = []
            self._bm25_info = []
            for i, m in enumerate(all_meta):
                did = m.get("docId", f"doc_{i}")
                self._bm25_ids.append(did)
                self._bm25_info.append({
                    "paragraphId": m.get("paragraphId", ""),
                    "section": m.get("section", ""),
                    "paragraphIdx": m.get("paragraphIdx", -1),
                })
            from rank_bm25 import BM25Okapi
            tokenized = [doc.lower().split() for doc in self._bm25_texts]
            self._bm25 = BM25Okapi(tokenized)
            self._bm25_ready = True
        except Exception:
            self._bm25 = None
            self._bm25_ready = False

    def _sparse_retrieve_impl(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        self._ensure_bm25()
        if not getattr(self, "_bm25_ready", False) or not self._bm25:
            return []
        import numpy as np
        q = query.lower().split()
        scores = self._bm25.get_scores(q)
        idxs = np.argsort(scores)[-max(top_k*3, 30):][::-1]
        seen = set()
        out: List[Dict[str, Any]] = []
        max_s = float(np.max(scores)) if len(scores) else 1.0
        if max_s <= 0:
            max_s = 1.0
        for i in idxs:
            did = self._bm25_ids[i]
            if did in seen:
                continue
            seen.add(did)
            raw = self._bm25_info[i]
            section = raw.get("section", "")
            pidx = raw.get("paragraphIdx", -1)
            text = self._bm25_texts[i]
            self._para_cache[did] = text
            self._meta_cache[did] = {"section": section, "paragraph_idx": pidx}
            out.append({
                "score": float(scores[i] / max_s),
                "text": text,
                "doc_id": did,
                "section": section,
                "paragraph_idx": pidx
            })
            if len(out) >= top_k:
                break
        return out

    def _fetch_full_document(self, doc_id: str) -> str:
        """Load the original document file from disk (JSON or HTML)."""
        if doc_id in self._full_doc_cache:
            return self._full_doc_cache[doc_id]
        if not self._data_path:
            self._full_doc_cache[doc_id] = ""
            return ""
        try:
            if self._data_format == "json":
                file_path = os.path.join(self._data_path, f"{doc_id}.json")
                if not os.path.exists(file_path):
                    self._full_doc_cache[doc_id] = ""
                    return ""
                with open(file_path, "r", encoding="utf-8") as f:
                    doc_data = json.load(f)
                # Remove embeddings to save tokens
                if "embeddings" in doc_data:
                    del doc_data["embeddings"]
                full_text = self._format_json_document(doc_id, doc_data)
            else:  # html
                file_path = os.path.join(self._data_path, f"{doc_id}.html")
                if not os.path.exists(file_path):
                    self._full_doc_cache[doc_id] = ""
                    return ""
                from bs4 import BeautifulSoup
                with open(file_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                soup = BeautifulSoup(html_content, "html.parser")
                full_text = soup.get_text(separator=" ", strip=True)
            self._full_doc_cache[doc_id] = full_text
            return full_text
        except Exception:
            self._full_doc_cache[doc_id] = ""
            return ""

    @staticmethod
    def _format_json_document(doc_id: str, doc_data: dict) -> str:
        """Format JSON document with metadata + section-based text."""
        metadata = doc_data.get("metadata", {})
        title = metadata.get("title", "")
        authors = metadata.get("authors", [])
        doi = metadata.get("doi", "")
        journal = metadata.get("journal", "")
        year = metadata.get("year", "")
        authors_str = ", ".join(authors) if authors else ""

        sections: List[str] = []
        content = doc_data.get("content", {})

        # Abstract
        if "abstract" in content:
            abstract = content["abstract"]
            if isinstance(abstract, dict):
                for atype, atext in abstract.items():
                    if atext and isinstance(atext, str):
                        sections.append(f"Abstract ({atype}): {atext}")
            elif isinstance(abstract, str):
                sections.append(f"Abstract: {abstract}")

        # Introduction
        if "introduction" in content and content["introduction"]:
            sections.append(f"Introduction: {content['introduction']}")

        # Experimental Section
        if "experimentalSection" in content:
            exp = content["experimentalSection"]
            if isinstance(exp, dict):
                for stype, sitems in exp.items():
                    if isinstance(sitems, list):
                        for item in sitems:
                            if isinstance(item, dict) and "content" in item:
                                sc = item["content"]
                                st = item.get("synthesisType", "")
                                if sc:
                                    sections.append(
                                        f"Experimental - {stype} ({st if st else 'general'}): {sc}"
                                    )
                    elif isinstance(sitems, str) and sitems:
                        sections.append(f"Experimental - {stype}: {sitems}")
            elif isinstance(exp, str) and exp:
                sections.append(f"Experimental: {exp}")

        # Results and Discussion
        if "resultsAndDiscussion" in content and content["resultsAndDiscussion"]:
            sections.append(f"Results and Discussion: {content['resultsAndDiscussion']}")

        # Conclusion
        if "conclusion" in content and content["conclusion"]:
            sections.append(f"Conclusion: {content['conclusion']}")

        header = (
            f"Document ID: {doc_id}\n"
            f"Title: {title}\n"
            f"Authors: {authors_str}\n"
            f"DOI: {doi}\n"
            f"Journal: {journal}\n"
            f"Year: {year}\n"
        )
        return header + "\n" + "\n\n".join(sections)

class JSONCRAG(_ChromaMixin, _BaseRAG):
    """Conventional RAG on JSON corpus: dense-only retrieval."""
    def __init__(self):
        super().__init__()
        print(f"[JSONCRAG] Initializing with collection={settings.RAG_JSON_COLLECTION}, "
              f"persist_dir={settings.RAG_JSON_PERSIST_DIR}")
        self._init_chroma(
            settings.RAG_JSON_COLLECTION,
            str(settings.RAG_JSON_PERSIST_DIR),
            settings.OPENAI_API_KEY,
            settings.RAG_EMBED_MODEL,
            data_path=str(settings.RAG_JSON_DATA_PATH),
            data_format="json",
        )
        print(f"[JSONCRAG] Collection count: {self._collection.count()}")

    def _dense_retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        res = self._collection.query(
            query_texts=[query],
            n_results=max(10, top_k * 2),
            include=["documents", "metadatas", "distances"],
        )
        return self._collect_to_paragraphs(res, top_k)

    def _sparse_retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self._sparse_retrieve_impl(query, top_k)

class HTMLCRAG(_ChromaMixin, _BaseRAG):
    """Conventional RAG on HTML corpus: dense-only retrieval."""
    def __init__(self):
        super().__init__()
        self._init_chroma(
            settings.RAG_HTML_COLLECTION,
            str(settings.RAG_HTML_PERSIST_DIR),
            settings.OPENAI_API_KEY,
            settings.RAG_EMBED_MODEL,
            data_path=str(settings.RAG_HTML_DATA_PATH),
            data_format="html",
        )

    def _dense_retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        res = self._collection.query(
            query_texts=[query],
            n_results=max(10, top_k * 2),
            include=["documents", "metadatas", "distances"],
        )
        return self._collect_to_paragraphs(res, top_k)

    def _sparse_retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self._sparse_retrieve_impl(query, top_k)


# ======================================================================================
# QR-RAG wrapper (decomposition + query reformulation + hybrid retrieval)
# ======================================================================================
class QRRAGWrapper:
    """Query-Reformulation RAG with decomposition and hybrid retrieval."""

    def __init__(self, base: _BaseRAG, fmt: Literal["json", "html"]):
        self.base = base
        self.fmt = fmt

    def answer(self, query: str, top_k: int = 5, rewritten: Optional[str] = None) -> tuple[str, List[Dict[str, Any]], List[str]]:
        """Returns (response, retrieved_paragraphs, context_blocks_used_for_llm)."""
        q = rewritten if rewritten else query
        paras = self.base.hybrid_retrieve(q, top_k=top_k, bm25_weight=0.5)
        doc_ids = list({p.get("doc_id") for p in paras if p.get("doc_id")})
        blocks = self.base._build_context_blocks(doc_ids)
        resp = self.base._llm_answer(query, blocks, strict_unknown=True)
        return resp, paras, blocks


# ======================================================================================
# Public Runner
# ======================================================================================
class UnifiedRAG:
    """
    Retrieval:
      - 'c-rag'  : Conventional RAG (dense-only)
      - 'qr-rag' : Query-Reformulation RAG (dense + BM25 hybrid, decomposition second-pass)
    Evaluation:
      - 'doi'         : exact DOI accuracy
      - 'descriptive' : RAGAS metrics
    """

    def __init__(
        self,
        dataset_format: Literal["json", "html"] = "json",
        retrieval: Literal["c-rag", "qr-rag", "baseline", "advanced"] = "c-rag",
        csv_path: Optional[str] = None,
    ):
        self.dataset_format = dataset_format
        retrieval = {"baseline": "c-rag", "advanced": "qr-rag"}.get(retrieval, retrieval)
        self.retrieval = retrieval
        if dataset_format == "json":
            self._base = JSONCRAG()
        else:
            self._base = HTMLCRAG()
        self.csv_path = csv_path or settings.RAG_QA_CSV_PATH
        self._runner = QRRAGWrapper(self._base, dataset_format) if retrieval == "qr-rag" else self._base

    @staticmethod
    def _detect_insufficiency(response: str) -> bool:
        insufficient_keywords = [
            "insufficient information",
            "not enough information",
            "information not available",
            "cannot determine",
            "unknown",
            "insufficient",
        ]
        ans = (response or "").strip().lower()
        if not ans:
            return True
        if any(k in ans for k in insufficient_keywords):
            return True

        if _OPENAI_OK:
            try:
                prompt = (
                    "Return STRICT JSON: {\"is_insufficient\": true/false}.\n\n"
                    f"Proposed Answer:\n{response[:4000]}\n\n"
                    "Decision rule:\n"
                    "- true  if the answer says it cannot find the information or doesn't know.\n"
                    "- false if the answer provides a concrete response.\n"
                )
                resp_chk = _call_openai_chat(
                    model=_RAG_CHECK_MODEL,
                    temperature=0,
                    max_tokens=64,
                    messages=[
                        {"role": "system", "content": "You return strict JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    top_p=1
                )
                content = (resp_chk.choices[0].message.content or "").strip()
                if "true" in content.lower().replace(" ", ""):
                    return True
            except Exception:
                pass
        return False

    @staticmethod
    def _format_retrieved_paragraphs(retrieved: List[Dict[str, Any]], gold_answer: str = "") -> Dict[str, Any]:
        """Format retrieved paragraphs for analysis-friendly output."""
        formatted = []
        gold_doc_id = ""
        gold_rank = -1

        # Extract gold doc_id from DOI if possible
        if gold_answer:
            norm_gold = _normalize_doi(gold_answer)

        for rank, para in enumerate(retrieved, 1):
            doc_id = para.get("doc_id", "")
            score = para.get("score", 0.0)
            text = para.get("text", "")
            section = para.get("section", "")
            para_idx = para.get("paragraph_idx", -1)

            # Check if this document contains the gold DOI
            is_gold = False
            if gold_answer and doc_id:
                # Check if doc_id matches or contains the gold DOI pattern
                if _normalize_doi(doc_id) == norm_gold or norm_gold in doc_id.lower():
                    is_gold = True
                    gold_rank = rank

            formatted.append({
                "rank": rank,
                "doc_id": doc_id,
                "similarity_score": round(score, 4),
                "section": section,
                "paragraph_idx": para_idx,
                "text": text,
                "text_length": len(text),
                "is_gold_document": is_gold,
            })

        return {
            "top_k_paragraphs": formatted,
            "gold_in_top_k": gold_rank > 0,
            "gold_rank": gold_rank if gold_rank > 0 else None,
            "unique_documents": list({p.get("doc_id") for p in retrieved if p.get("doc_id")}),
            "num_unique_documents": len({p.get("doc_id") for p in retrieved if p.get("doc_id")}),
        }

    def _process_one_crag(self, query: str, answer: str, top_k: int, eval_mode: str) -> Dict[str, Any]:
        start = time.time()
        # answer() now returns (response, retrieved_paragraphs, context_blocks)
        resp, retrieved, context_blocks = self._base.answer(query, top_k=top_k)

        # Format retrieval results
        retrieval_info = self._format_retrieved_paragraphs(retrieved, answer)

        item: Dict[str, Any] = {
            "question": query,
            "expected_answer": answer,
            "response": resp,
            "retrieval": retrieval_info,
        }

        if eval_mode == "doi":
            is_correct = _doi_match(resp or "", answer or "")
            item["evaluation"] = {
                "is_correct": is_correct,
                "gold_in_top_k": retrieval_info["gold_in_top_k"],
                "gold_rank": retrieval_info["gold_rank"],
            }
        else:
            # Extract '참고 문단' and '답변' from LLM response (matching reference code)
            response_parts = _extract_response_parts(resp)
            extracted_answer = response_parts["answer"]
            extracted_context = response_parts["context"]

            # RAGAS 평가 수행 (using extracted answer/context like reference code)
            ragas_scores = _ragas_eval(query, extracted_answer, extracted_context)
            item["extracted_answer"] = extracted_answer
            item["context"] = extracted_context
            item["ragas_results"] = ragas_scores

        item["execution_time_sec"] = round(time.time() - start, 2)
        return item

    def _process_one_qrrag(self, query: str, answer: str, top_k: int, eval_mode: str) -> Dict[str, Any]:
        start = time.time()
        search_query = _strip_doi_prefix(query)
        rewritten = _gpt_rewrite_query(search_query)
        # answer() now returns (response, retrieved_paragraphs, context_blocks)
        resp, retrieved, context_blocks = self._runner.answer(query, top_k=top_k, rewritten=rewritten)

        # Format initial retrieval results
        retrieval_info = self._format_retrieved_paragraphs(retrieved, answer)

        item: Dict[str, Any] = {
            "question": query,
            "expected_answer": answer,
            "response": resp,
            "query_reformulation": {
                "original_query": query,
                "search_query": search_query,
                "rewritten_query": rewritten,
            },
            "retrieval": retrieval_info,
            "decomposition": {
                "is_basic_sufficient": True,
                "is_decomposed": False,
                "sub_queries": None,
            },
        }

        # Track which context_blocks to use for RAGAS (may be updated if decomposition happens)
        ragas_context = context_blocks

        is_insufficient = self._detect_insufficiency(resp)
        item["decomposition"]["is_basic_sufficient"] = not is_insufficient

        if is_insufficient:
            sub_queries = _gpt_decompose_query(search_query)
            if len(sub_queries) == 1 and sub_queries[0] == search_query:
                pass
            else:
                item["decomposition"]["is_decomposed"] = True
                item["decomposition"]["sub_queries"] = sub_queries

                all_paras = list(retrieved)
                seen = {p.get("doc_id") for p in retrieved if p.get("doc_id")}
                for sq in sub_queries:
                    rq = _gpt_rewrite_query(sq)
                    sub_paras = self._base.hybrid_retrieve(rq, top_k=max(3, top_k // 2), bm25_weight=0.5)
                    for p in sub_paras:
                        did = p.get("doc_id")
                        if did and did not in seen:
                            all_paras.append(p)
                            seen.add(did)

                # _gpt_generate_advanced_answer now returns (answer, context_parts)
                adv_answer, adv_context = _gpt_generate_advanced_answer(query, sub_queries, all_paras, base_rag=self._base)
                item["basic_response"] = resp
                item["response"] = adv_answer
                ragas_context = adv_context  # Use the context from advanced answer

                # Update retrieval info with expanded results
                item["retrieval"] = self._format_retrieved_paragraphs(all_paras, answer)
                item["retrieval"]["expanded_by_decomposition"] = True

        final_resp = item["response"]
        if eval_mode == "doi":
            is_correct = _doi_match(final_resp or "", answer or "")
            item["evaluation"] = {
                "is_correct": is_correct,
                "gold_in_top_k": item["retrieval"]["gold_in_top_k"],
                "gold_rank": item["retrieval"]["gold_rank"],
            }
        else:
            # Extract '참고 문단' and '답변' from LLM response (matching reference code)
            response_parts = _extract_response_parts(final_resp)
            extracted_answer = response_parts["answer"]
            extracted_context = response_parts["context"]

            # RAGAS 평가 수행 (using extracted answer/context like reference code)
            ragas_scores = _ragas_eval(query, extracted_answer, extracted_context)
            item["extracted_answer"] = extracted_answer
            item["context"] = extracted_context
            item["ragas_results"] = ragas_scores

        item["execution_time_sec"] = round(time.time() - start, 2)
        return item

    def process_one(self, query: str, answer: str, top_k: int = 5, eval_mode: str = "doi") -> Dict[str, Any]:
        if self.retrieval == "c-rag":
            return self._process_one_crag(query, answer, top_k, eval_mode)
        else:
            return self._process_one_qrrag(query, answer, top_k, eval_mode)

    def evaluate_csv(
        self,
        save_dir: Optional[str] = None,
        filename_hint: str = "",
        eval_mode: Literal["doi", "descriptive"] = "doi",
        top_k: int = 5,
    ):
        # Reset Llama usage tracking at start
        reset_llama_usage()

        qs = _read_questions_from_csv(self.csv_path)
        n = len(qs)
        print(f"[RAG] loaded {n} questions from {self.csv_path}")

        results: List[Dict[str, Any]] = []
        _COND_LABELS = {"1": "single", "2": "multiple"}
        by_cond: defaultdict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
        ragas_stats = {"faithfulness": [], "answer_relevancy": [], "ContextRelevance": []}

        loop_start = time.time()
        for i, q in enumerate(qs, 1):
            query = q.get("question", "")
            gold = q.get("answer", "") or q.get("gold", "")
            raw_cond = str(q.get("condition", "") or q.get("cond", "") or "unknown").strip()
            cond = _COND_LABELS.get(raw_cond, raw_cond)

            pct = (i - 1) / n
            filled = int(30 * pct)
            bar = "█" * filled + " " * (30 - filled)
            elapsed = time.time() - loop_start
            if i > 1:
                avg = elapsed / (i - 1)
                eta_sec = avg * (n - i + 1)
                eta_m, eta_s = divmod(int(eta_sec), 60)
                eta_str = f"{eta_m}m {eta_s:02d}s left"
            else:
                eta_str = "estimating..."
            spin_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            stop_spin = threading.Event()

            def _spinner(_bar=bar, _pct=pct, _i=i, _eta=eta_str):
                idx = 0
                while not stop_spin.is_set():
                    ch = spin_chars[idx % len(spin_chars)]
                    print(f"\r  [{_bar}] {_pct*100:5.1f}%  ({_i}/{n}) {ch}  {_eta}", end="", flush=True)
                    idx += 1
                    stop_spin.wait(0.15)

            t = threading.Thread(target=_spinner, daemon=True)
            t.start()
            item = self.process_one(query, gold, top_k=top_k, eval_mode=eval_mode)
            stop_spin.set()
            t.join()
            pct_done = i / n
            filled_done = int(30 * pct_done)
            bar_done = "█" * filled_done + " " * (30 - filled_done)
            elapsed_now = time.time() - loop_start
            avg_done = elapsed_now / i
            eta_remain = avg_done * (n - i)
            rm, rs = divmod(int(eta_remain), 60)
            eta_done = f"{rm}m {rs:02d}s left" if i < n else "done"
            print(f"\r  [{bar_done}] {pct_done*100:5.1f}%  ({i}/{n})  {eta_done}      ", flush=True)

            # === DEBUG OUTPUT ===
            if eval_mode == "doi":
                is_correct = item.get("evaluation", {}).get("is_correct", False)
                response_text = (item.get("response") or "")[:1200].replace("\n", " ")
                print(f"\n    [Q{i}] {'✓ CORRECT' if is_correct else '✗ WRONG'}")
                print(f"    Question: {query[:300]}{'...' if len(query) > 300 else ''}")
                print(f"    Expected: {gold}")
                print(f"    Response: {response_text}{'...' if len(item.get('response') or '') > 1200 else ''}")
                print()
            else:
                # descriptive mode: show RAGAS scores
                rr = item.get("ragas_results", {})
                if rr and "error" not in rr:
                    faith = rr.get("faithfulness", 0.0)
                    relevancy = rr.get("answer_relevancy", 0.0)
                    ctx_rel = rr.get("ContextRelevance", 0.0)
                    print(f"\n    [Q{i}] Faith={faith:.3f} | Relevancy={relevancy:.3f} | CtxRel={ctx_rel:.3f}")
                else:
                    err_msg = rr.get("error", "unknown") if rr else "no result"
                    print(f"\n    [Q{i}] RAGAS error: {err_msg}")
                print(f"    Question: {query[:200]}{'...' if len(query) > 200 else ''}")
                print()

            if eval_mode == "doi":
                ok = item.get("evaluation", {}).get("is_correct", False)
                by_cond[cond]["total"] += 1
                by_cond[cond]["correct"] += int(bool(ok))
            else:
                rr = item.get("ragas_results", {})
                if rr and "error" not in rr:
                    ragas_stats["faithfulness"].append(rr.get("faithfulness", 0.0))
                    ragas_stats["answer_relevancy"].append(rr.get("answer_relevancy", 0.0))
                    ragas_stats["ContextRelevance"].append(rr.get("ContextRelevance", 0.0))

            results.append(item)

        print()  # newline after progress bar
        out_dir = save_dir or os.path.join(settings.RESULTS_DIR, "RAG")
        os.makedirs(out_dir, exist_ok=True)

        stamp = time.strftime("%Y%m%d_%H%M")
        base_name = f"{self.dataset_format}-{self.retrieval}_{stamp}"

        if eval_mode == "doi":
            total = sum(v["total"] for v in by_cond.values())
            correct = sum(v["correct"] for v in by_cond.values())
            acc = (correct / total) if total else 0.0

            # Calculate retrieval statistics
            gold_in_top_k_count = sum(
                1 for r in results
                if r.get("evaluation", {}).get("gold_in_top_k", False)
            )
            retrieval_recall = (gold_in_top_k_count / total) if total else 0.0

            # Calculate average gold rank (only for items where gold was found)
            gold_ranks = [
                r.get("evaluation", {}).get("gold_rank")
                for r in results
                if r.get("evaluation", {}).get("gold_rank") is not None
            ]
            avg_gold_rank = (sum(gold_ranks) / len(gold_ranks)) if gold_ranks else None

            print("\n" + "=" * 70)
            print(f"  {self.dataset_format.upper()} / {self.retrieval.upper()}  DOI Evaluation Results")
            print("=" * 70)
            print(f"  Overall Accuracy     : {round(acc * 100, 1)}%  ({correct}/{total})")
            print(f"  Retrieval Recall@{top_k}  : {round(retrieval_recall * 100, 1)}%  ({gold_in_top_k_count}/{total})")
            if avg_gold_rank:
                print(f"  Avg Gold Rank        : {round(avg_gold_rank, 2)}")
            print("-" * 70)
            print("  By Condition:")
            for c in sorted(by_cond.keys()):
                v = by_cond[c]
                c_acc = (v["correct"] / v["total"]) if v["total"] else 0.0
                print(f"    {c:<12} : {round(c_acc * 100, 1)}%  ({v['correct']}/{v['total']})")
            print("-" * 70)
            # Llama cost tracking
            llama_usage = get_llama_usage()
            if llama_usage["total_tokens"] > 0:
                print("  Llama API Usage:")
                print(f"    Prompt tokens      : {llama_usage['prompt_tokens']:,}")
                print(f"    Completion tokens  : {llama_usage['completion_tokens']:,}")
                print(f"    Total tokens       : {llama_usage['total_tokens']:,}")
                print(f"    Estimated cost     : ${llama_usage['cost_total_usd']:.4f}")
            print("=" * 70)

            # Build professional output structure
            llama_usage = get_llama_usage()
            output = {
                "metadata": {
                    "experiment_name": f"{self.dataset_format}-{self.retrieval}",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "evaluation_mode": eval_mode,
                    "dataset_format": self.dataset_format,
                    "retrieval_method": self.retrieval,
                    "top_k": top_k,
                    "total_questions": total,
                    "llm_backend": _ACTIVE_BACKEND,
                },
                "llama_cost": llama_usage if llama_usage["total_tokens"] > 0 else None,
                "summary": {
                    "overall_accuracy": {
                        "percentage": round(acc * 100, 2),
                        "correct": correct,
                        "total": total,
                    },
                    "retrieval_performance": {
                        "recall_at_k": round(retrieval_recall * 100, 2),
                        "gold_found_in_top_k": gold_in_top_k_count,
                        "average_gold_rank": round(avg_gold_rank, 2) if avg_gold_rank else None,
                    },
                    "by_condition": {
                        cond: {
                            "accuracy_pct": round((v["correct"] / v["total"]) * 100, 2) if v["total"] else 0.0,
                            "correct": v["correct"],
                            "total": v["total"],
                        }
                        for cond, v in by_cond.items()
                    },
                },
                "results": results,
            }

            out_json = os.path.join(out_dir, base_name + ".json")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            print(f"[save] {out_json}")

        else:
            averages = {k: (sum(v) / len(v) if v else 0.0) for k, v in ragas_stats.items()}
            all_scores = [s for _, arr in ragas_stats.items() for s in arr if not math.isnan(s)]
            total_avg = (sum(all_scores) / len(all_scores)) if all_scores else 0.0

            print("\n" + "=" * 64)
            print("RAGAS Evaluation Summary")
            print("=" * 64)
            print(f"{'Metric':<22}{'Mean':<12}{'Count':<8}")
            print("-" * 64)
            for k, v in averages.items():
                print(f"{k:<22}{v:<12.4f}{len(ragas_stats[k]):<8}")
            print("-" * 64)
            try:
                import pandas as pd
                rows = []
                for it in results:
                    rr = it.get("ragas_results", {})
                    if rr and "error" not in rr:
                        ctx = it.get("context", "")
                        # Handle context as list (from retrieved paragraphs)
                        if isinstance(ctx, list):
                            ctx = " | ".join(ctx[:3])  # Join first 3 contexts
                        rows.append({
                            "question": it.get("question", ""),
                            "answer": it.get("extracted_answer", ""),
                            "context": ctx[:2000] if ctx else "",  # Truncate for CSV
                            "ragas_faithfulness": rr.get("faithfulness", 0.0),
                            "ragas_answer_relevancy": rr.get("answer_relevancy", 0.0),
                            "ragas_ContextRelevance": rr.get("ContextRelevance", 0.0),
                        })
                ragas_csv = os.path.join(out_dir, base_name + ".csv")
                pd.DataFrame(rows).to_csv(ragas_csv, index=False, encoding="utf-8-sig")
                print(f"[save] {ragas_csv}")
            except Exception as e:
                print(f"[warn] pandas save failed: {e}")

            out_json = os.path.join(out_dir, base_name + ".json")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump({"ragas_averages": averages, "ragas_total_average": total_avg, "items": results}, f, ensure_ascii=False, indent=2)
            print(f"[save] {out_json}")


# ======================================================================================
# CLI
# ======================================================================================
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Unified RAG runner (C-RAG / QR-RAG)")
    p.add_argument("--format", choices=["json", "html"], default="json", help="dataset format")
    p.add_argument("--retrieval", choices=["c-rag", "qr-rag"], default="c-rag", help="retrieval style")
    p.add_argument("--mode", choices=["doi", "descriptive"], default="doi", help="evaluation mode")
    p.add_argument("--save_dir", default=None, help="output dir for results")
    p.add_argument("--filename_hint", default="", help="suffix for result filenames")
    p.add_argument("--top_k", type=int, default=5, help="retrieval depth")
    args = p.parse_args()

    rag = UnifiedRAG(dataset_format=args.format, retrieval=args.retrieval)
    rag.evaluate_csv(save_dir=args.save_dir, filename_hint=args.filename_hint, eval_mode=args.mode, top_k=args.top_k)
