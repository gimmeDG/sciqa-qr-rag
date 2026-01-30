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

@retry(
    retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
    wait=wait_exponential(multiplier=1, min=1, max=16),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _call_openai_chat(**kwargs):
    return _OPENAI_CLIENT.chat.completions.create(**kwargs)

# ======================================================================================
# Optional dependencies for RAGAS evaluation (used only in descriptive mode)
# - Compatible with both ragas 0.2.x (function-based) and 0.3.x (class-based)
# ======================================================================================
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    try:
        # ragas >= 0.3.x
        from ragas.metrics import ContextRelevance as _ContextRelevanceClass
        _CONTEXT_METRIC = _ContextRelevanceClass()
    except Exception:
        # ragas 0.2.x
        from ragas.metrics import context_relevancy as _context_relevancy_fn
        _CONTEXT_METRIC = _context_relevancy_fn
    from datasets import Dataset
except Exception:
    ragas_evaluate = None
    faithfulness = None
    answer_relevancy = None
    _CONTEXT_METRIC = None
    Dataset = None


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

_RAG_ANSWER_MODEL = os.getenv("RAG_LLM_MODEL", settings.GPT_MODEL_NAME)
_RAG_CHECK_MODEL  = os.getenv("RAG_INSUFFICIENCY_CHECK_MODEL", _RAG_ANSWER_MODEL)


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


def _doi_match(response: str, expected: str) -> bool:
    """Check if expected DOI is found in the response."""
    norm_expected = _normalize_doi(expected)
    if not norm_expected:
        return False
    if norm_expected in _normalize_doi(response):
        return True
    for doi in re.findall(r"10\.\d{4,}(?:\.\d+)*/[^\s,;)\]\"']+", response):
        if _normalize_doi(doi) == norm_expected:
            return True
    for url in re.findall(r"nature\.com/articles/[^\s,;)\]\"']+", response):
        if _normalize_doi(url) == norm_expected:
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
    result = {"context": "", "answer": ""}
    if not response:
        return result
    cm = re.search(r"Context:\s*(.*?)(?=Answer:|$)", response, re.DOTALL | re.IGNORECASE)
    if cm:
        result["context"] = cm.group(1).strip()
    am = re.search(r"Answer:\s*(.*?)$", response, re.DOTALL | re.IGNORECASE)
    if am:
        result["answer"] = am.group(1).strip()
    else:
        parts = re.split(r"Context:\s*", response, flags=re.IGNORECASE)
        result["answer"] = parts[1].strip() if len(parts) > 1 else response.strip()
    return result


# ======================================================================================
# Insufficient-information handling (rule + optional LLM re-check)
# ======================================================================================
def _handle_insufficient_info(
    question: str,
    answer: str,
    context: Any,
    use_llm_recheck: bool = True
) -> Optional[Dict[str, float]]:
    insufficient_keywords = [
        "insufficient information",
        "not enough information",
        "information not available",
        "cannot determine",
        "unknown",
        "insufficient"
    ]
    ans = (answer or "").strip()
    ctx = context if isinstance(context, str) else ("" if context is None else str(context))

    keyword_hit = (not ans) or any(k in ans.lower() for k in insufficient_keywords) or (not ctx.strip())
    if not keyword_hit:
        return None

    if use_llm_recheck and _OPENAI_OK:
        try:
            prompt = (
                "Return STRICT JSON with a single boolean field: {\"is_insufficient\": true/false}.\n\n"
                f"Question:\n{question}\n\n"
                f"Context:\n{ctx[:4000]}\n\n"
                f"Proposed Answer:\n{ans}\n\n"
                "Decision rule:\n"
                "- true  if the context does NOT provide enough information to answer the question faithfully.\n"
                "- false if the answer can be supported by the context.\n"
            )
            resp = _call_openai_chat(
                model=_RAG_CHECK_MODEL,
                temperature=0,
                max_tokens=64,
                messages=[
                    {"role": "system", "content": "You return strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                top_p=1
            )
            content = (resp.choices[0].message.content or "").strip()
            is_true = "true" in content.lower().replace(" ", "")
            if not is_true:
                return None
        except Exception:
            pass

    return {"faithfulness": 1.0, "answer_relevancy": 0.0, "ContextRelevance": 0.0}


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
    joined = "\n\n".join(context_blocks or [])
    if not _OPENAI_OK:
        return "Insufficient information"
    system = (
        "You are a scientific literature analysis assistant. "
        "Answer based only on the provided information. "
        "Do not guess or speculate on uncertain content."
    )
    user = (
        f"The following is document information related to the user's question:\n\n"
        f"{joined}\n\n"
        f"Based on the above information, please answer the following question.\n"
        f"If the information is insufficient, honestly answer 'Insufficient information'.\n\n"
        f"User question: {query}"
    )
    try:
        resp = _call_openai_chat(
            model=_RAG_ANSWER_MODEL,
            temperature=0,
            max_tokens=1000,
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
) -> str:
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
        return "Insufficient information"
    system = (
        "You are a scientific literature analysis assistant. "
        "Provide comprehensive answers based only on the provided information. "
        "Do not guess or speculate on uncertain content."
    )
    subqueries_text = "This question was decomposed into the following sub-questions:\n"
    for i, sq in enumerate(sub_queries, 1):
        subqueries_text += f"{i}. {sq}\n"
    subqueries_text += "\nPlease synthesize the information from each sub-question to provide a final answer.\n\n"
    user = (
        f"The following is document information related to the user's question:\n\n"
        f"{joined}\n\n"
        f"{subqueries_text}"
        f"Based on the above information, please answer the following question.\n"
        f"Comprehensively analyze the relevant content from each retrieved document "
        f"to provide a consistent and thorough answer.\n"
        f"If the information is insufficient, honestly answer 'Insufficient information'.\n\n"
        f"User question: {query}"
    )
    try:
        resp = _call_openai_chat(
            model=_RAG_ANSWER_MODEL,
            temperature=0,
            max_tokens=1500,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or "Insufficient information"
    except Exception:
        return "Insufficient information"


# ======================================================================================
# RAGAS single-sample evaluation (version-compatible)
# ======================================================================================
def _ragas_eval(question: str, answer: str, context: Any) -> Dict[str, float]:
    preset = _handle_insufficient_info(question, answer, context, use_llm_recheck=True)
    if preset is not None:
        return preset

    if ragas_evaluate is None or Dataset is None or _CONTEXT_METRIC is None:
        return {"error": "RAGAS dependencies are not available."}

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

    ds = Dataset.from_dict({
        "question": [question],
        "answer": [answer or ""],
        "contexts": _to_nested_list(context or ""),
    })

    def _to_float(v):
        try:
            if isinstance(v, (int, float)):
                return 0.0 if math.isnan(v) else float(v)
            if isinstance(v, list) and v:
                x = float(v[0]);  return 0.0 if math.isnan(x) else x
            if isinstance(v, dict) and "score" in v:
                x = float(v["score"]);  return 0.0 if math.isnan(x) else x
            return float(v)
        except Exception:
            return 0.0

    def _pick_ctx_score(scores: dict) -> float:
        for k in ("ContextRelevance", "context_relevancy", "context_relevance", "nv_context_relevance"):
            if k in scores:
                return _to_float(scores[k])
        return 0.0

    try:
        raw = ragas_evaluate(ds, metrics=[faithfulness, answer_relevancy, _CONTEXT_METRIC])
        scores_dict = raw.scores if hasattr(raw, "scores") else raw
        scores = {
            "faithfulness":     _to_float(scores_dict.get("faithfulness", scores_dict.get("Faithfulness", 0.0))),
            "answer_relevancy": _to_float(scores_dict.get("answer_relevancy", scores_dict.get("answer_relevance", 0.0))),
            "ContextRelevance": _pick_ctx_score(scores_dict),
        }
        return scores
    except Exception as e:
        return {"error": str(e)}


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

    def answer(self, query: str, top_k: int = 5, rewritten: Optional[str] = None) -> tuple[str, List[Dict[str, Any]]]:
        q = rewritten if rewritten else query
        paras = self._dense_retrieve(q, top_k=top_k)
        doc_ids = list({p.get("doc_id") for p in paras if p.get("doc_id")})
        blocks = self._build_context_blocks(doc_ids)
        resp = self._llm_answer(query, blocks, strict_unknown=True)
        return resp, paras


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
        self._init_chroma(
            settings.RAG_JSON_COLLECTION,
            str(settings.RAG_JSON_PERSIST_DIR),
            settings.OPENAI_API_KEY,
            settings.RAG_EMBED_MODEL,
            data_path=str(settings.RAG_JSON_DATA_PATH),
            data_format="json",
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

    def answer(self, query: str, top_k: int = 5, rewritten: Optional[str] = None) -> tuple[str, List[Dict[str, Any]]]:
        q = rewritten if rewritten else query
        paras = self.base.hybrid_retrieve(q, top_k=top_k, bm25_weight=0.5)
        doc_ids = list({p.get("doc_id") for p in paras if p.get("doc_id")})
        blocks = self.base._build_context_blocks(doc_ids)
        resp = self.base._llm_answer(query, blocks, strict_unknown=True)
        return resp, paras


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
    ):
        self.dataset_format = dataset_format
        retrieval = {"baseline": "c-rag", "advanced": "qr-rag"}.get(retrieval, retrieval)
        self.retrieval = retrieval
        if dataset_format == "json":
            self._base = JSONCRAG()
        else:
            self._base = HTMLCRAG()
        self.csv_path = settings.RAG_QA_CSV_PATH
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

    def _process_one_crag(self, query: str, answer: str, top_k: int, eval_mode: str) -> Dict[str, Any]:
        start = time.time()
        resp, retrieved = self._base.answer(query, top_k=top_k)
        item: Dict[str, Any] = {
            "question": query,
            "expected_answer": answer,
            "response": resp,
            "retrieved_documents": list({p.get("doc_id") for p in retrieved if p.get("doc_id")}),
            "retrieved_paragraphs": retrieved,
        }

        if eval_mode == "doi":
            item["is_correct"] = _doi_match(resp or "", answer or "")
        else:
            parts = _extract_response_parts(resp)
            ans = parts.get("answer", "")
            ctx = parts.get("context", "")
            ragas_scores = _ragas_eval(query, ans, ctx)
            item["extracted_answer"] = ans
            item["context"] = ctx
            item["ragas_results"] = ragas_scores

        item["execution_time"] = time.time() - start
        return item

    def _process_one_qrrag(self, query: str, answer: str, top_k: int, eval_mode: str) -> Dict[str, Any]:
        start = time.time()
        search_query = _strip_doi_prefix(query)
        rewritten = _gpt_rewrite_query(search_query)
        resp, retrieved = self._runner.answer(query, top_k=top_k, rewritten=rewritten)

        item: Dict[str, Any] = {
            "question": query,
            "expected_answer": answer,
            "response": resp,
            "rewritten_query": rewritten,
            "retrieved_documents": list({p.get("doc_id") for p in retrieved if p.get("doc_id")}),
            "retrieved_paragraphs": retrieved,
            "is_basic_sufficient": True,
            "is_decomposed": False,
        }

        is_insufficient = self._detect_insufficiency(resp)
        item["is_basic_sufficient"] = not is_insufficient

        if is_insufficient:
            sub_queries = _gpt_decompose_query(search_query)
            if len(sub_queries) == 1 and sub_queries[0] == search_query:
                pass
            else:
                item["is_decomposed"] = True
                item["sub_queries"] = sub_queries

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

                adv_answer = _gpt_generate_advanced_answer(query, sub_queries, all_paras, base_rag=self._base)
                item["response"] = adv_answer
                item["basic_response"] = resp
                item["retrieved_documents"] = list({p.get("doc_id") for p in all_paras if p.get("doc_id")})
                item["retrieved_paragraphs"] = all_paras

        final_resp = item["response"]
        if eval_mode == "doi":
            item["is_correct"] = _doi_match(final_resp or "", answer or "")
        else:
            parts = _extract_response_parts(final_resp)
            ans = parts.get("answer", "")
            ctx = parts.get("context", "")
            ragas_scores = _ragas_eval(query, ans, ctx)
            item["extracted_answer"] = ans
            item["context"] = ctx
            item["ragas_results"] = ragas_scores

        item["execution_time"] = time.time() - start
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

            if eval_mode == "doi":
                ok = item.get("is_correct", False)
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

            print("\n" + "=" * 50)
            print(f"  {self.dataset_format.upper()} / {self.retrieval.upper()}  DOI Evaluation")
            print("=" * 50)
            print(f"  Overall Accuracy : {round(acc * 100, 1)}%  ({correct}/{total})")
            print("-" * 50)
            for c in sorted(by_cond.keys()):
                v = by_cond[c]
                c_acc = (v["correct"] / v["total"]) if v["total"] else 0.0
                print(f"  condition={c:<10} {round(c_acc * 100, 1)}%  ({v['correct']}/{v['total']})")
            print("=" * 50)

            out_json = os.path.join(out_dir, base_name + ".json")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump({
                    "overall_accuracy_pct": round(acc * 100, 1),
                    "total": total,
                    "correct": correct,
                    "by_condition": dict(by_cond),
                    "dataset_format": self.dataset_format,
                    "retrieval": self.retrieval,
                    "top_k": top_k,
                    "items": results,
                }, f, ensure_ascii=False, indent=2)
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
                        rows.append({
                            "question": it.get("question", ""),
                            "answer": it.get("extracted_answer", ""),
                            "context": it.get("context", ""),
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
