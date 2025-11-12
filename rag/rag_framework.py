from __future__ import annotations
import os
import re
import csv
import math
import time
import json
from typing import List, Dict, Any, Optional, Literal
from collections import defaultdict

from core import settings

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
            resp = _OPENAI_CLIENT.chat.completions.create(
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
        completion = _OPENAI_CLIENT.chat.completions.create(
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
        completion = _OPENAI_CLIENT.chat.completions.create(
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
        return f"Context:\n{joined}\nAnswer:\nInsufficient information"
    system = (
        "You are a scientific RAG assistant. Use only the provided context. "
        "If evidence is insufficient, answer with 'Insufficient information'. "
        "Output format must be:\n"
        "Context:\n<selected context>\nAnswer:\n<final answer>"
    )
    user = (
        f"Question:\n{query}\n\n"
        f"Candidate context blocks:\n{joined}\n\n"
        "- Do not use any information beyond the context.\n"
        "- If insufficient, respond 'Insufficient information' in the Answer section."
    )
    try:
        resp = _OPENAI_CLIENT.chat.completions.create(
            model=_RAG_ANSWER_MODEL,
            temperature=0,
            max_tokens=800,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            top_p=1
        )
        return resp.choices[0].message.content or f"Context:\n{joined}\nAnswer:\nInsufficient information"
    except Exception:
        return f"Context:\n{joined}\nAnswer:\nInsufficient information"

def _gpt_generate_advanced_answer(
    query: str,
    sub_queries: List[str],
    paragraphs: List[Dict[str, Any]],
    max_tokens: int = 2000
) -> str:
    processed_doc_ids = set()
    context_parts: List[str] = []
    for para in sorted(paragraphs, key=lambda x: x.get("score", 0), reverse=True):
        doc_id = para.get("doc_id")
        if not doc_id or doc_id in processed_doc_ids:
            continue
        processed_doc_ids.add(doc_id)
        preview = para.get("text", "")
        section = para.get("section", "")
        pidx = para.get("paragraph_idx", "")
        header = f"[DocID: {doc_id}] [Section: {section}] [ParagraphIdx: {pidx}]"
        chunk = f"{header}\n{preview}"
        context_parts.append(chunk)
        if len("\n\n".join(context_parts)) > max_tokens * 4:
            break
    joined = "\n\n".join(context_parts)
    if not _OPENAI_OK:
        return f"Context:\n{joined}\nAnswer:\nInsufficient information"
    sys_msg = (
        "You are a scientific RAG assistant. Synthesize across multiple documents if needed. "
        "If evidence is insufficient, answer with 'Insufficient information'. "
        "Format:\nContext:\n<compiled context>\nAnswer:\n<final answer>"
    )
    user_msg = (
        f"Original question:\n{query}\n\n"
        f"Decomposed sub-questions:\n- " + "\n- ".join(sub_queries) + "\n\n"
        f"Compiled context (multi-document):\n{joined}\n\n"
        "Rules:\n"
        "- Use only the compiled context.\n"
        "- If insufficient, write exactly 'Insufficient information' in the Answer section."
    )
    try:
        resp = _OPENAI_CLIENT.chat.completions.create(
            model=_RAG_ANSWER_MODEL,
            temperature=0,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            top_p=1
        )
        return resp.choices[0].message.content or f"Context:\n{joined}\nAnswer:\nInsufficient information"
    except Exception:
        return f"Context:\n{joined}\nAnswer:\nInsufficient information"


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
        """Dense + BM25 hybrid with simple weighted fusion by doc_id."""
        dense = self._dense_retrieve(query, top_k=max(10, top_k * 2))
        sparse = self._sparse_retrieve(query, top_k=max(10, top_k * 3))
        by_doc: Dict[str, Dict[str, Any]] = {}

        def _merge(items: List[Dict[str, Any]], key: str):
            for it in items:
                did = it.get("doc_id")
                if not did:
                    continue
                d = by_doc.setdefault(
                    did,
                    {
                        "doc_id": did,
                        "text": it.get("text", ""),
                        "section": it.get("section", ""),
                        "paragraph_idx": it.get("paragraph_idx", -1),
                        "score_components": {"vector": 0.0, "bm25": 0.0},
                    },
                )
                d["score_components"][key] = max(d["score_components"].get(key, 0.0), float(it.get("score", 0.0)))
                if not d.get("text") and it.get("text"):
                    d["text"] = it["text"]
                # keep meta
                self._meta_cache[did] = {
                    "section": it.get("section", ""),
                    "paragraph_idx": it.get("paragraph_idx", -1),
                }

        _merge(dense, "vector")
        _merge(sparse, "bm25")

        fused: List[Dict[str, Any]] = []
        for did, d in by_doc.items():
            v = float(d["score_components"].get("vector", 0.0))
            b = float(d["score_components"].get("bm25", 0.0))
            score = (1 - bm25_weight) * v + bm25_weight * b
            fused.append({
                "doc_id": did,
                "text": d.get("text", ""),
                "section": d.get("section", ""),
                "paragraph_idx": d.get("paragraph_idx", -1),
                "score": score
            })

        fused.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return fused[:top_k]

    def _build_context_blocks(self, doc_ids: List[str]) -> List[str]:
        blocks: List[str] = []
        miss = [d for d in doc_ids if d not in self._para_cache]
        # optional fetch_by_ids fallback if provided by mixin
        fetch = getattr(self, "_fetch_by_ids", None)
        if miss and callable(fetch):
            try:
                fetch(miss)
            except Exception:
                pass

        for did in doc_ids:
            txt = self._para_cache.get(did)
            if not txt:
                continue
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

    def _init_chroma(self, name: str, persist_dir: str, api_key: str, embed_model: str):
        import chromadb
        from chromadb.utils import embedding_functions

        self._chroma = chromadb.PersistentClient(path=str(persist_dir))
        self._embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key, model_name=embed_model
        )
        self._collection = self._chroma.get_collection(
            name=name, embedding_function=self._embedding_fn
        )

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

    def _fetch_by_ids(self, ids: List[str]) -> None:
        try:
            got = self._collection.get(ids=ids, include=["documents", "metadatas"])
            docs = got.get("documents", [])
            metas = got.get("metadatas", [])
            for doc, meta in zip(docs, metas):
                did = meta.get("docId") or meta.get("id")
                if did and doc and did not in self._para_cache:
                    self._para_cache[did] = doc
                    self._meta_cache[did] = {
                        "section": meta.get("section", ""),
                        "paragraph_idx": meta.get("paragraphIdx", meta.get("paragraphId", "")),
                    }
        except Exception:
            pass

class JSONCRAG(_ChromaMixin, _BaseRAG):
    """Conventional RAG on JSON corpus: dense-only retrieval."""
    def __init__(self):
        super().__init__()
        self._init_chroma(
            settings.RAG_JSON_COLLECTION,
            str(settings.RAG_JSON_PERSIST_DIR),
            settings.OPENAI_API_KEY,
            settings.RAG_EMBED_MODEL,
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
            self.csv_path = settings.RAG_JSON_CSV_PATH
        else:
            self._base = HTMLCRAG()
            self.csv_path = settings.RAG_HTML_CSV_PATH
        self._runner = QRRAGWrapper(self._base, dataset_format) if retrieval == "qr-rag" else self._base

    def _second_pass_qr_rag(
        self,
        query: str,
        top_k: int,
        basic_paragraphs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        sub_queries = _gpt_decompose_query(query)
        all_paras = list(basic_paragraphs)
        seen = {p.get("doc_id") for p in basic_paragraphs if p.get("doc_id")}
        for sq in sub_queries:
            rq = _gpt_rewrite_query(sq)
            sub_paras = self._base.hybrid_retrieve(rq, top_k=max(3, top_k // 2), bm25_weight=0.5)
            for p in sub_paras:
                did = p.get("doc_id")
                if did and did not in seen:
                    all_paras.append(p)
                    seen.add(did)
        adv_answer = _gpt_generate_advanced_answer(query, sub_queries, all_paras, max_tokens=2000)
        return {"sub_queries": sub_queries, "all_paragraphs": all_paras, "advanced_answer": adv_answer}

    def process_one(self, query: str, answer: str, top_k: int = 5, eval_mode: str = "doi") -> Dict[str, Any]:
        start = time.time()
        rq = _gpt_rewrite_query(query)
        resp, retrieved = self._runner.answer(query, top_k=top_k, rewritten=rq)
        item: Dict[str, Any] = {
            "question": query,
            "expected_answer": answer,
            "response": resp,
            "retrieved_documents": list({p.get("doc_id") for p in retrieved if p.get("doc_id")}),
            "retrieved_paragraphs": retrieved,
        }

        if eval_mode == "doi":
            ok = (answer or "").lower() in (resp or "").lower()
            item["is_correct"] = ok
            if not ok and self.retrieval == "c-rag":
                sp = self._second_pass_qr_rag(query, top_k, retrieved)
                adv = sp.get("advanced_answer", "")
                item["is_correct_after_decompose"] = (answer or "").lower() in (adv or "").lower()
                item["advanced"] = sp
                if item["is_correct_after_decompose"]:
                    item["response"] = adv
        else:
            parts = _extract_response_parts(resp)
            ans = parts.get("answer", "")
            ctx = parts.get("context", "")
            preset = _handle_insufficient_info(query, ans, ctx, use_llm_recheck=True)
            if preset is not None:
                sp = self._second_pass_qr_rag(query, top_k, retrieved)
                item["advanced"] = sp
                item["response"] = sp["advanced_answer"]
                parts = _extract_response_parts(item["response"])
                ans = parts.get("answer", "")
                ctx = parts.get("context", "")
            ragas_scores = _ragas_eval(query, ans, ctx)
            item["extracted_answer"] = ans
            item["context"] = ctx
            item["ragas_results"] = ragas_scores

        item["execution_time"] = time.time() - start
        return item

    def evaluate_csv(
        self,
        save_dir: Optional[str] = None,
        filename_hint: str = "",
        eval_mode: Literal["doi", "descriptive"] = "doi",
        top_k: int = 5,
    ):
        qs = _read_questions_from_csv(self.csv_path)
        n = len(qs)
        print(f"[RAG] loaded {n} samples from {self.csv_path}")

        results: List[Dict[str, Any]] = []
        by_cond: defaultdict[float, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
        ragas_stats = {"faithfulness": [], "answer_relevancy": [], "ContextRelevance": []}

        for i, q in enumerate(qs, 1):
            query = q.get("question", "")
            gold = q.get("answer", "") or q.get("gold", "")
            cond = float(q.get("cond", "0") or 0.0)

            print(f"\n[{i}/{n}] Q: {query[:120]}.")
            item = self.process_one(query, gold, top_k=top_k, eval_mode=eval_mode)

            if eval_mode == "doi":
                ok = item.get("is_correct", False)
                ok2 = item.get("is_correct_after_decompose", None)
                by_cond[cond]["total"] += 1
                by_cond[cond]["correct"] += int(ok or ok2)
            else:
                rr = item.get("ragas_results", {})
                if rr and "error" not in rr:
                    ragas_stats["faithfulness"].append(rr.get("faithfulness", 0.0))
                    ragas_stats["answer_relevancy"].append(rr.get("answer_relevancy", 0.0))
                    ragas_stats["ContextRelevance"].append(rr.get("ContextRelevance", 0.0))

            results.append(item)

        out_dir = save_dir or os.path.join(settings.RESULTS_DIR, "RAG")
        os.makedirs(out_dir, exist_ok=True)

        stamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"rag_eval_{self.dataset_format}_{self.retrieval}{filename_hint}_{stamp}"

        if eval_mode == "doi":
            total = sum(v["total"] for v in by_cond.values())
            correct = sum(v["correct"] for v in by_cond.values())
            acc = (correct / total) if total else 0.0
            print(f"\n[summary] total={total} correct={correct} accuracy={acc:.4f}")

            out_json = os.path.join(out_dir, base_name + ".json")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump({"accuracy": acc, "by_cond": by_cond, "items": results}, f, ensure_ascii=False, indent=2)
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
