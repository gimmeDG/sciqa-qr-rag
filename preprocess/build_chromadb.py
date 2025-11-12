from __future__ import annotations
import os
import re
import json
import argparse
from typing import Dict, Any, List, Tuple, Iterable
import chromadb
from chromadb.utils import embedding_functions

from core import settings
from core.data_utils import read_csv_safely

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def _open_collection(name: str, persist_dir: str, api_key: str, embed_model: str):
    """
    Open (or create) a ChromaDB collection with the OpenAI embedding function attached.
    """
    client = chromadb.PersistentClient(path=persist_dir)
    emb_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=embed_model
    )
    col = client.get_or_create_collection(name=name, embedding_function=emb_fn)
    return client, col


def _row_to_metadata(row: Dict[str, Any]) -> Tuple[str, Dict[str, Any], str]:
    """
    Convert a CSV row to (id, metadata, text).

    Expected columns (minimum):
      - id (docId)
      - text
    Optional columns:
      - section
      - paragraphIdx (or paragraphId)
    """
    doc_id = str(row.get("id") or row.get("docId") or "").strip()
    section = str(row.get("section") or "").strip()
    pidx = row.get("paragraphIdx")
    if pidx is None:
        pidx = row.get("paragraphId")
    pidx = str(pidx if pidx is not None else "").strip()
    text = str(row.get("text") or "").strip()

    uid = f"{doc_id}:{pidx}" if doc_id and pidx != "" else (doc_id or os.urandom(4).hex())
    meta = {"docId": doc_id, "section": section, "paragraphIdx": pidx}
    return uid, meta, text


def _load_csv_records(csv_path: str) -> List[Tuple[str, Dict[str, Any], str]]:
    """
    Load CSV and convert to (id, metadata, text) tuples.
    """
    df = read_csv_safely(csv_path)
    missing = [c for c in ["id", "text"] if c not in df.columns]
    if missing:
        raise ValueError(f"[ingest] missing columns {missing}; expected at least: id, text (+ section, paragraphIdx)")

    records: List[Tuple[str, Dict[str, Any], str]] = []
    for _, row in df.iterrows():
        uid, meta, text = _row_to_metadata({k: row.get(k) for k in df.columns})
        if not text:
            continue
        records.append((uid, meta, text))
    return records


def _add_in_batches(collection, ids: List[str], metadatas: List[Dict[str, Any]], documents: List[str], batch_size: int = 1000):
    """
    Add documents to Chroma in batches.
    """
    n = len(ids)
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        collection.add(
            ids=ids[i:j],
            metadatas=metadatas[i:j],
            documents=documents[i:j]
        )
        print(f"[ingest] added {j}/{n}")


# --------------------------------------------------------------------------------------
# Raw directory ingestion (HTML / JSON)
# --------------------------------------------------------------------------------------
def _strip_html_tags(s: str) -> str:
    s = re.sub(r"<script[\s\S]*?</script>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"<style[\s\S]*?</style>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _split_into_chunks(text: str, max_chars: int) -> List[str]:
    if not text:
        return []
    out: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for seg in re.split(r"(?<=[\.\?\!])\s+", text):
        seg = seg.strip()
        if not seg:
            continue
        add_len = len(seg) + (1 if cur_len else 0)
        if cur_len + add_len <= max_chars:
            cur.append(seg)
            cur_len += add_len
        else:
            if cur:
                out.append(" ".join(cur))
            cur = [seg]
            cur_len = len(seg)
    if cur:
        out.append(" ".join(cur))
    return out


def _safe_json_load(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        with open(path, "r", encoding="cp949", errors="replace") as f:
            return json.load(f)


def _iter_html_dir_records(html_dir: str, section_name: str, max_chars: int) -> Iterable[Tuple[str, Dict[str, Any], str]]:
    """
    Yield (uid, meta, text) from all .html files in a directory.
    uid: <docId>:<paragraphIdx>
    meta: {docId, section, paragraphIdx, file_name}
    """
    if not os.path.isdir(html_dir):
        print(f"[ingest] html dir not found: {html_dir}")
        return

    for fname in sorted(os.listdir(html_dir)):
        if not fname.lower().endswith(".html"):
            continue
        fpath = os.path.join(html_dir, fname)
        try:
            raw = open(fpath, "r", encoding="utf-8", errors="replace").read()
        except Exception:
            try:
                raw = open(fpath, "r", encoding="cp949", errors="replace").read()
            except Exception as e:
                print(f"[skip] {fname}: {e}")
                continue

        text = _strip_html_tags(raw)
        if not text:
            continue
        chunks = _split_into_chunks(text, max_chars=max_chars)
        doc_id = os.path.splitext(fname)[0]
        for i, ck in enumerate(chunks):
            uid = f"{doc_id}:{i}"
            meta = {
                "docId": doc_id,
                "section": section_name,
                "paragraphIdx": i,
                "file_name": fname
            }
            yield (uid, meta, ck)


def _extract_json_paragraphs(js: Dict[str, Any]) -> List[Tuple[str, str, int]]:
    """
    Extract [(section, text, paragraphIdx)] from a combined-like JSON.
    Tries common keys but is defensive to remain format-agnostic.
    """
    out: List[Tuple[str, str, int]] = []

    def _append_list(sec: str, lst: Any):
        if isinstance(lst, list):
            for i, t in enumerate(lst):
                s = str(t or "").strip()
                if s:
                    out.append((sec, s, i))
        elif isinstance(lst, str):
            s = lst.strip()
            if s:
                out.append((sec, s, 0))

    content = js.get("content") if isinstance(js, dict) else None
    base = content if isinstance(content, dict) else js

    for k in ("title", "abstract", "introduction", "results", "discussion", "conclusion"):
        _append_list(k, (base or {}).get(k))

    exp = (base or {}).get("experimentalSection")
    if isinstance(exp, dict):
        for sec, lst in exp.items():
            _append_list(f"experimental:{sec}", lst)

    if not out:
        s = json.dumps(js, ensure_ascii=False)
        if s:
            out.append(("raw", s, 0))

    return out


def _iter_json_dir_records(json_dir: str, max_chars: int) -> Iterable[Tuple[str, Dict[str, Any], str]]:
    """
    Yield (uid, meta, text) from all .json files in a directory.
    uid: <docId>:<paragraphIdx>
    meta: {docId, section, paragraphIdx, file_name}
    """
    if not os.path.isdir(json_dir):
        print(f"[ingest] json dir not found: {json_dir}")
        return

    for fname in sorted(os.listdir(json_dir)):
        if not fname.lower().endswith(".json"):
            continue
        fpath = os.path.join(json_dir, fname)
        try:
            js = _safe_json_load(fpath)
        except Exception as e:
            print(f"[skip] {fname}: {e}")
            continue

        doc_id = str(js.get("id") or js.get("docId") or os.path.splitext(fname)[0])
        for sec, txt, idx in _extract_json_paragraphs(js):
            for j, ck in enumerate(_split_into_chunks(txt, max_chars=max_chars)):
                pidx = idx if j == 0 else f"{idx}_{j}"
                uid = f"{doc_id}:{pidx}"
                meta = {
                    "docId": doc_id,
                    "section": str(sec),
                    "paragraphIdx": pidx,
                    "file_name": fname
                }
                yield (uid, meta, ck)


def _ingest_records(records: Iterable[Tuple[str, Dict[str, Any], str]], collection, batch_size: int) -> None:
    ids: List[str] = []
    metas: List[Dict[str, Any]] = []
    docs: List[str] = []
    seen = set()

    for uid, meta, text in records:
        if not text:
            continue
        _uid = uid
        while _uid in seen:
            _uid = f"{uid}:{len(seen)}"
        seen.add(_uid)
        ids.append(_uid)
        metas.append(meta)
        docs.append(text)

    if not ids:
        print("[ingest] no records to ingest")
        return

    _add_in_batches(collection, ids, metas, docs, batch_size=batch_size)
    print(f"[ingest] completed. items={len(ids)}")


# --------------------------------------------------------------------------------------
# Ingestion (CSV → Chroma collection)
# --------------------------------------------------------------------------------------
def ingest_csv_to_chroma(
    csv_path: str,
    collection_name: str,
    persist_dir: str,
    api_key: str,
    embed_model: str,
    batch_size: int = 1000
) -> None:
    """
    Ingest a CSV file where each row is one paragraph.
    """
    # (kept for backward-compat; not used after RAW migration)
    print(f"[ingest] csv={csv_path}")
    _, col = _open_collection(collection_name, persist_dir, api_key, embed_model)

    recs = _load_csv_records(csv_path)
    if not recs:
        print("[ingest] no records to ingest")
        return

    ids, metas, docs = [], [], []
    seen = set()
    for uid, meta, text in recs:
        if uid in seen:
            uid = f"{uid}:{len(seen)}"
        seen.add(uid)
        ids.append(uid)
        metas.append(meta)
        docs.append(text)

    _add_in_batches(col, ids, metas, docs, batch_size=batch_size)
    print(f"[ingest] completed. collection={collection_name} dir={persist_dir} items={len(ids)}")


# --------------------------------------------------------------------------------------
# High-level tasks
# --------------------------------------------------------------------------------------
def ingest_json_csv(batch_size: int = 1000) -> None:
    """
    Ingest JSON corpus CSV into the JSON collection.
    """
    # switched to RAW-directory ingestion
    _, col = _open_collection(
        settings.RAG_JSON_COLLECTION,
        str(settings.RAG_JSON_PERSIST_DIR),
        settings.OPENAI_API_KEY,
        settings.RAG_EMBED_MODEL
    )
    recs = _iter_json_dir_records(str(settings.RAG_JSON_DATA_PATH), max_chars=400)
    _ingest_records(recs, col, batch_size=batch_size)
    print(f"[ingest] completed. collection={settings.RAG_JSON_COLLECTION} dir={settings.RAG_JSON_PERSIST_DIR}")


def ingest_html_csv(batch_size: int = 1000) -> None:
    """
    Ingest HTML corpus CSV into the HTML collection.
    """
    # switched to RAW-directory ingestion
    _, col = _open_collection(
        settings.RAG_HTML_COLLECTION,
        str(settings.RAG_HTML_PERSIST_DIR),
        settings.OPENAI_API_KEY,
        settings.RAG_EMBED_MODEL
    )
    recs = _iter_html_dir_records(str(settings.RAG_HTML_DATA_PATH), section_name="html_content", max_chars=300)
    _ingest_records(recs, col, batch_size=batch_size)
    print(f"[ingest] completed. collection={settings.RAG_HTML_COLLECTION} dir={settings.RAG_HTML_PERSIST_DIR}")


# --------------------------------------------------------------------------------------
# CLI entrypoint
# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Create/Update ChromaDB collections from CSV (paragraph-level).")
    parser.add_argument("--format", choices=["json", "html", "both"], default="both")
    parser.add_argument("--batch_size", type=int, default=1000)
    args = parser.parse_args()

    if args.format in ("json", "both"):
        ingest_json_csv(batch_size=args.batch_size)
    if args.format in ("html", "both"):
        ingest_html_csv(batch_size=args.batch_size)


if __name__ == "__main__":
    main()
