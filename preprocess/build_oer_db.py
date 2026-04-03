from __future__ import annotations
import os
import re
import json
import glob
import argparse
import datetime
from typing import Dict, Any, List, Tuple

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from tqdm import tqdm
from bs4 import BeautifulSoup
import tiktoken

from core import settings

# --------------------------------------------------------------------------------------
# Token counting / splitting
# --------------------------------------------------------------------------------------
_TOKENIZER: tiktoken.Encoding | None = None
MAX_EMBED_TOKENS = 8191
OVERLAP_TOKENS   = 200


def _get_tokenizer() -> tiktoken.Encoding:
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = tiktoken.encoding_for_model("text-embedding-3-large")
    return _TOKENIZER


def _split_long_text(text: str) -> List[str]:
    """Split text into chunks that fit within MAX_EMBED_TOKENS.

    If the text is within the limit, returns [text] as-is.
    Otherwise splits by token count, prepending OVERLAP_TOKENS from the
    previous chunk to preserve context.
    """
    enc = _get_tokenizer()
    tokens = enc.encode(text)
    if len(tokens) <= MAX_EMBED_TOKENS:
        return [text]

    parts: List[str] = []
    start = 0
    while start < len(tokens):
        end = start + MAX_EMBED_TOKENS
        chunk_tokens = tokens[start:end]
        parts.append(enc.decode(chunk_tokens))
        start = end - OVERLAP_TOKENS
        if start >= len(tokens):
            break
    return parts


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------
def _open_collection(name: str, persist_dir: str):
    """Open (or create) a ChromaDB collection with OpenAI embedding function."""
    client = chromadb.PersistentClient(path=persist_dir)
    emb_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=settings.OPENAI_API_KEY,
        model_name=settings.RAG_EMBED_MODEL,
        dimensions=settings.RAG_EMBED_DIMENSIONS,
    )
    col = client.get_or_create_collection(
        name=name,
        embedding_function=emb_fn,
        metadata={
            "embed_model": settings.RAG_EMBED_MODEL,
            "embed_dimensions": settings.RAG_EMBED_DIMENSIONS,
            "created_date": str(datetime.datetime.now()),
        },
    )
    return client, col


def _get_openai_client() -> OpenAI:
    return OpenAI(api_key=settings.OPENAI_API_KEY)


def _embed_texts(client: OpenAI, texts: List[str], batch_size: int = 64) -> List[List[float]]:
    """Generate embeddings for a list of texts using OpenAI API in batches."""
    all_embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(
            model=settings.RAG_EMBED_MODEL,
            input=batch,
            dimensions=settings.RAG_EMBED_DIMENSIONS,
        )
        all_embeddings.extend([d.embedding for d in resp.data])
    return all_embeddings


def _add_in_batches(
    collection,
    ids: List[str],
    metadatas: List[Dict[str, Any]],
    documents: List[str],
    embeddings: List[List[float]],
    batch_size: int = 500,
):
    """Add documents + pre-computed embeddings to ChromaDB in batches."""
    n = len(ids)
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        collection.add(
            ids=ids[i:j],
            metadatas=metadatas[i:j],
            documents=documents[i:j],
            embeddings=embeddings[i:j],
        )
        print(f"  [ingest] added {j}/{n}")


# ======================================================================================
# HTML ingestion  (file -> BeautifulSoup -> 300-char chunks -> embed -> Chroma)
# ======================================================================================
class HtmlChunker:
    """Split HTML file text into fixed-size character chunks with sentence awareness."""

    def __init__(self, chunk_size: int = 300):
        self.chunk_size = chunk_size

    def extract_text(self, html_content: str) -> str:
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup(["script", "style"]):
            tag.extract()
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\n+", "\n", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def chunk_text(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: List[str] = []
        current = ""
        for sentence in sentences:
            if len(sentence) > self.chunk_size:
                words = sentence.split()
                temp = ""
                for word in words:
                    candidate = (temp + " " + word) if temp else word
                    if len(candidate) <= self.chunk_size:
                        temp = candidate
                    else:
                        if temp:
                            if current and len(current + " " + temp) <= self.chunk_size:
                                current = current + " " + temp
                            else:
                                if current:
                                    chunks.append(current)
                                current = temp
                        temp = word
                if temp:
                    if current and len(current + " " + temp) <= self.chunk_size:
                        current = current + " " + temp
                    else:
                        if current:
                            chunks.append(current)
                        current = temp
            else:
                candidate = (current + " " + sentence) if current else sentence
                if len(candidate) <= self.chunk_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    current = sentence
        if current:
            chunks.append(current)
        return chunks

    def process_file(self, path: str) -> Tuple[str, List[str]]:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            html = f.read()
        text = self.extract_text(html)
        if not text:
            return os.path.basename(path), []
        return os.path.basename(path), self.chunk_text(text)


def ingest_html(batch_size: int = 500) -> None:
    """Parse HTML files -> chunk -> embed (3072-dim) -> store in ChromaDB."""
    data_path = str(settings.RAG_HTML_DATA_PATH)
    html_files = sorted(glob.glob(os.path.join(data_path, "*.html")))
    if not html_files:
        print(f"[html] no .html files found in {data_path}")
        return

    print(f"[html] found {len(html_files)} HTML files in {data_path}")
    _, col = _open_collection(
        name=settings.RAG_HTML_COLLECTION,
        persist_dir=str(settings.RAG_HTML_PERSIST_DIR),
    )
    oai = _get_openai_client()
    chunker = HtmlChunker(chunk_size=settings.RAG_HTML_CHUNK_SIZE)

    total_chunks = 0
    error_files = 0
    for html_file in tqdm(html_files, desc="[html] processing files"):
        try:
            file_name, chunks = chunker.process_file(html_file)
            if not chunks:
                continue

            doc_id = file_name.replace(".html", "")
            if doc_id.endswith("_cleaned"):
                doc_id = doc_id[:-8]

            ids: List[str] = []
            metas: List[Dict[str, Any]] = []
            docs: List[str] = []

            for idx, chunk_text in enumerate(chunks):
                paragraph_id = f"{doc_id}_{idx}"
                ids.append(paragraph_id)
                docs.append(chunk_text)
                metas.append({
                    "docId": doc_id,
                    "paragraphId": paragraph_id,
                    "section": "html_content",
                    "paragraphIdx": idx,
                    "file_name": file_name,
                    "chunk_size": settings.RAG_HTML_CHUNK_SIZE,
                })

            embeddings = _embed_texts(oai, docs)
            _add_in_batches(col, ids, metas, docs, embeddings, batch_size=batch_size)
            total_chunks += len(ids)

        except Exception as e:
            error_files += 1
            print(f"\n[html] ERROR processing {os.path.basename(html_file)}: {e}")

    print(f"[html] completed. collection={settings.RAG_HTML_COLLECTION} "
          f"total_chunks={total_chunks} errors={error_files}")


# ======================================================================================
# JSON ingestion  (file -> section-level paragraphs + SynthesisEntity -> embed -> Chroma)
# ======================================================================================
def _extract_paragraphs_from_json(json_path: str) -> List[Dict[str, Any]]:
    """Extract section-level paragraphs and SynthesisEntity fields from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        doc_data = json.load(f)

    doc_id = doc_data["metadata"]["id"]
    title = doc_data["metadata"].get("title", "")
    authors_list = doc_data["metadata"].get("authors", [])
    authors_str = json.dumps(authors_list)
    content = doc_data.get("content", {})

    paragraphs: List[Dict[str, Any]] = []

    def _add(text: str, section: str, pidx: int = 0, synth_type: str = "", **extra):
        if not text or not isinstance(text, str):
            return
        entry: Dict[str, Any] = {
            "textContent": text.strip(),
            "docId": doc_id,
            "section": section,
            "paragraphIdx": pidx,
            "synthesisType": synth_type,
            "authors": authors_str,
        }
        entry.update(extra)
        paragraphs.append(entry)

    # 1. Title
    if title:
        _add(title, "title")

    # 2. Abstract
    abstract = content.get("abstract")
    if isinstance(abstract, dict):
        for atype, atext in abstract.items():
            _add(atext, f"abstract.{atype}")
    elif isinstance(abstract, str):
        _add(abstract, "abstract")

    # 3. Introduction
    _add(content.get("introduction", ""), "introduction")

    # 4. experimentalSection
    exp = content.get("experimentalSection")
    if isinstance(exp, dict):
        for sec_type in ("synthesis", "system", "performance"):
            items = exp.get(sec_type)
            if isinstance(items, list):
                for item_idx, item in enumerate(items):
                    if isinstance(item, dict) and "content" in item:
                        synth_type = item.get("synthesisType") or ""
                        _add(item["content"], f"experimentalSection.{sec_type}",
                             pidx=item_idx, synth_type=synth_type)
                        # SynthesisEntity fields
                        se = item.get("synthesisEntity")
                        if se:
                            _extract_synthesis_entities(
                                se, doc_id, f"experimentalSection.{sec_type}",
                                item_idx, synth_type, authors_str, paragraphs,
                            )
            elif isinstance(items, str):
                _add(items, f"experimentalSection.{sec_type}")
    elif isinstance(exp, str):
        _add(exp, "experimentalSection")

    # 5. resultsAndDiscussion
    _add(content.get("resultsAndDiscussion", ""), "resultsAndDiscussion")

    # 6. conclusion
    _add(content.get("conclusion", ""), "conclusion")

    return paragraphs


def _extract_synthesis_entities(
    se: Dict[str, Any],
    doc_id: str,
    section: str,
    pidx: int,
    synth_type: str,
    authors_str: str,
    out: List[Dict[str, Any]],
) -> None:
    """Extract originalForm / formula from SynthesisEntity categories."""
    categories = ("target", "precursors", "solvents", "additives", "substrates")
    for cat in categories:
        raw = se.get(cat)
        if raw is None:
            continue
        items = [raw] if cat == "target" else (raw if isinstance(raw, list) else [])
        for idx, item in enumerate(items):
            if not item:
                continue
            for field in ("originalForm", "formula"):
                val = item.get(field)
                if val:
                    out.append({
                        "textContent": str(val),
                        "docId": doc_id,
                        "section": section,
                        "paragraphIdx": pidx,
                        "synthesisType": synth_type,
                        "authors": authors_str,
                        "isSynthesisEntity": True,
                        "entityType": f"{cat}_{field}",
                        "entityIdx": idx,
                    })


def ingest_json(batch_size: int = 500) -> None:
    """Parse JSON files -> section-level paragraphs -> embed (3072-dim) -> store in ChromaDB."""
    data_path = str(settings.RAG_JSON_DATA_PATH)
    json_files = sorted(glob.glob(os.path.join(data_path, "*.json")))
    if not json_files:
        print(f"[json] no .json files found in {data_path}")
        return

    print(f"[json] found {len(json_files)} JSON files in {data_path}")
    _, col = _open_collection(
        name=settings.RAG_JSON_COLLECTION,
        persist_dir=str(settings.RAG_JSON_PERSIST_DIR),
    )
    oai = _get_openai_client()

    total_items = 0
    error_files = 0
    split_count = 0
    for json_file in tqdm(json_files, desc="[json] processing files"):
        try:
            paragraphs = _extract_paragraphs_from_json(json_file)
            if not paragraphs:
                continue

            ids: List[str] = []
            metas: List[Dict[str, Any]] = []
            docs: List[str] = []

            seen: set = set()
            for p in paragraphs:
                base_id = f"{p['docId']}_{p['section']}_{p['paragraphIdx']}"
                if p.get("synthesisType"):
                    base_id += f"_{p['synthesisType']}"
                if p.get("isSynthesisEntity"):
                    base_id += f"_{p['entityType']}_{p['entityIdx']}"

                text_parts = _split_long_text(p["textContent"])
                is_multipart = len(text_parts) > 1
                if is_multipart:
                    split_count += 1

                for part_idx, part_text in enumerate(text_parts):
                    if is_multipart:
                        uid_base = f"{base_id}_part{part_idx + 1}"
                    else:
                        uid_base = base_id

                    uid = uid_base
                    counter = 0
                    while uid in seen:
                        counter += 1
                        uid = f"{uid_base}_{counter}"
                    seen.add(uid)

                    ids.append(uid)
                    docs.append(part_text)

                    meta: Dict[str, Any] = {
                        "docId": p["docId"],
                        "paragraphId": uid,
                        "section": p["section"],
                        "paragraphIdx": p["paragraphIdx"],
                        "synthesisType": p.get("synthesisType", ""),
                        "authors": p.get("authors", ""),
                    }
                    if p.get("isSynthesisEntity"):
                        meta["isSynthesisEntity"] = True
                        meta["entityType"] = p["entityType"]
                        meta["entityIdx"] = p["entityIdx"]
                    if is_multipart:
                        meta["partIdx"] = part_idx + 1
                        meta["totalParts"] = len(text_parts)
                    metas.append(meta)

            embeddings = _embed_texts(oai, docs)
            _add_in_batches(col, ids, metas, docs, embeddings, batch_size=batch_size)
            total_items += len(ids)

        except Exception as e:
            error_files += 1
            print(f"\n[json] ERROR processing {os.path.basename(json_file)}: {e}")

    print(f"[json] completed. collection={settings.RAG_JSON_COLLECTION} "
          f"total_items={total_items} errors={error_files} long_text_splits={split_count}")


# --------------------------------------------------------------------------------------
# CLI entrypoint
# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Create ChromaDB collections from HTML/JSON source files (direct parsing)."
    )
    parser.add_argument("--format", choices=["json", "html", "both"], default="both")
    parser.add_argument("--batch_size", type=int, default=500)
    args = parser.parse_args()

    if args.format in ("json", "both"):
        ingest_json(batch_size=args.batch_size)
    if args.format in ("html", "both"):
        ingest_html(batch_size=args.batch_size)


if __name__ == "__main__":
    main()
