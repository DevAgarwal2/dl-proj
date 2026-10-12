"""Chunk and embed a medical-literature corpus into a vector store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from openmed.rag.embeddings import BaseEmbedder, build_embedder
from openmed.rag.vectorstore import NumpyVectorStore
from openmed.utils import ensure_dir, get_logger, save_json

LOGGER = get_logger(__name__)

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".jsonl", ".json"}


def chunk_text(
    text: str,
    chunk_size: int = 900,
    overlap: int = 150,
    min_chars: int = 80,
) -> List[str]:
    """Split text into overlapping character windows, preferring paragraph breaks."""
    text = " ".join(text.split())
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text] if len(text) >= min_chars else []

    chunks: List[str] = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(text), step):
        window = text[start : start + chunk_size]
        if len(window) < min_chars:
            break
        chunks.append(window.strip())
        if start + chunk_size >= len(text):
            break
    return chunks


def iter_documents(corpus_dir: str | Path) -> Iterator[Tuple[str, str, Dict]]:
    """Yield ``(doc_id, title, metadata)`` and attach text via metadata['text']."""
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        return
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    text = record.get("text") or record.get("abstract") or record.get("content", "")
                    title = record.get("title") or record.get("question") or f"{path.stem}#{line_no}"
                    metadata = {k: v for k, v in record.items() if k not in {"text", "content"}}
                    metadata.setdefault("source", path.name)
                    yield f"{path.stem}:{line_no}", str(title), {**metadata, "text": str(text)}
        elif path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            records = data if isinstance(data, list) else [data]
            for index, record in enumerate(records):
                text = record.get("text") or record.get("abstract") or record.get("content", "")
                title = record.get("title", f"{path.stem}#{index}")
                metadata = {k: v for k, v in record.items() if k not in {"text", "content"}}
                metadata.setdefault("source", path.name)
                yield f"{path.stem}:{index}", str(title), {**metadata, "text": str(text)}
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
            yield path.stem, path.stem, {"source": path.name, "text": text}


def ingest_corpus(
    corpus_dir: str | Path,
    store_path: str | Path,
    embedder: Optional[BaseEmbedder] = None,
    chunk_size: int = 900,
    overlap: int = 150,
    min_chars: int = 80,
    metric: str = "cosine",
) -> NumpyVectorStore:
    """Read the corpus, chunk it, embed it and persist a vector store."""
    embedder = embedder or build_embedder({"backend": "hashing"})
    store = NumpyVectorStore(metric=metric)

    ids: List[str] = []
    texts: List[str] = []
    metadatas: List[Dict] = []
    for doc_id, title, metadata in iter_documents(corpus_dir):
        chunks = chunk_text(metadata.pop("text", ""), chunk_size, overlap, min_chars)
        for chunk_index, chunk in enumerate(chunks):
            ids.append(f"{doc_id}::{chunk_index}")
            texts.append(chunk)
            metadatas.append({**metadata, "title": title, "chunk": chunk_index})

    if texts:
        vectors = embedder.encode(texts, batch_size=32, normalize=metric == "cosine")
        store.add(ids, vectors, texts, metadatas)

    store.save(store_path)
    save_json(
        {
            "corpus_dir": str(corpus_dir),
            "store_path": str(store_path),
            "embedder": embedder.name,
            "dim": store.dim,
            "documents": len({m.get("source") for m in metadatas}) if metadatas else 0,
            "chunks": len(texts),
            "chunk_size": chunk_size,
            "overlap": overlap,
        },
        Path(store_path) / "ingest_manifest.json",
    )
    LOGGER.info("Ingested %d chunks from %s -> %s", len(texts), corpus_dir, store_path)
    return store


def ingest_from_config(cfg) -> NumpyVectorStore:
    section = cfg.ingest
    return ingest_corpus(
        corpus_dir=section.corpus_dir,
        store_path=cfg.vectorstore.path,
        embedder=build_embedder(cfg),
        chunk_size=int(section.get("chunk_size", 900)),
        overlap=int(section.get("chunk_overlap", 150)),
        min_chars=int(section.get("min_chunk_chars", 80)),
        metric=str(cfg.vectorstore.get("metric", "cosine")),
    )
