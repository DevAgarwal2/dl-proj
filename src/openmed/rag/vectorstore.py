"""A tiny, dependency-free persistent vector store (cosine / dot product)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class SearchResult:
    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "text": self.text, "score": self.score, "metadata": self.metadata}


class NumpyVectorStore:
    def __init__(self, dim: Optional[int] = None, metric: str = "cosine") -> None:
        if metric not in {"cosine", "dot"}:
            raise ValueError("metric must be 'cosine' or 'dot'")
        self.dim = dim
        self.metric = metric
        self._vectors: Optional[np.ndarray] = None
        self._records: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ write
    def add(
        self,
        ids: Sequence[str],
        vectors: np.ndarray,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2-D (n, dim)")
        if self.dim is None:
            self.dim = vectors.shape[1]
        elif vectors.shape[1] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {vectors.shape[1]}")
        if len(ids) != vectors.shape[0] or len(texts) != vectors.shape[0]:
            raise ValueError("ids, vectors and texts must have the same length")

        if self.metric == "cosine":
            vectors = _normalize(vectors)
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])
        metadatas = metadatas or [{} for _ in ids]
        for identifier, text, metadata in zip(ids, texts, metadatas):
            self._records.append({"id": identifier, "text": text, "metadata": dict(metadata)})

    # ------------------------------------------------------------------- read
    def records(self) -> List[Dict[str, Any]]:
        """Return a shallow copy of the stored records (id/text/metadata)."""
        return [dict(record) for record in self._records]

    def dense_scores(self, query: np.ndarray) -> np.ndarray:
        """Cosine/dot score of ``query`` against every stored vector.

        Aligned with :meth:`records` by index. Used by hybrid retrieval, which
        needs the full score vector rather than a truncated top-k.
        """
        if self._vectors is None or not len(self._records):
            return np.zeros(0, dtype=np.float32)
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dim:
            raise ValueError(f"Query dim {query.shape[0]} != store dim {self.dim}")
        if self.metric == "cosine":
            query = query / max(float(np.linalg.norm(query)), 1e-8)
        return (self._vectors @ query).astype(np.float32)

    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        min_score: Optional[float] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search the store.

        ``where`` is an exact-match metadata filter, e.g. ``{"finding": "Pneumonia"}``.
        It is applied before ranking, so top-k is drawn only from matching records.
        """
        if self._vectors is None or not len(self._records):
            return []
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dim:
            raise ValueError(f"Query dim {query.shape[0]} != store dim {self.dim}")
        if self.metric == "cosine":
            query = query / max(float(np.linalg.norm(query)), 1e-8)

        if where:
            candidates = np.array(
                [
                    index
                    for index, record in enumerate(self._records)
                    if all(record.get("metadata", {}).get(key) == value for key, value in where.items())
                ],
                dtype=int,
            )
            if candidates.size == 0:
                return []
        else:
            candidates = np.arange(len(self._records))

        scores = self._vectors[candidates] @ query
        order = np.argsort(scores)[::-1]  # positions into `candidates`
        results: List[SearchResult] = []
        for position in order:
            score = float(scores[position])
            if min_score is not None and score < min_score:
                break
            record = self._records[int(candidates[position])]
            results.append(
                SearchResult(
                    id=record["id"],
                    text=record["text"],
                    score=score,
                    metadata=record.get("metadata", {}),
                )
            )
            if len(results) >= k:
                break
        return results

    # -------------------------------------------------------------------- io
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        vectors = self._vectors if self._vectors is not None else np.zeros((0, self.dim or 0), np.float32)
        np.save(path / "vectors.npy", vectors)
        with (path / "records.jsonl").open("w", encoding="utf-8") as handle:
            for record in self._records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        with (path / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {"dim": self.dim, "metric": self.metric, "count": len(self._records)},
                handle,
                indent=2,
            )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "NumpyVectorStore":
        path = Path(path)
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No vector store at {path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        store = cls(dim=manifest.get("dim"), metric=manifest.get("metric", "cosine"))
        vectors_path = path / "vectors.npy"
        if vectors_path.exists():
            store._vectors = np.load(vectors_path).astype(np.float32)
        records_path = path / "records.jsonl"
        if records_path.exists():
            with records_path.open("r", encoding="utf-8") as handle:
                store._records = [json.loads(line) for line in handle if line.strip()]
        return store

    def __len__(self) -> int:
        return len(self._records)

    @property
    def is_empty(self) -> bool:
        return len(self._records) == 0


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-8, None)
