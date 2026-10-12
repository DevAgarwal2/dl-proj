"""Semantic retrieval over the ingested medical corpus.

Two ranking signals are available:

* **dense** - cosine similarity in the embedding space (EmbeddingGemma / ST).
* **lexical** - Okapi BM25 over the stored chunks, which recovers exact medical
  terms and acronyms (``PTX``, ``CHF``, ``ground-glass``) that embeddings blur.

When ``lexical`` is enabled the two rankings are fused with Reciprocal Rank
Fusion (RRF). ``min_score`` is still applied to the *dense cosine* score so the
"insufficient evidence" guard remains a calibrated relevance check.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from openmed.rag.embeddings import BaseEmbedder
from openmed.rag.vectorstore import NumpyVectorStore, SearchResult

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_RRF_K = 60


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class _BM25:
    """Minimal Okapi BM25 index built over the store's chunk texts."""

    def __init__(self, texts: List[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs = [_tokenize(text) for text in texts]
        self.lengths = np.array([len(doc) for doc in self.docs], dtype=np.float32)
        self.avgdl = float(self.lengths.mean()) if len(self.lengths) else 0.0
        self.term_freqs = [Counter(doc) for doc in self.docs]
        df = Counter()
        for doc in self.docs:
            df.update(set(doc))
        n = len(self.docs)
        self.idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def scores(self, query: str) -> np.ndarray:
        n = len(self.docs)
        scores = np.zeros(n, dtype=np.float32)
        if not n:
            return scores
        for term in _tokenize(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, freq in enumerate(self.term_freqs):
                tf = freq.get(term, 0)
                if not tf:
                    continue
                norm = tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * self.lengths[index] / (self.avgdl or 1.0))
                )
                scores[index] += idf * norm
        return scores


class Retriever:
    def __init__(
        self,
        store: NumpyVectorStore,
        embedder: BaseEmbedder,
        top_k: int = 5,
        min_score: float = 0.0,
        lexical: bool = False,
        alpha: float = 0.5,
        max_per_doc: int = 1,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        self.min_score = min_score
        self.lexical = lexical
        self.alpha = alpha
        self.max_per_doc = max_per_doc
        self._bm25: Optional[_BM25] = None

    def _lexical_index(self) -> _BM25:
        if self._bm25 is None:
            self._bm25 = _BM25([record.get("text", "") for record in self.store.records()])
        return self._bm25

    @staticmethod
    def _parent_id(result: SearchResult) -> str:
        """Group chunks of the same source document (pmid, else id prefix)."""
        pmid = result.metadata.get("pmid")
        if pmid not in (None, ""):
            return f"pmid:{pmid}"
        return result.id.rsplit("::", 1)[0]

    def _diversify(self, results: List[SearchResult], top_k: int, max_per_doc: int) -> List[SearchResult]:
        """Cap how many chunks of the same document may appear in the top-k.

        Without this, a long abstract split into near-identical chunks floods the
        whole context window and the generator just copies its title.
        """
        if not max_per_doc or max_per_doc <= 0:
            return results[:top_k]
        counts: Dict[str, int] = {}
        selected: List[SearchResult] = []
        for result in results:
            parent = self._parent_id(result)
            if counts.get(parent, 0) >= max_per_doc:
                continue
            counts[parent] = counts.get(parent, 0) + 1
            selected.append(result)
            if len(selected) >= top_k:
                break
        return selected

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        where: Optional[dict] = None,
        max_per_doc: Optional[int] = None,
    ) -> List[SearchResult]:
        if self.store.is_empty:
            return []
        top_k = k or self.top_k
        cap = self.max_per_doc if max_per_doc is None else max_per_doc
        # Fetch a wider pool so de-duplication still leaves `top_k` results.
        pool = top_k * 12 if cap and cap > 0 else top_k
        vector = self.embedder.encode(
            [query], normalize=self.store.metric == "cosine", role="query"
        )[0]

        if not self.lexical:
            raw = self.store.search(vector, k=pool, min_score=self.min_score, where=where)
        else:
            raw = self._hybrid_search(query, vector, pool, where)
        return self._diversify(raw, top_k, cap)

    # ------------------------------------------------------------------ hybrid
    def _hybrid_search(
        self, query: str, vector: np.ndarray, top_k: int, where: Optional[dict]
    ) -> List[SearchResult]:
        records = self.store.records()
        dense = self.store.dense_scores(vector)
        lexical = self._lexical_index().scores(query)

        allowed = np.arange(len(records))
        if where:
            allowed = np.array(
                [
                    index
                    for index, record in enumerate(records)
                    if all(record.get("metadata", {}).get(key) == value for key, value in where.items())
                ],
                dtype=int,
            )
            if allowed.size == 0:
                return []

        def _rank_order(scores: np.ndarray) -> Dict[int, int]:
            order = allowed[np.argsort(scores[allowed])[::-1]]
            return {int(index): rank for rank, index in enumerate(order)}

        dense_rank = _rank_order(dense)
        lexical_rank = _rank_order(lexical)

        fused: List[tuple] = []
        for index in allowed:
            score = self.alpha / (_RRF_K + dense_rank[int(index)]) + (1 - self.alpha) / (
                _RRF_K + lexical_rank[int(index)]
            )
            fused.append((score, int(index)))
        fused.sort(reverse=True)

        results: List[SearchResult] = []
        for _, index in fused:
            if dense[index] < self.min_score:
                continue
            record = records[index]
            results.append(
                SearchResult(
                    id=record["id"],
                    text=record["text"],
                    score=float(dense[index]),
                    metadata=record.get("metadata", {}),
                )
            )
            if len(results) >= top_k:
                break
        return results

    @staticmethod
    def format_evidence(results: List[SearchResult], max_chars: int = 1000) -> str:
        """Render retrieved chunks as a numbered, citable block."""
        lines = []
        for index, result in enumerate(results, start=1):
            title = result.metadata.get("title") or result.metadata.get("source", "source")
            snippet = " ".join(result.text.split())[:max_chars]
            lines.append(f"[{index}] {title} (score {result.score:.3f})\n{snippet}")
        return "\n\n".join(lines)
