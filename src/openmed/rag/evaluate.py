"""Retrieval evaluation for the RAG layer.

Without a labelled set, "the RAG is accurate" is a guess. This module builds a
gold set from the ingested PubMed corpus and scores retrieval with the standard
IR metrics (Recall@k, MRR, nDCG), so any change to the embedder, chunking or the
hybrid weights can be measured instead of asserted.

Gold-set modes
--------------
* ``title``   - each abstract *title* is the query, its abstract is the gold
  document. A standard semantic-retrieval benchmark; note titles overlap with
  their abstracts, so it is a friendly (upper-bound-ish) signal.
* ``finding`` - a templated topical question per finding (e.g. "What is
  cardiomegaly ...?") whose gold set is every abstract tagged with that finding.
  Closer to real usage, but easier (many correct documents).

Usage::

    python -m openmed.cli build-gold --config configs/rag.yaml
    python -m openmed.cli eval-retrieval --config configs/rag.yaml
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from openmed.rag.embeddings import build_embedder
from openmed.rag.retriever import Retriever
from openmed.rag.vectorstore import NumpyVectorStore
from openmed.utils import ensure_dir, get_logger, save_json

LOGGER = get_logger(__name__)

FINDING_TEMPLATES = [
    "What is {finding} on a chest X-ray?",
    "How is {finding} diagnosed on chest radiograph?",
    "{finding} chest X-ray radiographic findings and causes",
]

# Off-topic queries with no relevant passage in the corpus. Their top-1 score
# bounds the "insufficient evidence" guard from above; the in-domain first-hit
# score bounds it from below.
NEGATIVE_QUERIES = [
    "How do I bake sourdough bread with a live starter culture?",
    "What is the capital of Australia and its population?",
    "Recommend a good hiking trail in the Alps for beginners.",
    "How does a car's four-stroke combustion engine work?",
    "Explain the plot of the movie Inception in detail.",
    "What is the best way to invest in index funds for retirement?",
    "How do I train a puppy to sit and stay?",
    "What is the chemical formula for table salt?",
]


def _pubmed_dir(corpus_dir: str | Path) -> Path:
    corpus = Path(corpus_dir)
    pubmed = corpus / "pubmed"
    return pubmed if pubmed.exists() else corpus


def build_gold_set(
    corpus_dir: str | Path,
    out_path: str | Path,
    modes: Sequence[str] = ("title", "finding"),
    max_per_finding: Optional[int] = None,
) -> Path:
    """Write a JSONL gold set from the raw PubMed corpus."""
    pubmed = _pubmed_dir(corpus_dir)
    queries: List[Dict] = []
    per_finding_pmids: Dict[str, List[str]] = {}

    for path in sorted(pubmed.glob("*.jsonl")):
        finding = path.stem
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                pmid = str(record.get("pmid") or "").strip()
                title = str(record.get("title") or "").strip()
                if not pmid or not title:
                    continue
                per_finding_pmids.setdefault(finding, []).append(pmid)
                if "title" in modes:
                    queries.append(
                        {"query": title, "query_type": "title", "finding": finding, "gold": [pmid]}
                    )
                count += 1
                if max_per_finding and count >= max_per_finding:
                    break

    if "finding" in modes:
        for finding, pmids in per_finding_pmids.items():
            for template in FINDING_TEMPLATES:
                queries.append(
                    {
                        "query": template.format(finding=finding.replace("_", " ")),
                        "query_type": "finding",
                        "finding": finding,
                        "gold": pmids,
                    }
                )

    if "negative" in modes:
        for question in NEGATIVE_QUERIES:
            queries.append({"query": question, "query_type": "negative", "finding": None, "gold": []})

    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as handle:
        for query in queries:
            handle.write(json.dumps(query, ensure_ascii=False) + "\n")
    LOGGER.info("Wrote %d gold queries -> %s", len(queries), out_path)
    return out_path


def load_gold_set(path: str | Path) -> List[Dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rank_of_first_hit(results, gold: set) -> Optional[int]:
    for rank, result in enumerate(results, start=1):
        if str(result.metadata.get("pmid")) in gold:
            return rank
    return None


def _distinct_hits(results, gold: set, k: int) -> int:
    """Number of distinct gold documents found in the top-k (dedupes chunks)."""
    seen = {
        str(result.metadata.get("pmid"))
        for result in results[:k]
        if str(result.metadata.get("pmid")) in gold
    }
    return len(seen)


def _ndcg_at_k(results, gold: set, k: int) -> float:
    seen: set = set()
    dcg = 0.0
    for rank, result in enumerate(results[:k], start=1):
        pmid = str(result.metadata.get("pmid"))
        if pmid in gold and pmid not in seen:
            seen.add(pmid)
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def _recall_at_k(results, gold: set, k: int) -> float:
    return _distinct_hits(results, gold, k) / len(gold) if gold else 0.0


def evaluate_retrieval(
    cfg,
    gold_path: str | Path,
    ks: Sequence[int] = (1, 3, 5, 10),
    lexical: Optional[bool] = None,
    limit: Optional[int] = None,
) -> Dict:
    """Score the configured retriever against the gold set.

    ``min_score`` is forced to 0 so we see the full ranking; the score of the
    first hit is recorded so the guard threshold can be calibrated afterwards.
    """
    ks = sorted({int(k) for k in ks})
    max_k = max(ks)
    store = NumpyVectorStore.load(cfg.vectorstore.path)
    if store.is_empty:
        raise RuntimeError(f"Vector store at {cfg.vectorstore.path} is empty; run `make ingest`.")
    embedder = build_embedder(cfg)

    use_lexical = bool(cfg.retrieval.get("lexical", False)) if lexical is None else lexical
    retriever = Retriever(
        store,
        embedder,
        top_k=max_k,
        min_score=0.0,
        lexical=use_lexical,
        alpha=float(cfg.retrieval.get("alpha", 0.5)),
        max_per_doc=int(cfg.retrieval.get("max_per_doc", 1)),
    )

    queries = load_gold_set(gold_path)
    if limit:
        queries = queries[:limit]

    recall: Dict[int, List[float]] = {k: [] for k in ks}
    ndcg: Dict[int, List[float]] = {k: [] for k in ks}
    reciprocal_ranks: List[float] = []
    first_hit_scores: List[float] = []
    top1_scores: List[float] = []
    negative_top1: List[float] = []
    diversity: List[float] = []
    by_type: Dict[str, List[float]] = {}

    for item in queries:
        gold = {str(pmid) for pmid in item["gold"]}
        results = retriever.retrieve(item["query"], k=max_k)
        top_score = float(results[0].score) if results else 0.0
        # Fraction of distinct source documents in the top-k (diversity).
        if results:
            parents = [str(r.metadata.get("pmid") or r.id.rsplit("::", 1)[0]) for r in results]
            diversity.append(len(set(parents)) / len(parents))
        if item["query_type"] == "negative":
            negative_top1.append(top_score)
            continue
        if results:
            top1_scores.append(top_score)
        rank = _rank_of_first_hit(results, gold)
        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
            first_hit_scores.append(float(results[rank - 1].score))
        else:
            reciprocal_ranks.append(0.0)
        for k in ks:
            recall[k].append(_recall_at_k(results, gold, k))
            ndcg[k].append(_ndcg_at_k(results, gold, k))
        by_type.setdefault(item["query_type"], []).append(rank is not None)

    metrics: Dict = {
        "n_queries": len(recall[ks[0]]),
        "n_negative": len(negative_top1),
        "lexical": use_lexical,
        "embedder": embedder.name,
        "store": {"path": str(cfg.vectorstore.path), "dim": store.dim, "count": len(store)},
        "mrr@10": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
    }
    for k in ks:
        metrics[f"recall@{k}"] = float(np.mean(recall[k])) if recall[k] else 0.0
        metrics[f"ndcg@{k}"] = float(np.mean(ndcg[k])) if ndcg[k] else 0.0
    metrics["hit_rate_by_type"] = {
        qtype: float(np.mean(hits)) for qtype, hits in by_type.items()
    }
    metrics[f"doc_diversity@{max_k}"] = float(np.mean(diversity)) if diversity else 0.0
    if top1_scores:
        scores = np.array(top1_scores, dtype=np.float32)
        metrics["top1_score"] = {
            "mean": float(scores.mean()),
            "p10": float(np.percentile(scores, 10)),
            "median": float(np.median(scores)),
            "min": float(scores.min()),
        }
    if first_hit_scores:
        hit_scores = np.array(first_hit_scores, dtype=np.float32)
        metrics["first_hit_score"] = {
            "mean": float(hit_scores.mean()),
            "p10": float(np.percentile(hit_scores, 10)),
            "p25": float(np.percentile(hit_scores, 25)),
        }
    if negative_top1:
        neg = np.array(negative_top1, dtype=np.float32)
        metrics["negative_top1_score"] = {
            "mean": float(neg.mean()),
            "p50": float(np.percentile(neg, 50)),
            "p90": float(np.percentile(neg, 90)),
            "max": float(neg.max()),
        }
    # Calibrate the guard: below the in-domain first-hit floor, above off-topic.
    if first_hit_scores and negative_top1:
        lo = float(np.percentile(np.array(first_hit_scores, dtype=np.float32), 10))
        hi = float(np.percentile(np.array(negative_top1, dtype=np.float32), 90))
        metrics["suggested_min_score"] = round((lo + hi) / 2.0, 3)
    return metrics


def report(metrics: Dict) -> str:
    lines = [
        f"embedder      : {metrics.get('embedder')}",
        f"hybrid(BM25)  : {metrics.get('lexical')}",
        f"queries       : {metrics.get('n_queries')} (+{metrics.get('n_negative', 0)} negative)",
        f"store         : {metrics.get('store')}",
        "",
        f"{'k':>3}  {'recall@k':>9}  {'ndcg@k':>8}",
    ]
    ks = sorted(int(key.split("@")[1]) for key in metrics if key.startswith("recall@"))
    for k in ks:
        lines.append(f"{k:>3}  {metrics[f'recall@{k}']:>9.3f}  {metrics[f'ndcg@{k}']:>8.3f}")
    lines.append(f"MRR@10        : {metrics.get('mrr@10', 0.0):.3f}")
    for key, value in metrics.items():
        if key.startswith("doc_diversity@"):
            lines.append(f"doc-diversity : {value:.3f}  (distinct source docs in top-k)")
    for qtype, value in (metrics.get("hit_rate_by_type") or {}).items():
        lines.append(f"hit-rate[{qtype:<7}]: {value:.3f}")
    if "top1_score" in metrics:
        top1 = metrics["top1_score"]
        lines.append(
            f"top1 score    : mean {top1['mean']:.3f} median {top1['median']:.3f} p10 {top1['p10']:.3f}"
        )
    if "first_hit_score" in metrics:
        first = metrics["first_hit_score"]
        lines.append(f"first-hit sc  : mean {first['mean']:.3f} p10 {first['p10']:.3f}")
    if "negative_top1_score" in metrics:
        neg = metrics["negative_top1_score"]
        lines.append(
            f"negative sc   : mean {neg['mean']:.3f} p90 {neg['p90']:.3f} max {neg['max']:.3f}"
        )
    if "suggested_min_score" in metrics:
        lines.append(f"suggested min_score : {metrics['suggested_min_score']}")
    return "\n".join(lines)


def evaluate_and_report(cfg, gold_path: str | Path, out_path: Optional[str | Path] = None, **kwargs) -> Dict:
    metrics = evaluate_retrieval(cfg, gold_path, **kwargs)
    print(report(metrics))
    if out_path:
        save_json(metrics, out_path)
        print(f"\nSaved -> {out_path}")
    return metrics
