"""The OpenMed RAG pipeline: model finding -> retrieval -> grounded explanation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from openmed.rag.embeddings import build_embedder
from openmed.rag.llm import LLMClient
from openmed.rag.prompts import (
    SYSTEM_PROMPT,
    build_finding_prompt,
    build_qa_prompt,
    finalize_answer,
    format_citations,
)
from openmed.rag.retriever import Retriever
from openmed.rag.vectorstore import NumpyVectorStore, SearchResult
from openmed.utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class Citation:
    index: int
    doc_id: str
    title: str
    source: str
    score: float
    snippet: str


@dataclass
class Explanation:
    text: str
    citations: List[Citation] = field(default_factory=list)
    evidence: List[Dict] = field(default_factory=list)
    finding: Optional[str] = None
    probability: Optional[float] = None
    model: Optional[str] = None
    prompt: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def references(self) -> str:
        return format_citations(
            [SearchResult(c.doc_id, c.snippet, c.score, {"title": c.title, "source": c.source}) for c in self.citations]
        )


class OpenMedRAG:
    """Retrieval-augmented explanation on top of a local LFM2.5 model."""

    def __init__(self, retriever: Retriever, llm: LLMClient) -> None:
        self.retriever = retriever
        self.llm = llm

    @classmethod
    def from_config(cls, cfg) -> "OpenMedRAG":
        store = NumpyVectorStore.load(cfg.vectorstore.path)
        embedder = build_embedder(cfg)
        retriever = Retriever(
            store,
            embedder,
            top_k=int(cfg.retrieval.get("top_k", 5)),
            min_score=float(cfg.retrieval.get("min_score", 0.0)),
            lexical=bool(cfg.retrieval.get("lexical", False)),
            alpha=float(cfg.retrieval.get("alpha", 0.5)),
            max_per_doc=int(cfg.retrieval.get("max_per_doc", 1)),
        )
        llm = LLMClient(
            base_url=cfg.llm.base_url,
            model=cfg.llm.model,
            temperature=float(cfg.llm.get("temperature", 0.2)),
            top_p=float(cfg.llm.get("top_p", 0.9)),
            max_tokens=int(cfg.llm.get("max_tokens", 512)),
            timeout=int(cfg.llm.get("timeout", 120)),
            strip_reasoning_output=bool(cfg.llm.get("strip_reasoning", True)),
        )
        return cls(retriever, llm)

    # ------------------------------------------------------------- retrieval
    def _retrieve(
        self, query: str, k: Optional[int], where: Optional[Dict] = None
    ) -> List[SearchResult]:
        results = self.retriever.retrieve(query, k=k, where=where)
        if not results:
            LOGGER.warning("No evidence retrieved for query: %s", query)
        return results

    @staticmethod
    def _citations(results: List[SearchResult]) -> List[Citation]:
        return [
            Citation(
                index=index,
                doc_id=result.id,
                title=str(result.metadata.get("title") or result.metadata.get("source", "")),
                source=str(result.metadata.get("source", "")),
                score=float(result.score),
                snippet=" ".join(result.text.split())[:240],
            )
            for index, result in enumerate(results, start=1)
        ]

    def _generate(
        self,
        prompt: str,
        results: List[SearchResult],
        finding: Optional[str] = None,
        probability: Optional[float] = None,
    ) -> Explanation:
        text = self.llm.complete(prompt, system=SYSTEM_PROMPT)
        text = finalize_answer(text, len(results))
        return Explanation(
            text=text,
            citations=self._citations(results),
            evidence=[result.to_dict() for result in results],
            finding=finding,
            probability=probability,
            model=self.llm.model,
            prompt=prompt,
        )

    @staticmethod
    def _insufficient(
        finding: Optional[str] = None, probability: Optional[float] = None
    ) -> Explanation:
        return Explanation(
            text=(
                "Insufficient evidence: no passage in the corpus was relevant enough, "
                "so I didn't guess. Name the condition in your question "
                "(e.g. \"What causes emphysema?\"), or use the Diagnose tab and pick a "
                "finding. You can also add literature to data/corpus and re-run "
                "`make ingest`."
            ),
            citations=[],
            evidence=[],
            finding=finding,
            probability=probability,
            model=None,
        )

    # ---------------------------------------------------------------- public
    def explain_finding(
        self,
        finding: str,
        probability: float,
        question: Optional[str] = None,
        k: Optional[int] = None,
    ) -> Explanation:
        """Explain a classifier prediction, grounded in retrieved literature."""
        query = f"{finding} chest X-ray radiographic findings causes diagnosis"
        # Prefer passages tagged with this finding (each abstract carries its
        # finding label); fall back to an unfiltered search for free-text findings.
        results = self._retrieve(query, k, where={"finding": finding})
        if not results:
            results = self._retrieve(query, k)
        if not results:
            return self._insufficient(finding, probability)
        evidence = self.retriever.format_evidence(results)
        prompt = build_finding_prompt(finding, probability, evidence, question)
        return self._generate(prompt, results, finding=finding, probability=probability)

    def explain_predictions(self, findings: List[Dict], k: Optional[int] = None) -> List[Explanation]:
        """Explain a list of ``{"label", "probability"}`` predictions."""
        return [
            self.explain_finding(item["label"], float(item["probability"]), k=k)
            for item in findings
        ]

    def answer(self, question: str, k: Optional[int] = None) -> Explanation:
        results = self._retrieve(question, k)
        if not results:
            return self._insufficient()
        evidence = self.retriever.format_evidence(results)
        prompt = build_qa_prompt(question, evidence)
        return self._generate(prompt, results)
