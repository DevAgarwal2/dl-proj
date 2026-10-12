"""Answer-faithfulness evaluation for the RAG layer.

Retrieval metrics say whether the *right passage* was found; they say nothing
about whether the generator stuck to it. This module asks the local LLM to
act as a judge and scores two RAGAS-style properties:

* **faithfulness** - is every claim in the answer supported by the retrieved
  evidence (no invented facts)?
* **answer_relevance** - does the answer actually address the question?

The judge returns strict JSON, which is parsed leniently (small models wander).
Requires the generation server (``make llama``).
"""

from __future__ import annotations

import json
import re
from statistics import mean
from typing import Dict, List, Optional, Sequence

from openmed.data.labels import NIH_LABELS
from openmed.rag.llm import LLMClient, LLMError
from openmed.utils import get_logger, save_json

LOGGER = get_logger(__name__)

JUDGE_SYSTEM = (
    "You are a strict evaluation judge for a medical question-answering system. "
    "You are given a QUESTION, an ANSWER, and the EVIDENCE the answer was allowed "
    "to use. Judge the ANSWER objectively and reply with ONLY a JSON object:\n"
    '{"faithfulness": <0.0-1.0>, "answer_relevance": <0.0-1.0>, '
    '"unsupported_claims": [<short strings>], "reason": "<one sentence>"}\n'
    "faithfulness = fraction of claims supported by the evidence. "
    "answer_relevance = how well the answer addresses the question. "
    "Do not add commentary outside the JSON."
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_CITATION_RE = re.compile(r"\[(\d+)\]")
# A bracketed token that is NOT a numeric citation, e.g. [hydrostatic pressure],
# [0.793], [x] - a small-model artifact.
_ARTIFACT_RE = re.compile(r"\[(?![0-9]+\])[^\]\n]{1,40}\]")


def default_qa_set() -> List[str]:
    """One grounded question per finding - a lightweight smoke eval set."""
    return [f"What are the radiographic signs and common causes of {label.replace('_', ' ')}?"
            for label in NIH_LABELS]


def answer_diagnostics(answer: str, evidence: Sequence[str], n_citations: int) -> Dict:
    """Cheap, deterministic checks for the failure modes a small LLM exhibits."""
    text = " ".join((answer or "").split())
    lower = text.lower()
    citations = [int(match) for match in _CITATION_RE.findall(text)]
    artifacts = _ARTIFACT_RE.findall(text)

    # Title echo: the answer literally contains a full source title.
    title_echo = False
    for item in evidence:
        title = " ".join(str(item).split())
        if len(title) >= 25 and title.lower() in lower:
            title_echo = True
            break

    valid_citations = bool(citations) and all(1 <= c <= n_citations for c in citations)
    in_range = all(1 <= c <= n_citations for c in citations)
    return {
        "answer_words": len(text.split()),
        "n_citation_marks": len(citations),
        "has_citation": bool(citations),
        "citations_in_range": in_range,
        "citations_valid": valid_citations,
        "title_echo": title_echo,
        "n_artifacts": len(artifacts),
        "artifacts": artifacts[:5],
    }


def _usable(row: Dict) -> bool:
    """A coarse 'the answer is actually usable' heuristic."""
    return (
        row.get("answer_words", 0) >= 25
        and not row.get("title_echo")
        and row.get("n_artifacts", 0) == 0
        and row.get("citations_valid")
    )


def _parse_judgement(text: str) -> Dict:
    match = _JSON_RE.search(text or "")
    raw = (text or "").strip()[:400]
    if not match:
        return {"faithfulness": None, "answer_relevance": None, "unsupported_claims": [], "reason": "unparseable", "_raw": raw}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"faithfulness": None, "answer_relevance": None, "unsupported_claims": [], "reason": "invalid json", "_raw": raw}
    parsed = _coerce_scores(parsed)
    parsed.setdefault("unsupported_claims", [])
    parsed.setdefault("reason", "")
    parsed["_raw"] = raw
    return parsed


_QUALITATIVE = {
    "high": 0.9,
    "very high": 0.95,
    "strong": 0.9,
    "yes": 0.9,
    "fully": 0.95,
    "medium": 0.6,
    "moderate": 0.6,
    "partial": 0.5,
    "partially": 0.5,
    "somewhat": 0.5,
    "low": 0.3,
    "weak": 0.3,
    "poor": 0.2,
    "no": 0.0,
    "none": 0.0,
}


def _coerce_scores(parsed: Dict) -> Dict:
    """Map qualitative words and 0-10 scales onto a 0-1 score per metric."""
    for key in ("faithfulness", "answer_relevance"):
        value = parsed.get(key)
        number: Optional[float] = None
        if isinstance(value, str):
            lowered = value.strip().lower()
            for word, mapped in _QUALITATIVE.items():
                if word in lowered:
                    number = mapped
                    break
            if number is None:
                try:
                    number = float(lowered)
                except ValueError:
                    number = None
        else:
            try:
                number = float(value)
            except (TypeError, ValueError):
                number = None
        if number is None:
            parsed[key] = None
        else:
            if number > 1.0:  # model answered on a 0-10 scale
                number = number / 10.0
            parsed[key] = max(0.0, min(1.0, number))
    return parsed


def judge_answer(llm: LLMClient, question: str, answer: str, evidence: Sequence[str]) -> Dict:
    evidence_block = "\n\n".join(f"[{i}] {e}" for i, e in enumerate(evidence, start=1)) or "(none)"
    prompt = (
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer}\n\n"
        f"EVIDENCE:\n{evidence_block}\n\n"
        "Return the JSON verdict."
    )
    try:
        # Small models drift from JSON; force valid JSON where the server supports it.
        raw = llm.complete(
            prompt,
            system=JUDGE_SYSTEM,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    except (LLMError, TypeError):
        try:
            raw = llm.complete(prompt, system=JUDGE_SYSTEM, temperature=0.0)
        except LLMError as exc:
            return {"faithfulness": None, "answer_relevance": None, "unsupported_claims": [], "reason": str(exc)}
    return _parse_judgement(raw)


def evaluate_answers(
    cfg,
    questions: Optional[Sequence[str]] = None,
    out_path: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict:
    """Run the RAG pipeline on each question and judge the answer."""
    from openmed.rag.pipeline import OpenMedRAG

    rag = OpenMedRAG.from_config(cfg)
    questions = list(questions or default_qa_set())
    if limit:
        questions = questions[:limit]

    rows: List[Dict] = []
    for question in questions:
        explanation = rag.answer(question)
        evidence = [item.get("text", "") for item in explanation.evidence]
        titles = [
            str(item.get("metadata", {}).get("title") or item.get("metadata", {}).get("source", ""))
            for item in explanation.evidence
        ]
        verdict = judge_answer(rag.llm, question, explanation.text, evidence)
        diagnostics = answer_diagnostics(explanation.text, titles, len(explanation.citations))
        rows.append(
            {
                "question": question,
                "answer": explanation.text,
                "n_citations": len(explanation.citations),
                "refused": "Insufficient evidence" in (explanation.text or ""),
                **verdict,
                **diagnostics,
            }
        )

    faithful = [row["faithfulness"] for row in rows if row["faithfulness"] is not None]
    relevant = [row["answer_relevance"] for row in rows if row["answer_relevance"] is not None]
    metrics = {
        "n_questions": len(rows),
        "n_judged": len(faithful),
        "mean_faithfulness": float(mean(faithful)) if faithful else None,
        "mean_answer_relevance": float(mean(relevant)) if relevant else None,
        "refusal_rate": float(mean([row["refused"] for row in rows])) if rows else 0.0,
        "usable_rate": float(mean([_usable(row) for row in rows])) if rows else 0.0,
        "title_echo_rate": float(mean([row["title_echo"] for row in rows])) if rows else 0.0,
        "artifact_rate": float(mean([row["n_artifacts"] > 0 for row in rows])) if rows else 0.0,
        "citation_ok_rate": float(mean([row["citations_valid"] for row in rows])) if rows else 0.0,
        "citation_in_range_rate": float(mean([row["citations_in_range"] for row in rows])) if rows else 0.0,
        "has_citation_rate": float(mean([row["has_citation"] for row in rows])) if rows else 0.0,
        "mean_answer_words": float(mean([row["answer_words"] for row in rows])) if rows else 0.0,
        "rows": rows,
    }
    if out_path:
        save_json(metrics, out_path)
    LOGGER.info(
        "Faithfulness eval: %d questions, mean_faithfulness=%s",
        len(rows),
        metrics["mean_faithfulness"],
    )
    return metrics
