"""Prompt templates for grounded medical explanation."""

from __future__ import annotations

import re
from typing import Sequence

SYSTEM_PROMPT = (
    "You are OpenMed, a careful clinical decision-support assistant. "
    "You explain a chest X-ray classifier's output using ONLY the numbered medical "
    "evidence provided. Write 3-5 complete sentences of plain prose in your own words. "
    "Hard rules: (1) NEVER copy, quote or list the titles of the sources. "
    "(2) Support every statement with a bracketed citation like [1] or [2]; only use "
    "indices that appear in the evidence. At least three sentences must end with a "
    "citation. (3) If the evidence does not cover something, write 'the retrieved "
    "evidence does not specify' instead of guessing or inventing. "
    "(4) Do not output raw numbers, scores, or placeholder tokens such as [x]. "
    "(5) Never present this as a diagnosis. End with exactly one sentence: "
    "'This is a research tool, not medical advice.'"
)

_CITATION_RE = re.compile(r"\[(\d+)\]")
_DISCLAIMER_RE = re.compile(r"This is a research tool, not medical advice\.?", re.IGNORECASE)
_DISCLAIMER = "This is a research tool, not medical advice."


def finalize_answer(text: str, n_citations: int) -> str:
    """Clean small-model output: drop out-of-range citations, collapse the
    duplicated disclaimer, and guarantee a single trailing disclaimer."""
    text = " ".join((text or "").split())
    # Remove citation indices that point outside the provided evidence.
    text = _CITATION_RE.sub(
        lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_citations else "", text
    )
    text = _DISCLAIMER_RE.sub("", text)
    text = " ".join(text.split()).strip()
    if not text:
        return text
    return f"{text} {_DISCLAIMER}"


def build_finding_prompt(
    finding: str,
    probability: float,
    evidence: str,
    question: str | None = None,
) -> str:
    probability_pct = f"{probability * 100:.1f}%"
    focus = question or (
        f"Explain what the finding '{finding}' is, how it appears on a chest X-ray, "
        f"and its common causes."
    )
    return (
        f"A chest X-ray classifier produced the following output:\n"
        f"- Finding: {finding}\n"
        f"- Probability: {probability_pct}\n\n"
        f"Retrieved medical evidence (numbered):\n{evidence}\n\n"
        f"Task: {focus}\n\n"
        "Write 3-5 complete sentences of prose that cover: (a) what the finding is, "
        "(b) its radiographic appearance / signs, and (c) common causes. "
        "Cite evidence as [n]. Do NOT list, quote, or copy the source titles, and do "
        "not echo the evidence list. If a part is unsupported, say so."
    )


def build_qa_prompt(question: str, evidence: str) -> str:
    return (
        f"Retrieved medical evidence (numbered):\n{evidence}\n\n"
        f"Question: {question}\n\n"
        "Answer in 3-5 complete sentences of your own words, using only the evidence "
        "above. Cite sources with [n]. Do NOT list or copy the source titles. If the "
        "evidence does not answer the question, say 'the retrieved evidence does not "
        "specify' rather than guessing."
    )


def format_citations(results: Sequence) -> str:
    lines = []
    for index, result in enumerate(results, start=1):
        title = result.metadata.get("title") or result.metadata.get("source", "source")
        source = result.metadata.get("source", "")
        lines.append(f"[{index}] {title}" + (f" - {source}" if source and source != title else ""))
    return "\n".join(lines)
