import unittest

from openmed.rag.embeddings import HashingEmbedder
from openmed.rag.llm import strip_reasoning
from openmed.rag.pipeline import OpenMedRAG
from openmed.rag.prompts import build_finding_prompt
from openmed.rag.retriever import Retriever
from openmed.rag.vectorstore import NumpyVectorStore


class FakeLLM:
    model = "fake-lfm"

    def __init__(self):
        self.last_prompt = None

    def complete(self, prompt, system=None, **overrides):
        self.last_prompt = prompt
        return "Pneumonia is an infection [1]. This is not medical advice."

    def health(self):
        return True


def build_rag():
    embedder = HashingEmbedder(dim=256)
    texts = [
        "Pneumonia is an infection of the lungs with consolidation on X-ray.",
        "Cardiomegaly is an enlarged heart silhouette.",
    ]
    store = NumpyVectorStore(metric="cosine")
    store.add(
        ids=["d0", "d1"],
        vectors=embedder.encode(texts),
        texts=texts,
        metadatas=[{"title": "Pneumonia", "source": "p.txt"}, {"title": "Cardiomegaly", "source": "c.txt"}],
    )
    return OpenMedRAG(Retriever(store, embedder, top_k=2, min_score=-1.0), FakeLLM())


class PipelineTests(unittest.TestCase):
    def test_explain_finding_has_citations(self):
        rag = build_rag()
        explanation = rag.explain_finding("Pneumonia", 0.86)
        self.assertEqual(explanation.finding, "Pneumonia")
        self.assertAlmostEqual(explanation.probability, 0.86)
        self.assertIn("not medical advice", explanation.text)
        self.assertEqual(len(explanation.citations), 2)
        self.assertEqual(explanation.citations[0].index, 1)

    def test_prompt_contains_finding_and_probability(self):
        prompt = build_finding_prompt("Pneumonia", 0.86, "[1] some evidence")
        self.assertIn("Pneumonia", prompt)
        self.assertIn("86.0%", prompt)
        self.assertIn("[1] some evidence", prompt)

    def test_answer_question(self):
        rag = build_rag()
        explanation = rag.answer("What is pneumonia?")
        self.assertIsNone(explanation.finding)
        self.assertTrue(explanation.citations)

    def test_strip_reasoning(self):
        raw = "<think>hidden chain</think>The answer is [1]."
        self.assertEqual(strip_reasoning(raw), "The answer is [1].")


if __name__ == "__main__":
    unittest.main()
