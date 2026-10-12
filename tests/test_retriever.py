import unittest

from openmed.rag.embeddings import HashingEmbedder
from openmed.rag.retriever import Retriever
from openmed.rag.vectorstore import NumpyVectorStore


class RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.embedder = HashingEmbedder(dim=256)
        texts = [
            "Cardiomegaly is an enlarged cardiac silhouette on chest X-ray.",
            "Pneumothorax is air in the pleural space with lung collapse.",
            "Pleural effusion is fluid accumulation in the pleural space.",
        ]
        vectors = self.embedder.encode(texts)
        self.store = NumpyVectorStore(metric="cosine")
        self.store.add(
            ids=[f"d{i}" for i in range(len(texts))],
            vectors=vectors,
            texts=texts,
            metadatas=[{"title": t.split(" is ")[0], "source": f"d{i}.txt"} for i, t in enumerate(texts)],
        )
        self.retriever = Retriever(self.store, self.embedder, top_k=2, min_score=-1.0)

    def test_retrieve_returns_matching_doc(self):
        results = self.retriever.retrieve("collapsed lung due to air in pleural space")
        self.assertTrue(results)
        self.assertIn("Pneumothorax", results[0].text)

    def test_format_evidence_numbering(self):
        results = self.retriever.retrieve("enlarged heart", k=1)
        block = self.retriever.format_evidence(results)
        self.assertTrue(block.startswith("[1]"))

    def test_empty_store(self):
        empty = Retriever(NumpyVectorStore(metric="cosine"), self.embedder)
        self.assertEqual(empty.retrieve("anything"), [])


if __name__ == "__main__":
    unittest.main()
