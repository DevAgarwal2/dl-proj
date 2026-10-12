import tempfile
import unittest
from pathlib import Path

import numpy as np

from openmed.rag.vectorstore import NumpyVectorStore


class VectorStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = NumpyVectorStore(metric="cosine")
        vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32)
        self.store.add(
            ids=["a", "b", "c"],
            vectors=vectors,
            texts=["alpha", "beta", "gamma"],
            metadatas=[{"title": "A"}, {"title": "B"}, {"title": "C"}],
        )

    def test_search_orders_by_similarity(self):
        results = self.store.search(np.array([1.0, 0.0], dtype=np.float32), k=2)
        self.assertEqual(results[0].id, "a")
        self.assertGreater(results[0].score, 0.9)
        self.assertEqual(len(results), 2)

    def test_min_score_filters(self):
        results = self.store.search(np.array([1.0, 0.0], dtype=np.float32), k=5, min_score=0.8)
        self.assertEqual([r.id for r in results], ["a"])

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.store.save(tmp)
            loaded = NumpyVectorStore.load(tmp)
        self.assertEqual(len(loaded), 3)
        self.assertEqual(loaded.dim, 2)
        result = loaded.search(np.array([0.0, 1.0], dtype=np.float32), k=1)[0]
        self.assertEqual(result.id, "b")
        self.assertEqual(result.metadata["title"], "B")

    def test_dimension_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.store.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=1)

    def test_where_filter_restricts_candidates(self):
        store = NumpyVectorStore(metric="cosine")
        store.add(
            ids=["a", "b", "c"],
            vectors=np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            texts=["alpha", "beta", "gamma"],
            metadatas=[{"finding": "X"}, {"finding": "Y"}, {"finding": "X"}],
        )
        results = store.search(
            np.array([1.0, 0.0], dtype=np.float32), k=5, where={"finding": "X"}
        )
        self.assertEqual([r.id for r in results], ["a", "c"])
        self.assertEqual(store.search(np.array([1.0, 0.0], dtype=np.float32), k=5, where={"finding": "Z"}), [])


if __name__ == "__main__":
    unittest.main()
