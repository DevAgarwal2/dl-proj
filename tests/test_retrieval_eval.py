"""Offline tests for the retrieval-eval and hybrid-retrieval additions."""

from __future__ import annotations

import unittest

import numpy as np

from openmed.rag.embeddings import HashingEmbedder, LlamaServerEmbedder
from openmed.rag.evaluate import _ndcg_at_k, _recall_at_k
from openmed.rag.faithfulness import _parse_judgement, answer_diagnostics
from openmed.rag.prompts import finalize_answer
from openmed.rag.retriever import Retriever, _BM25
from openmed.rag.vectorstore import NumpyVectorStore, SearchResult


class LlamaServerEmbedderFormatTest(unittest.TestCase):
    def test_query_and_document_prefixes(self):
        embedder = LlamaServerEmbedder()
        self.assertEqual(
            embedder._format("what is ptx?", "query"),
            "task: search result | query: what is ptx?",
        )
        self.assertEqual(
            embedder._format("pneumothorax   on  xray", "document"),
            "title: none | text: pneumothorax on xray",
        )


class Bm25Test(unittest.TestCase):
    def test_exact_term_ranks_first(self):
        docs = [
            "pneumothorax with a visible pleural line and absent lung markings",
            "cardiomegaly with an enlarged cardiac silhouette",
            "pulmonary edema with perihilar opacities",
        ]
        bm25 = _BM25(docs)
        scores = bm25.scores("cardiomegaly silhouette")
        self.assertEqual(int(np.argmax(scores)), 1)


class HybridRetrieverTest(unittest.TestCase):
    def _store(self):
        store = NumpyVectorStore(metric="cosine")
        embedder = HashingEmbedder(dim=256)
        texts = [
            "pneumothorax pleural line absent lung markings",
            "cardiomegaly enlarged cardiac silhouette",
            "pulmonary edema perihilar opacities",
        ]
        vectors = embedder.encode(texts)
        store.add([f"d{i}" for i in range(len(texts))], vectors, texts, [{"pmid": str(i)} for i in range(len(texts))])
        return store, embedder

    def test_hybrid_returns_relevant_doc(self):
        store, embedder = self._store()
        retriever = Retriever(store, embedder, top_k=2, min_score=0.0, lexical=True, alpha=0.6)
        results = retriever.retrieve("cardiomegaly silhouette")
        self.assertTrue(results)
        self.assertEqual(results[0].id, "d1")

    def test_where_filter_applies(self):
        store, embedder = self._store()
        retriever = Retriever(store, embedder, top_k=3, min_score=0.0, lexical=True)
        results = retriever.retrieve("anything", where={"pmid": "2"})
        self.assertEqual([r.id for r in results], ["d2"])


class MetricsTest(unittest.TestCase):
    def _results(self, pmids):
        return [SearchResult(id=f"d{i}", text="", score=1.0, metadata={"pmid": p}) for i, p in enumerate(pmids)]

    def test_recall_multigold(self):
        results = self._results(["9", "1", "8", "2"])
        gold = {"1", "2"}
        self.assertAlmostEqual(_recall_at_k(results, gold, 3), 0.5)
        self.assertAlmostEqual(_recall_at_k(results, gold, 5), 1.0)

    def test_ndcg_single_gold(self):
        results = self._results(["x", "gold", "y"])
        self.assertAlmostEqual(_ndcg_at_k(results, {"gold"}, 10), 1.0 / np.log2(3))


class JudgeParseTest(unittest.TestCase):
    def test_qualitative_label_mapped(self):
        verdict = _parse_judgement('{"faithfulness": 0.8, "answer_relevance": "high", "reason": "ok"}')
        self.assertAlmostEqual(verdict["faithfulness"], 0.8)
        self.assertAlmostEqual(verdict["answer_relevance"], 0.9)

    def test_ten_point_scale_normalised(self):
        verdict = _parse_judgement('{"faithfulness": 9, "answer_relevance": 7}')
        self.assertAlmostEqual(verdict["faithfulness"], 0.9)
        self.assertAlmostEqual(verdict["answer_relevance"], 0.7)

    def test_unparseable_returns_none(self):
        verdict = _parse_judgement("I think the answer is fine, roughly 8/10.")
        self.assertIsNone(verdict["faithfulness"])
        self.assertEqual(verdict["reason"], "unparseable")


class AnswerDiagnosticsTest(unittest.TestCase):
    def test_flags_title_echo_and_artifacts(self):
        diag = answer_diagnostics(
            "Types and mechanisms of pulmonary atelectasis. [hydrostatic pressure] [1]",
            ["Types and mechanisms of pulmonary atelectasis."],
            5,
        )
        self.assertTrue(diag["title_echo"])
        self.assertGreaterEqual(diag["n_artifacts"], 1)

    def test_clean_answer_passes(self):
        diag = answer_diagnostics(
            "Atelectasis shows crowded vessels and displaced fissures on the radiograph [1]. "
            "It commonly results from airway obstruction [2].",
            ["Some totally different title."],
            5,
        )
        self.assertFalse(diag["title_echo"])
        self.assertEqual(diag["n_artifacts"], 0)
        self.assertTrue(diag["citations_valid"])


class FinalizeAnswerTest(unittest.TestCase):
    def test_drops_out_of_range_citation_and_collapses_disclaimer(self):
        cleaned = finalize_answer(
            "Pneumonia shows air bronchograms [1] and [7] here. "
            "This is a research tool, not medical advice. "
            "This is a research tool, not medical advice.",
            3,
        )
        self.assertIn("[1]", cleaned)
        self.assertNotIn("[7]", cleaned)
        self.assertEqual(cleaned.count("This is a research tool, not medical advice."), 1)

    def test_empty_stays_empty(self):
        self.assertEqual(finalize_answer("", 5), "")


if __name__ == "__main__":
    unittest.main()
