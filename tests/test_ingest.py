import tempfile
import unittest
from pathlib import Path

from openmed.rag.embeddings import HashingEmbedder
from openmed.rag.ingest import chunk_text, ingest_corpus, iter_documents


class IngestTests(unittest.TestCase):
    def test_chunk_text_overlap(self):
        text = " ".join(f"word{i}" for i in range(400))
        chunks = chunk_text(text, chunk_size=200, overlap=50, min_chars=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 200 for c in chunks))

    def test_short_text_single_chunk(self):
        self.assertEqual(chunk_text("short text", chunk_size=200, min_chars=5), ["short text"])

    def test_iter_documents_txt_and_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "note.txt").write_text("Pneumonia is an infection of the lungs.")
            (root / "lit.jsonl").write_text(
                '{"title": "Effusion", "text": "Pleural effusion is fluid around the lung."}\n'
            )
            docs = list(iter_documents(root))
        self.assertEqual(len(docs), 2)
        titles = {title for _, title, _ in docs}
        self.assertIn("Effusion", titles)

    def test_ingest_and_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus"
            corpus.mkdir()
            (corpus / "a.txt").write_text(
                "Cardiomegaly refers to an enlarged heart silhouette on a chest radiograph. " * 5
            )
            (corpus / "b.txt").write_text(
                "Pneumothorax is air in the pleural space causing lung collapse. " * 5
            )
            store_path = Path(tmp) / "store"
            store = ingest_corpus(
                corpus, store_path, embedder=HashingEmbedder(256),
                chunk_size=200, overlap=40, min_chars=20,
            )
            self.assertGreater(len(store), 0)
            self.assertTrue((store_path / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
