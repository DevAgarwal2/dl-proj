"""Text embedding backends.

Three backends are provided:

* ``llama-server`` - EmbeddingGemma-2 (or any GGUF embedding model) served by a
  local ``llama-server`` over its OpenAI-compatible ``/v1/embeddings`` API. This
  is the recommended backend: it is a stronger retriever than the biomedical
  sentence-transformer and runs fully on-device.
* ``sentence-transformers`` - a HuggingFace sentence embedder (e.g. PubMedBERT).
* ``hashing`` - a deterministic, dependency-free bag-of-words projection used
  for tests and fully offline runs.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from typing import List, Optional, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def resolve_device(device: str | None) -> str:
    """Map 'auto'/None to a concrete torch device sentence-transformers accepts."""
    if device and device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class BaseEmbedder:
    name = "base"
    dim = 0

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        normalize: bool = True,
        role: str = "document",
    ) -> np.ndarray:
        """Embed ``texts``.

        ``role`` is ``"query"`` or ``"document"``. Prompt-conditioned models
        (EmbeddingGemma) use it to pick the correct task prefix; models that do
        not care ignore it.
        """
        raise NotImplementedError


class HashingEmbedder(BaseEmbedder):
    """Deterministic hashed bag-of-words embedding (offline / tests)."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _encode_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = _TOKEN_RE.findall(text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return vector

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        normalize: bool = True,
        role: str = "document",
    ) -> np.ndarray:
        matrix = np.vstack([self._encode_one(text) for text in texts]) if texts else np.zeros((0, self.dim), np.float32)
        if normalize and len(matrix):
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.clip(norms, 1e-8, None)
        return matrix.astype(np.float32)


class SentenceTransformerEmbedder(BaseEmbedder):
    """Wrapper around ``sentence-transformers`` (e.g. S-PubMedBert-MS-MARCO)."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.device = resolve_device(device)
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        normalize: bool = True,
        role: str = "document",
    ) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class LlamaServerEmbedder(BaseEmbedder):
    """EmbeddingGemma-2 (or any GGUF embedder) served by a local ``llama-server``.

    EmbeddingGemma is prompt-conditioned: a *query* and a *document* must be
    wrapped in the task prefixes the model was trained with, otherwise retrieval
    quality degrades. We apply those prefixes here so callers pass raw text and
    only need to say whether it is a query or a document.

    Start the server with::

        llama-server -m embeddinggemma-2-Q8_0.gguf --embedding --pooling mean \\
            --port 8081 -ngl 99 -c 8192

    (see ``scripts/run_embed_server.sh``).
    """

    QUERY_TEMPLATE = "task: search result | query: {text}"
    DOCUMENT_TEMPLATE = "title: {title} | text: {text}"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8081",
        model_name: str = "embeddinggemma-2",
        timeout: int = 600,
        dim: Optional[int] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.name = f"llama-server:{model_name}"
        self.model_name = model_name
        self.timeout = int(timeout)
        self.dim = int(dim or 768)

    # ------------------------------------------------------------------ format
    def _format(self, text: str, role: str) -> str:
        text = " ".join(str(text).split())
        if role == "query":
            return self.QUERY_TEMPLATE.format(text=text)
        return self.DOCUMENT_TEMPLATE.format(title="none", text=text)

    # -------------------------------------------------------------------- http
    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.URLError as exc:  # pragma: no cover - network path
            raise RuntimeError(
                f"Could not reach embedding server at {self.base_url} ({exc}). "
                "Start it with `make embed-server`."
            ) from exc

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=5) as response:
                return response.status == 200
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------ encode
    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        normalize: bool = True,
        role: str = "document",
    ) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        prompts = [self._format(text, role) for text in texts]

        vectors: List[List[float]] = []
        for start in range(0, len(prompts), max(int(batch_size), 1)):
            batch = prompts[start : start + max(int(batch_size), 1)]
            payload = self._post("/v1/embeddings", {"input": batch, "model": self.model_name})
            rows = sorted(payload.get("data", []), key=lambda item: item.get("index", 0))
            vectors.extend(row["embedding"] for row in rows)

        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim == 1:  # server returned a single flat vector
            matrix = matrix.reshape(1, -1)
        if matrix.shape[0] != len(texts):
            raise RuntimeError(f"Embedding server returned {matrix.shape[0]} vectors for {len(texts)} inputs")
        if matrix.shape[1] != self.dim:
            self.dim = int(matrix.shape[1])
        if normalize:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.clip(norms, 1e-8, None)
        return matrix.astype(np.float32)


def build_embedder(cfg) -> BaseEmbedder:
    """Build an embedder from a RAG config's ``embeddings`` section."""
    section = cfg.embeddings if hasattr(cfg, "embeddings") else cfg
    backend = str(section.get("backend", "llama-server")).lower()
    if backend in {"hashing", "hash", "offline"}:
        return HashingEmbedder(dim=int(section.get("dim", 512)))
    if backend in {"llama-server", "llama_server", "llamaserver", "embeddinggemma", "gemma"}:
        return LlamaServerEmbedder(
            base_url=str(section.get("base_url", "http://127.0.0.1:8081")),
            model_name=str(section.get("model_name", "embeddinggemma-2")),
            timeout=int(section.get("timeout", 600)),
            dim=section.get("dim"),
        )
    if backend in {"sentence-transformers", "st", "sbert"}:
        return SentenceTransformerEmbedder(
            section.get("model_name", "pritamdeka/S-PubMedBert-MS-MARCO"),
            device=section.get("device", "cpu"),
        )
    raise ValueError(f"Unknown embedding backend: {backend}")
