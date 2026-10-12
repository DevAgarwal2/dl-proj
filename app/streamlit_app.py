"""OpenMed browser UI (Streamlit).

Run from the project root:
    make ui
    # or: PYTHONPATH=src .venv/bin/streamlit run app/streamlit_app.py

The UI degrades gracefully: with no trained checkpoint it offers a clearly
labelled *untrained* demo model so the whole pipeline can still be exercised,
and the RAG tab works from the seed corpus with the offline hashing embedder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openmed.config import Config, load_config  # noqa: E402
from openmed.data.labels import NIH_LABELS  # noqa: E402
from openmed.rag.embeddings import build_embedder  # noqa: E402
from openmed.rag.ingest import ingest_corpus  # noqa: E402
from openmed.rag.llm import LLMClient  # noqa: E402
from openmed.rag.pipeline import OpenMedRAG  # noqa: E402
from openmed.rag.retriever import Retriever  # noqa: E402
from openmed.rag.vectorstore import NumpyVectorStore  # noqa: E402
from openmed.utils import default_checkpoint  # noqa: E402

st.set_page_config(page_title="OpenMed", page_icon="🩺", layout="wide")

DEFAULT_CHECKPOINT = default_checkpoint()
DEFAULT_RAG_CONFIG = "configs/rag.yaml"


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading vision model...")
def get_predictor(checkpoint: str, use_demo: bool, device: str, backbone: str):
    from openmed.inference import Predictor

    path = Path(checkpoint)
    if use_demo or not path.exists():
        return Predictor(
            checkpoint=None, device=device, backbone=backbone, pretrained=False
        ), f"untrained demo backbone ({backbone})"
    return Predictor(checkpoint=path, device=device), "trained checkpoint"


@st.cache_resource(show_spinner="Loading embedder...")
def get_embedder(backend: str, model_name: str, device: str, base_url: str = ""):
    section = {"backend": backend, "model_name": model_name, "device": device}
    if base_url:
        section["base_url"] = base_url
    return build_embedder(Config(section))


@st.cache_resource(show_spinner=False)
def get_llm(config_path: str) -> LLMClient:
    cfg = load_config(config_path)
    return LLMClient(
        base_url=cfg.llm.base_url,
        model=cfg.llm.model,
        temperature=float(cfg.llm.get("temperature", 0.2)),
        top_p=float(cfg.llm.get("top_p", 0.9)),
        max_tokens=int(cfg.llm.get("max_tokens", 512)),
        timeout=int(cfg.llm.get("timeout", 120)),
        strip_reasoning_output=bool(cfg.llm.get("strip_reasoning", True)),
    )


@st.cache_data(ttl=15, show_spinner=False)
def check_health(base_url: str, timeout: int = 3) -> bool:
    import requests

    try:
        return requests.get(base_url.replace("/v1", "") + "/health", timeout=timeout).status_code == 200
    except Exception:
        return False


def load_store(path: str):
    try:
        return NumpyVectorStore.load(path)
    except FileNotFoundError:
        return None


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("🩺 OpenMed")
st.sidebar.caption("Medical vision model + retrieval-augmented explanation")

with st.sidebar.expander("⚙️ Model", expanded=True):
    checkpoint = st.text_input("Checkpoint path", DEFAULT_CHECKPOINT)
    backbone = st.selectbox(
        "Backbone", ["densenet121", "efficientnet_b0", "resnet50", "vit_base_patch16_224"], index=0
    )
    device = st.selectbox("Device", ["auto", "mps", "cpu"], index=0)
    checkpoint_exists = Path(checkpoint).exists()
    use_demo = st.checkbox(
        "Use untrained demo model", value=not checkpoint_exists,
        help="Only for testing the UI. Predictions from this model are meaningless.",
    )

with st.sidebar.expander("📚 Retrieval (RAG)", expanded=True):
    rag_config = st.text_input("RAG config", DEFAULT_RAG_CONFIG)
    backend = st.selectbox(
        "Embedding backend", ["llama-server", "sentence-transformers", "hashing"], index=0
    )
    embedding_model = st.text_input(
        "Embedding model",
        "embeddinggemma-2" if backend == "llama-server" else "NeuML/pubmedbert-base-embeddings",
    )
    embed_base_url = st.text_input("Embed server URL", "http://127.0.0.1:8081")
    store_path = st.text_input("Vector store path", "data/vectorstore/openmed")
    top_k = st.slider("Top-k evidence", 1, 10, 3)
    min_relevance = st.slider(
        "Minimum relevance", 0.0, 1.0, 0.72, 0.01,
        help="Passages scoring below this are discarded. If none pass, OpenMed "
             "refuses to answer instead of hallucinating. Calibrated for "
             "EmbeddingGemma-2 via `make eval-retrieval`.",
    )
    threshold = st.slider("Prediction threshold", 0.0, 0.9, 0.0, 0.05)
    corpus_dir = st.text_input("Corpus directory", "data/corpus")

col_a, col_b = st.sidebar.columns(2)
rebuild = col_a.button("Build store", use_container_width=True)
check_llm = col_b.button("Test LLM", use_container_width=True)

# Build status
predictor, model_kind = get_predictor(checkpoint, use_demo, device, backbone)
store = load_store(store_path)
try:
    llm = get_llm(rag_config)
    llm_ok = check_health(llm.base_url)
except Exception as exc:  # pragma: no cover
    llm, llm_ok = None, False
    st.sidebar.error(f"LLM config error: {exc}")

st.sidebar.markdown("### Status")
st.sidebar.write(f"**Model:** {model_kind}")
st.sidebar.write(f"**Vector store:** {len(store) if store else 0} chunks")
st.sidebar.write(f"**llama-server:** {'🟢 online' if llm_ok else '🔴 offline'}")
if backend == "hashing":
    st.sidebar.warning("Hashing embedder: offline but low retrieval quality. "
                       "Use llama-server (EmbeddingGemma) for real retrieval.")
if backend == "llama-server":
    try:
        import requests

        embed_ok = requests.get(f"{embed_base_url}/health", timeout=3).status_code == 200
    except Exception:  # noqa: BLE001
        embed_ok = False
    st.sidebar.write(f"**Embed server:** {'🟢 online' if embed_ok else '🔴 offline (make embed-server)'}")

if rebuild:
    with st.spinner("Chunking and embedding the corpus..."):
        try:
            embedder = get_embedder(backend, embedding_model, device, embed_base_url)
            ingest_corpus(corpus_dir, store_path, embedder=embedder, chunk_size=900, overlap=150)
            st.sidebar.success("Vector store rebuilt.")
            st.cache_resource.clear()
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"Ingest failed: {exc}")

if check_llm:
    if llm is None:
        st.sidebar.error("No LLM configured.")
    elif llm.health():
        try:
            reply = llm.complete("Reply with the single word: ready", max_tokens=16)
            st.sidebar.success(f"LLM online: {reply[:60]!r}")
        except Exception as exc:
            st.sidebar.error(f"LLM call failed: {exc}")
    else:
        st.sidebar.error("llama-server is not reachable. Run: make llama")


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("OpenMed")
st.caption(
    "Fine-tuned chest X-ray classifier → Grad-CAM → literature retrieval → "
    "grounded explanation from a local LFM2.5 model."
)

if not predictor.is_trained:
    st.warning(
        "⚠️ **No trained checkpoint found** — running an *untrained* demo backbone. "
        "Predictions and Grad-CAM maps are meaningless. Train a model with "
        "`make prepare && make train`, or point the sidebar at a checkpoint."
    )

tab_predict, tab_ask, tab_setup = st.tabs(["🫁 Diagnose", "❓ Ask the literature", "🛠️ Setup"])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def render_explanation(explanation) -> None:
    if not explanation.citations and explanation.model is None:
        st.warning(explanation.text)
        return
    st.markdown(explanation.text)
    if explanation.citations:
        st.markdown("##### References")
        for citation in explanation.citations:
            title = citation.title or citation.source or "source"
            with st.expander(f"[{citation.index}] {title} — score {citation.score:.3f}"):
                st.write(citation.snippet)
    else:
        st.info("No evidence was retrieved. Ingest a corpus or check the vector store path.")


def make_rag() -> OpenMedRAG:
    if store is None:
        raise RuntimeError(f"No vector store at {store_path}. Click 'Build store'.")
    if llm is None:
        raise RuntimeError("LLM is not configured.")
    embedder = get_embedder(backend, embedding_model, device, embed_base_url)
    if store.dim and embedder.dim and store.dim != embedder.dim:
        raise RuntimeError(
            f"Embedding dim mismatch: store={store.dim}, embedder={embedder.dim}. "
            "Rebuild the store with the selected backend."
        )
    try:
        retrieval_cfg = load_config(rag_config).retrieval
    except Exception:  # noqa: BLE001
        retrieval_cfg = Config({})
    retriever = Retriever(
        store, embedder, top_k=top_k, min_score=min_relevance,
        lexical=bool(retrieval_cfg.get("lexical", False)),
        alpha=float(retrieval_cfg.get("alpha", 0.5)),
    )
    return OpenMedRAG(retriever, llm)


# --------------------------------------------------------------------------- #
# Diagnose tab
# --------------------------------------------------------------------------- #
with tab_predict:
    uploaded = st.file_uploader(
        "Upload a chest X-ray (PNG / JPG)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
    )

    if uploaded is not None:
        from PIL import Image

        image = Image.open(uploaded).convert("RGB")
        left, right = st.columns(2)
        left.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Running the classifier..."):
            findings = predictor.predict(image, top_k=len(NIH_LABELS), threshold=threshold or None)
            top = predictor.predict_with_cam(image, top_k=1)[0]
        right.image(top["overlay"], caption=f"Grad-CAM — {top['label']}", use_container_width=True)

        chart_df = pd.DataFrame(
            {"probability": [f["probability"] for f in findings]},
            index=[f["label"] for f in findings],
        ).sort_values("probability")
        st.bar_chart(chart_df, horizontal=True)

        labels = [f["label"] for f in findings] or list(NIH_LABELS)
        default = labels[0] if labels else NIH_LABELS[0]
        choice = st.selectbox("Explain a finding with RAG", labels, index=0)
        probability = next(
            (f["probability"] for f in findings if f["label"] == choice),
            st.session_state.get("manual_probability", 0.5),
        )
        question = st.text_input("Optional question", "")

        if st.button("Explain with RAG", type="primary"):
            with st.spinner("Retrieving evidence and generating..."):
                try:
                    rag = make_rag()
                    explanation = rag.explain_finding(
                        choice, float(probability), question or None, k=top_k
                    )
                    render_explanation(explanation)
                except Exception as exc:
                    st.error(f"RAG failed: {exc}")
    else:
        st.info("Upload an image to see ranked findings, a Grad-CAM overlay and an explanation.")
        st.markdown("#### No image? Explain a finding manually")
        manual = st.selectbox("Finding", list(NIH_LABELS), key="manual_finding")
        manual_probability = st.slider("Probability", 0.0, 1.0, 0.86, 0.01)
        st.session_state["manual_probability"] = manual_probability
        if st.button("Explain with RAG", key="manual_explain"):
            with st.spinner("Retrieving evidence and generating..."):
                try:
                    rag = make_rag()
                    render_explanation(rag.explain_finding(manual, manual_probability, k=top_k))
                except Exception as exc:
                    st.error(f"RAG failed: {exc}")


# --------------------------------------------------------------------------- #
# Ask tab
# --------------------------------------------------------------------------- #
with tab_ask:
    st.caption(
        "Name the condition in your question — vague questions with no medical "
        "term are refused rather than guessed at."
    )
    EXAMPLES = [
        "What causes pneumonia?",
        "What are the radiographic features of pleural effusion?",
        "How is cardiomegaly assessed on a chest X-ray?",
        "What is emphysema?",
    ]
    if "qa_question" not in st.session_state:
        st.session_state.qa_question = ""
    example_cols = st.columns(len(EXAMPLES))
    for col, example in zip(example_cols, EXAMPLES):
        if col.button(example, use_container_width=True):
            st.session_state.qa_question = example
            st.session_state.qa_run = True
            st.rerun()

    question = st.text_input(
        "Ask a question about the ingested literature", key="qa_question"
    )
    run = st.button("Ask", type="primary")
    if (run or st.session_state.pop("qa_run", False)) and question:
        with st.spinner("Retrieving and generating..."):
            try:
                rag = make_rag()
                render_explanation(rag.answer(question, k=top_k))
            except Exception as exc:
                st.error(f"RAG failed: {exc}")
    st.caption(
        "Answers are grounded only in the ingested corpus and are not medical advice."
    )


# --------------------------------------------------------------------------- #
# Setup tab
# --------------------------------------------------------------------------- #
with tab_setup:
    st.markdown("#### 1. Train the vision model")
    st.code(
        "# Put NIH ChestX-ray14 under data/raw/nih-chestxray14/\n"
        "make prepare\n"
        "make smoke      # 1-epoch sanity run\n"
        "make train      # full fine-tuning on Apple Silicon (MPS)",
        language="bash",
    )
    st.markdown("#### 2. Build the retrieval index")
    st.code(
        "# Put .md/.txt/.jsonl literature in data/corpus/\n"
        "make ingest\n"
        "# or use the 'Build store' button in the sidebar",
        language="bash",
    )
    st.markdown("#### 3. Start the local LLM")
    st.code(
        "make llama        # llama-server + LFM2.5-1.2B-Instruct-Q6_K.gguf\n"
        "make llama-stop",
        language="bash",
    )
    st.markdown("#### 4. (Optional) REST API")
    st.code("make serve        # http://127.0.0.1:8000/docs", language="bash")

    st.markdown("#### Current configuration")
    st.json(
        {
            "checkpoint": checkpoint,
            "checkpoint_exists": checkpoint_exists,
            "model_kind": model_kind,
            "device": predictor.device,
            "image_size": predictor.image_size,
            "backend": backend,
            "store_path": store_path,
            "store_chunks": len(store) if store else 0,
            "store_dim": store.dim if store else None,
            "llm_model": getattr(llm, "model", None),
            "llm_online": llm_ok,
        }
    )
