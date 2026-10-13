"""OpenMed web UI — a simple, demo-friendly screen.

The point of this UI is a clear story anyone can follow in 30 seconds:

    upload a chest X-ray  →  see what the model found and where it looked
                          →  get a plain-language explanation with sources

Technical controls live under "Advanced settings" in the sidebar, collapsed by
default, so the demo stays clean.

Run from the project root:
    make ui
    # or: PYTHONPATH=src .venv/bin/streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

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
@st.cache_resource(show_spinner="Loading the vision model...")
def get_predictor(checkpoint: str, use_demo: bool, device: str, backbone: str):
    from openmed.inference import Predictor

    path = Path(checkpoint)
    if use_demo or not path.exists():
        return Predictor(checkpoint=None, device=device, backbone=backbone, pretrained=False), False
    return Predictor(checkpoint=path, device=device), True


@st.cache_resource(show_spinner="Loading the search model...")
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


@st.cache_data(ttl=10, show_spinner=False)
def is_up(url: str, timeout: int = 3) -> bool:
    import requests

    try:
        return requests.get(url, timeout=timeout).status_code == 200
    except Exception:  # noqa: BLE001
        return False


def load_store(path: str):
    try:
        return NumpyVectorStore.load(path)
    except FileNotFoundError:
        return None


# --------------------------------------------------------------------------- #
# Advanced settings (sidebar, collapsed by default)
# --------------------------------------------------------------------------- #
st.sidebar.title("🩺 OpenMed")
st.sidebar.caption("Chest X-ray classifier **+** cited explanation, on your laptop.")

with st.sidebar.expander("⚙️ Advanced settings", expanded=False):
    checkpoint = st.text_input("Vision checkpoint", DEFAULT_CHECKPOINT)
    device = st.selectbox("Device", ["auto", "mps", "cpu"], index=0)
    backbone = st.selectbox(
        "Vision backbone", ["densenet121", "efficientnet_b0", "resnet50", "vit_base_patch16_224"], index=0
    )
    use_demo = st.checkbox("Use untrained demo model", value=not Path(checkpoint).exists())

    st.divider()
    rag_config = st.text_input("RAG config", DEFAULT_RAG_CONFIG)
    embed_base_url = st.text_input("Embedding server", "http://127.0.0.1:8081")
    backend = st.selectbox("Embedding backend", ["llama-server", "hashing"], index=0)
    embedding_model = st.text_input("Embedding model", "embeddinggemma-2")
    store_path = st.text_input("Vector store", "data/vectorstore/openmed")
    corpus_dir = st.text_input("Corpus folder", "data/corpus")

    st.divider()
    top_k = st.slider("Passages used for the answer", 1, 10, 5)
    min_relevance = st.slider(
        "Minimum relevance", 0.0, 1.0, 0.72, 0.01,
        help="Passages below this score are ignored. If nothing passes, OpenMed "
             "refuses to answer instead of guessing.",
    )

    c1, c2 = st.columns(2)
    rebuild = c1.button("Rebuild index", use_container_width=True)
    check_llm = c2.button("Test writer", use_container_width=True)

    if rebuild:
        with st.spinner("Reading the corpus and building the index..."):
            try:
                embedder = get_embedder(backend, embedding_model, device, embed_base_url)
                ingest_corpus(corpus_dir, store_path, embedder=embedder, chunk_size=2000, overlap=200)
                st.success("Index rebuilt.")
                st.cache_resource.clear()
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Index build failed: {exc}")

# --------------------------------------------------------------------------- #
# Resolve everything + a simple status panel
# --------------------------------------------------------------------------- #
predictor, model_trained = get_predictor(checkpoint, use_demo, device, backbone)
store = load_store(store_path)
try:
    llm = get_llm(rag_config)
except Exception:  # noqa: BLE001
    llm = None

embed_ok = is_up(f"{embed_base_url}/health") if backend == "llama-server" else True
llm_ok = is_up("http://127.0.0.1:8080/health") if llm else False

st.sidebar.markdown("### Status")
st.sidebar.write(f"{'🟢' if model_trained else '🟡'} **Vision model**"
                 + ("" if model_trained else " — untrained demo"))
st.sidebar.write(f"{'🟢' if store else '🔴'} **Medical library** — "
                 f"{len(store) if store else 0} passages")
st.sidebar.write(f"{'🟢' if embed_ok else '🔴'} **Search server** (:8081)")
st.sidebar.write(f"{'🟢' if llm_ok else '🔴'} **Writer server** (:8080)")

if check_llm:
    if llm is None:
        st.sidebar.error("No writer configured.")
    elif llm.health():
        st.sidebar.success("Writer is online.")
    else:
        st.sidebar.error("Writer offline. Run: `make llama`")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_rag() -> OpenMedRAG:
    if store is None:
        raise RuntimeError("No medical library found. Click 'Rebuild index' in Advanced settings.")
    if llm is None:
        raise RuntimeError("Writer not configured.")
    embedder = get_embedder(backend, embedding_model, device, embed_base_url)
    if store.dim and embedder.dim and store.dim != embedder.dim:
        raise RuntimeError(
            f"Index/model mismatch ({store.dim} vs {embedder.dim}). Rebuild the index."
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


def show_finding_bars(findings) -> None:
    for item in findings:
        pct = item["probability"] * 100
        st.write(f"**{item['label']}** — {pct:.0f}%")
        st.progress(min(max(item["probability"], 0.0), 1.0))


def show_explanation(explanation) -> None:
    refused = explanation.model is None
    if refused:
        st.warning(explanation.text)
        return
    st.success("Explanation (written from the sources below)")
    st.markdown(explanation.text)
    if explanation.citations:
        st.markdown("##### 📚 Sources used")
        for c in explanation.citations:
            title = c.title or c.source or "source"
            with st.expander(f"[{c.index}] {title}  ·  relevance {c.score:.2f}"):
                st.write(c.snippet)
    else:
        st.info("No evidence was retrieved.")


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🩺 OpenMed")
st.markdown(
    "**Upload a chest X-ray and get a plain-language explanation with real medical sources — "
    "all running on this machine, no internet.**"
)
st.caption("A research tool for learning. It is **not** a medical diagnosis.")

if not model_trained:
    st.warning(
        "⚠️ No trained checkpoint found — showing the **untrained demo model**. "
        "Findings will be meaningless. Train one with `make train-chestmnist`."
    )

tab_demo, tab_ask, tab_how = st.tabs(["🔎 Diagnose an X-ray", "📖 Ask the literature", "ℹ️ How it works"])


# --------------------------------------------------------------------------- #
# Tab 1 — Diagnose
# --------------------------------------------------------------------------- #
with tab_demo:
    st.markdown("##### How to use this screen")
    st.markdown(
        "1. **Upload** a chest X-ray below. &nbsp;&nbsp; "
        "2. See **what the model found** and **where it looked**. &nbsp;&nbsp; "
        "3. Press **Explain** to get the reasoning with **sources**."
    )

    uploaded = st.file_uploader(
        "Drop a chest X-ray here (PNG / JPG)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
    )

    if uploaded is not None:
        from PIL import Image

        image = Image.open(uploaded).convert("RGB")

        st.divider()
        st.markdown("#### Step 1 · What the model sees")
        col_img, col_cam = st.columns(2)
        with col_img:
            st.image(image, caption="Your X-ray", use_container_width=True)
        with st.spinner("Analysing the image..."):
            findings = predictor.predict(image, top_k=len(NIH_LABELS))
            top = predictor.predict_with_cam(image, top_k=1)[0]
        with col_cam:
            st.image(
                top["overlay"],
                caption=f"Where it looked — Grad-CAM for '{top['label']}'",
                use_container_width=True,
            )
        st.caption(
            "🔥 The red/yellow areas of the heat-map are the image regions that most "
            "influenced the top prediction."
        )

        st.divider()
        st.markdown("#### Step 2 · What the model found")
        ranked = findings[:6]
        show_finding_bars(ranked)
        st.caption("Higher % = the model is more confident that condition is present. "
                   "This is a statistical guess, not a diagnosis.")

        st.divider()
        st.markdown("#### Step 3 · Why — explained with medical sources")
        labels = [f["label"] for f in findings]
        choice = st.selectbox("Explain this finding", labels, index=0)
        probability = next((f["probability"] for f in findings if f["label"] == choice), 0.5)
        question = st.text_input("Optional: ask something specific (leave blank for the default)", "")

        if st.button("✨ Explain", type="primary"):
            with st.spinner("Searching the literature and writing the explanation..."):
                try:
                    explanation = make_rag().explain_finding(
                        choice, float(probability), question or None, k=top_k
                    )
                    show_explanation(explanation)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not generate the explanation: {exc}")
                    st.info("Make sure the two servers are running: `make embed-server` and `make llama`.")
    else:
        st.info("👆 Upload a chest X-ray to start. No image handy? Try the button below.")
        with st.expander("No image? Explain a condition by name instead"):
            manual = st.selectbox("Condition", list(NIH_LABELS), key="manual_finding")
            manual_probability = st.slider("Assumed probability", 0.0, 1.0, 0.86, 0.01)
            if st.button("✨ Explain", key="manual_explain"):
                with st.spinner("Searching the literature and writing the explanation..."):
                    try:
                        show_explanation(make_rag().explain_finding(manual, manual_probability, k=top_k))
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Could not generate the explanation: {exc}")


# --------------------------------------------------------------------------- #
# Tab 2 — Ask the literature
# --------------------------------------------------------------------------- #
with tab_ask:
    st.markdown("Ask a question and OpenMed answers using **only** the medical papers it has, with citations.")
    st.caption("Tip: name the condition (e.g. 'pneumonia'). Vague, non-medical questions are politely refused.")

    examples = [
        "What causes pneumonia?",
        "What are the radiographic features of pleural effusion?",
        "How is cardiomegaly assessed on a chest X-ray?",
        "What is emphysema?",
    ]
    if "qa_question" not in st.session_state:
        st.session_state.qa_question = ""
    cols = st.columns(len(examples))
    for col, example in zip(cols, examples):
        if col.button(example, use_container_width=True):
            st.session_state.qa_question = example
            st.session_state.qa_run = True
            st.rerun()

    question = st.text_input("Your question", key="qa_question")
    if (st.button("Ask", type="primary") or st.session_state.pop("qa_run", False)) and question:
        with st.spinner("Searching and writing..."):
            try:
                show_explanation(make_rag().answer(question, k=top_k))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not answer: {exc}")


# --------------------------------------------------------------------------- #
# Tab 3 — How it works
# --------------------------------------------------------------------------- #
with tab_how:
    st.markdown("#### The 30-second version")
    st.markdown(
        "A normal AI X-ray tool gives you a bare number like *'Cardiomegaly 89%'* — "
        "no reason, no proof. OpenMed turns that number into something you can check:\n\n"
        "**1. Predict** → a fine-tuned CNN scores 14 chest conditions.\n\n"
        "**2. Show where** → a heat-map (Grad-CAM) marks the image region behind the prediction.\n\n"
        "**3. Find evidence** → a search model pulls the most relevant passages from ~500 real abstracts.\n\n"
        "**4. Explain** → a small local language model writes a short, cited explanation — "
        "and refuses if the evidence is too weak."
    )
    st.divider()
    st.markdown("#### What it is *not*")
    st.markdown(
        "- **Not a diagnosis** — no clinical validation, not a medical device.\n"
        "- **Chest X-rays only** — exactly 14 conditions; not CT, MRI or other body parts.\n"
        "- **Research-grade accuracy** — low-resolution images, so false alarms happen."
    )
    st.divider()
    with st.expander("Setup commands (for the demo)"):
        st.code(
            "make embed-server     # start the search server   (:8081)\n"
            "make llama            # start the writer server   (:8080)\n"
            "make ingest           # build the medical library\n"
            "make ui               # this screen",
            language="bash",
        )
