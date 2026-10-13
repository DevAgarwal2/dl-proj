# OpenMed

**An AI that reads chest X-rays and explains its answer in plain language, with real medical references — running entirely on your own computer.**

> ⚠️ OpenMed is a **research / learning project**, not a medical device. It does not
> diagnose anyone. Always talk to a doctor.

---

## 1. What problem does it solve?

When an AI looks at a chest X-ray it usually gives you just a number:

```
Cardiomegaly: 89%
```

That number is hard to trust or act on. You don't know *why* the model said it,
*where* in the image it looked, or *what it even means*. It's a black box.

OpenMed turns that black box into something a human can actually use:

```
Cardiomegaly (89%) — the model looked at the heart region (heatmap).

"The chest X-ray shows an enlarged cardiac silhouette, a key sign of
 cardiomegaly [1]. Common causes include hypertension, cardiomyopathy and
 valvular disease [2]. This is a research tool, not medical advice."

References:
  [1] Identifying cardiomegaly in chest X-rays: a cross-sectional study...
  [2] Cardiomegaly on chest radiographs as a predictor of heart disease...
```

So OpenMed does four things, in one pipeline:

1. **Predicts** which of 14 common chest conditions might be present.
2. **Shows where** — a heat map (Grad-CAM) highlights the image region behind each prediction.
3. **Finds evidence** — searches a local library of ~500 real medical abstracts.
4. **Explains** — a small language model writes a short, cited explanation using only that evidence.

Everything runs locally. No internet, no cloud, no per-query cost.

---

## 2. How it works

```
                 Chest X-ray image
                        │
                        ▼
        ┌───────────────────────────────┐
        │  Trained CNN (DenseNet121)    │   14 chest conditions
        │  → probabilities + Grad-CAM   │   + a heat map
        └───────────────────────────────┘
                        │  finding + probability
                        ▼
        ┌───────────────────────────────┐
        │  EmbeddingGemma-2 (on-device) │   turns the query into a vector
        │  → semantic search            │   over 590 medical text chunks
        └───────────────────────────────┘
                        │  top 5 distinct, most-relevant abstracts
                        │  (refuses if nothing is relevant enough)
                        ▼
        ┌───────────────────────────────┐
        │  LFM2.5-1.2B-Instruct (local) │   writes the grounded explanation
        │  → answer + [1][2] citations  │
        └───────────────────────────────┘
```

Two small servers run in the background, both powered by `llama.cpp`:
an **embedding server** (retrieval) and a **chat server** (writing the explanation).

---

## 3. What's inside

| Part | What it is |
|---|---|
| **Vision model** | DenseNet121 (pretrained on ImageNet) fine-tuned on **MedMNIST ChestMNIST** — a chest X-ray dataset derived from NIH ChestX-ray14, with the same 14 labels. |
| **Explainability** | Grad-CAM — shows the image region that drove each prediction. |
| **Retrieval** | On-device **EmbeddingGemma-2** embeddings + a tiny numpy vector store (no FAISS/Chroma needed). Optional BM25 hybrid search. |
| **Generator** | Local **LFM2.5-1.2B-Instruct** via `llama-server`. |
| **Corpus** | ~500 open-access PubMed abstracts (fetched automatically) + a small seed corpus. |
| **Interfaces** | Streamlit web UI, FastAPI REST API, and a command-line tool. |
| **Evaluation** | A retrieval benchmark (Recall@k / MRR / nDCG) and an answer-quality check (faithfulness, citations, failure modes). |
| **Tests** | 33 offline unit tests. |

---

## 4. How accurate is it?

Accuracy is different at each stage — here are the real, measured numbers.

**1. The classifier** (measured on 22,433 unseen test images)

| Metric | Result |
|---|---|
| Mean AUROC (all 14 conditions) | **0.784** |
| Best condition (Cardiomegaly) | 0.895 |
| Hardest condition (Nodule) | 0.664 |

AUROC 0.78 is honest and normal for 64×64 images. (Full-resolution X-rays would score higher.)
The model is a **research prototype** — it produces false alarms on normal X-rays.

**2. The retrieval** (measured on 547 labelled queries + 8 off-topic queries)

| Metric | Result |
|---|---|
| Recall@5 (right paper in top 5) | **0.922** |
| MRR@10 (how highly the right paper ranks) | **0.968** |
| Source diversity (no duplicate papers) | **1.000** |
| Off-topic questions refused (not answered) | ✅ reliable |

**3. The explanation** (measured on 14 finding questions)

| Metric | Result |
|---|---|
| Usable answers (coherent, cited, no artifacts) | **100%** |
| Valid citations | **100%** |
| Faithfulness (self-judged) | 0.90 |

**Honest caveats:** the explanation faithfulness is scored by the *same* model that wrote it, so treat it as indicative, not a medical audit. The classifier and retrieval numbers are hard measurements.

---

## 5. Getting started

### Requirements
- Python 3.10+
- [`llama.cpp`](https://github.com/ggml-org/llama.cpp) (`llama-server` on your PATH — e.g. `brew install llama.cpp`)
  > ⚠️ EmbeddingGemma-2 needs a **recent** `llama-server` build. If you see
  > `unknown model architecture: 'gemma-embedding2'`, point `OPENMED_EMBED_BIN`
  > at a newer build of `llama-server`.
- ~1.5 GB free disk for the two models
- A Mac with Apple Silicon (uses MPS) or a CPU. CUDA also works.

### Step 1 — Install
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
make doctor          # checks your environment
```

### Step 2 — Download the two models into `models/`

```bash
# generation model
hf download LiquidAI/LFM2.5-1.2B-Instruct-GGUF LFM2.5-1.2B-Instruct-Q6_K.gguf --local-dir models

# embedding model
hf download ggml-org/embeddinggemma-2-GGUF embeddinggemma-2-Q8_0.gguf --local-dir models
# (any GGUF build of EmbeddingGemma-2 works; set OPENMED_EMBED_GGUF if the path differs)
```

### Step 3 — Build the vision model (optional — the training config auto-downloads data)

```bash
make smoke             # 1-epoch sanity run on a small subset
make train-chestmnist  # full training run → runs/chestmnist/best.pt
```
> You can skip this — the UI falls back to an untrained demo model so the RAG side
> is still explorable.

### Step 4 — Build the search index
```bash
make embed-server      # start the embedding server   (:8081)
make fetch-corpus      # download ~500 PubMed abstracts
make ingest            # embed the corpus into the vector store
```

### Step 5 — Start the writer and use it
```bash
make llama             # start the generator server    (:8080)

make query Q="What is cardiomegaly?"   # ask from the terminal
make ui                                # open the web UI at http://127.0.0.1:8501
make serve                             # or the REST API at http://127.0.0.1:8000/docs
```

---

## 6. Using it

**Visual one-pager** — open `docs/demo.html` in a browser for a slide-ready
summary (problem → solution → results) you can show in a presentation.

**Live terminal demo** — one command, no UI:
```bash
make embed-server && make llama     # start both servers
make demo                           # uses a bundled sample X-ray
make demo IMAGE=samples/images/single_000018.png
```
It prints the ranked findings, saves a Grad-CAM heat-map to `outputs/`, then
writes the cited explanation.

**Web UI (`make ui` → http://127.0.0.1:8501)** — three tabs:
- **Diagnose** — upload a chest X-ray, see the Grad-CAM overlay and a probability chart, then generate a cited explanation.
- **Ask the literature** — ask any medical question; get a grounded, cited answer.
- **Setup** — live configuration and status.

**REST API (`make serve`)**

| Endpoint | What it does |
|---|---|
| `GET  /health` | status of the model, vector store and writer |
| `POST /predict` | X-ray image → ranked findings (`?gradcam=true` adds a heat map) |
| `POST /explain` | finding + probability → grounded explanation |
| `POST /qa` | free-text question → grounded answer |

---

## 7. Project layout

```
openmed/
├── app/streamlit_app.py         # web UI
├── configs/                     # training + RAG configuration
├── scripts/                     # model server launchers, PubMed fetcher
├── src/openmed/
│   ├── data/                    # datasets, labels, transforms
│   ├── models/                  # classifier + Grad-CAM
│   ├── train/                   # training loop, metrics, evaluation
│   ├── rag/                     # embeddings, vector store, retrieval,
│   │                            #   prompts, LLM client, pipeline, evaluation
│   ├── api/                     # FastAPI server
│   └── cli.py                   # command-line interface
├── tests/                       # 33 unit tests
├── models/                      # (you put the two .gguf files here)
└── data/                        # corpus, vector store, outputs (mostly git-ignored)
```

---

## 8. Evaluation (reproduce the numbers)

```bash
make build-gold        # build the retrieval benchmark from the corpus
make eval-retrieval    # Recall@k, MRR, nDCG, score calibration
make eval-answers      # answer quality: citations, failure modes, faithfulness
```

Results are written to `runs/rag/`.

---

## 9. Useful commands

| Command | Purpose |
|---|---|
| `make doctor` | check the environment |
| `make train-chestmnist` | train the vision model |
| `make embed-server` / `make embed-server-stop` | start/stop the retrieval server |
| `make llama` / `make llama-stop` | start/stop the writer server |
| `make ingest` | build the search index |
| `make query Q="..."` | ask a question from the terminal |
| `make ui` / `make serve` | web UI / REST API |
| `make test` | run the test suite |

---

## 10. Limitations (please read)

- **Chest X-rays only.** The model predicts exactly 14 chest conditions. Feed it a CT scan, an MRI or a photo and it will still return 14 chest probabilities without warning.
- **Not a diagnosis.** No clinical validation, no doctor in the loop. It is a research prototype.
- **Small writer.** The local 1.2B language model grounds its answer in the retrieved abstracts but can still misstate details — always check the cited sources.
- **Low resolution.** ChestMNIST images are 64×64, far below clinical quality, so absolute accuracy is modest.
- **Limited library.** Explanations are only as good as the ~500 abstracts in `data/corpus/`.

---

## 11. Background & credits

Built as an MCA deep-learning project. It combines transfer learning for medical image
classification with retrieval-augmented generation (RAG) for grounded, cited explanations.

Technology used: PyTorch, timm, MedMNIST, Grad-CAM, Sentence-Transformers/EmbeddingGemma,
llama.cpp, LFM (Liquid AI), FastAPI, Streamlit, pytest.
