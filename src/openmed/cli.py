"""OpenMed command line interface.

Examples:
    python -m openmed.cli doctor
    python -m openmed.cli prepare-data --config configs/chestxray14.yaml
    python -m openmed.cli train --config configs/smoke.yaml
    python -m openmed.cli ingest --config configs/rag.yaml
    python -m openmed.cli query --config configs/rag.yaml -q "What is cardiomegaly?"
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

from openmed.config import load_config
from openmed.utils import get_logger

LOGGER = get_logger("openmed.cli")

DEFAULT_GGUF = "models/LFM2.5-1.2B-Instruct-Q6_K.gguf"
DEFAULT_EMBED_GGUF = "models/embeddinggemma-2-Q8_0.gguf"
DEFAULT_GOLD_SET = "data/eval/retrieval.jsonl"


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def cmd_doctor(_args) -> int:
    print("OpenMed environment check")
    print("-" * 46)
    print(f"python                : {sys.version.split()[0]}")
    for module in ["torch", "torchvision", "timm", "sentence_transformers", "transformers", "cffi", "fastapi"]:
        print(f"{module:<22}: {'yes' if _has(module) else 'no'}")

    try:
        import torch

        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        print(f"{'torch version':<22}: {torch.__version__}")
        print(f"{'MPS available':<22}: {mps}")
    except ImportError:
        print(f"{'MPS available':<22}: unknown (torch missing)")

    llama = shutil.which("llama-server")
    print(f"{'llama-server':<22}: {llama or 'not found (brew install llama.cpp)'}")

    gguf = os.environ.get("OPENMED_GGUF", DEFAULT_GGUF)
    print(f"{'GGUF model':<22}: {gguf} {'(found)' if Path(gguf).exists() else '(missing)'}")

    embed_gguf = os.environ.get("OPENMED_EMBED_GGUF", DEFAULT_EMBED_GGUF)
    print(
        f"{'embed GGUF':<22}: {embed_gguf} "
        f"{'(found)' if Path(embed_gguf).exists() else '(missing - set OPENMED_EMBED_GGUF)'}"
    )

    try:
        import urllib.request

        with urllib.request.urlopen("http://127.0.0.1:8081/health", timeout=3) as response:
            embed_up = response.status == 200
    except Exception:  # noqa: BLE001
        embed_up = False
    print(f"{'embed server :8081':<22}: {'up' if embed_up else 'down (make embed-server)'}")

    for cfg in ["configs/chestxray14.yaml", "configs/rag.yaml"]:
        print(f"{'config':<22}: {cfg} {'(found)' if Path(cfg).exists() else '(missing)'}")
    return 0


def cmd_prepare_data(args) -> int:
    from openmed.data.prepare import prepare_from_config

    prepare_from_config(load_config(args.config))
    return 0


def cmd_train(args) -> int:
    from openmed.train.train import run_training

    run_training(load_config(args.config))
    return 0


def cmd_evaluate(args) -> int:
    from openmed.train.evaluate import run_evaluation

    run_evaluation(load_config(args.config), args.checkpoint, split=args.split)
    return 0


def cmd_gradcam(args) -> int:
    from openmed.train.evaluate import export_gradcam_grid

    export_gradcam_grid(load_config(args.config), args.checkpoint, split=args.split, n=args.n)
    return 0


def cmd_predict(args) -> int:
    from openmed.inference import Predictor

    predictor = Predictor(args.checkpoint, device=args.device)
    findings = predictor.predict(args.image, top_k=args.top_k, threshold=args.threshold)
    print(f"Predictions for {args.image}:")
    for item in findings:
        print(f"  {item['label']:<20} {item['probability'] * 100:5.1f}%")

    if args.gradcam:
        results = predictor.predict_with_cam(args.image, top_k=1)
        output = Path(args.gradcam)
        results[0]["overlay"].save(output)
        print(f"Grad-CAM saved -> {output}")
    return 0


def cmd_ingest(args) -> int:
    from openmed.rag.ingest import ingest_from_config

    ingest_from_config(load_config(args.config))
    return 0


def cmd_query(args) -> int:
    from openmed.rag.pipeline import OpenMedRAG

    rag = OpenMedRAG.from_config(load_config(args.config))
    if args.finding:
        explanation = rag.explain_finding(args.finding, args.probability, args.question)
    else:
        explanation = rag.answer(args.question)
    print("\n" + explanation.text + "\n")
    if explanation.citations:
        print("References:")
        for citation in explanation.citations:
            print(f"  [{citation.index}] {citation.title} (score {citation.score:.3f})")
    return 0


def cmd_llama(args) -> int:
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_llama_server.sh"
    if not script.exists():
        script = Path("scripts/run_llama_server.sh")
    action = "stop" if args.stop else "start"
    return subprocess.call(["bash", str(script), action])


def cmd_embed_server(args) -> int:
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_embed_server.sh"
    if not script.exists():
        script = Path("scripts/run_embed_server.sh")
    action = "stop" if args.stop else "start"
    return subprocess.call(["bash", str(script), action])


def cmd_build_gold(args) -> int:
    from openmed.rag.evaluate import build_gold_set

    cfg = load_config(args.config)
    build_gold_set(
        cfg.ingest.corpus_dir,
        args.out,
        modes=tuple(args.modes),
        max_per_finding=args.max_per_finding,
    )
    return 0


def cmd_eval_retrieval(args) -> int:
    from openmed.rag.evaluate import evaluate_and_report

    cfg = load_config(args.config)
    lexical = None if args.lexical is None else bool(args.lexical)
    evaluate_and_report(
        cfg,
        args.gold,
        out_path=args.out,
        ks=tuple(args.ks),
        lexical=lexical,
        limit=args.limit,
    )
    return 0


def cmd_eval_answers(args) -> int:
    from openmed.rag.faithfulness import evaluate_answers

    cfg = load_config(args.config)
    questions = None
    if args.questions:
        questions = [
            line.strip()
            for line in Path(args.questions).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    metrics = evaluate_answers(cfg, questions=questions, out_path=args.out, limit=args.limit)
    print(f"questions        : {metrics['n_questions']}")
    print(f"judged           : {metrics['n_judged']}")
    print(f"mean faithfulness: {metrics['mean_faithfulness']}")
    print(f"mean relevance   : {metrics['mean_answer_relevance']}")
    print(f"usable rate      : {metrics['usable_rate']:.3f}")
    print(f"title-echo rate  : {metrics['title_echo_rate']:.3f}")
    print(f"artifact rate    : {metrics['artifact_rate']:.3f}")
    print(f"citation-ok rate : {metrics['citation_ok_rate']:.3f}")
    print(f"in-range rate    : {metrics['citation_in_range_rate']:.3f}")
    print(f"has-citation rate: {metrics['has_citation_rate']:.3f}")
    print(f"mean answer words: {metrics['mean_answer_words']:.1f}")
    print(f"refusal rate     : {metrics['refusal_rate']:.3f}")
    if args.out:
        print(f"saved -> {args.out}")
    return 0


def cmd_serve(args) -> int:
    os.environ.setdefault("OPENMED_RAG_CONFIG", args.rag_config)
    os.environ.setdefault("OPENMED_CHECKPOINT", args.checkpoint)
    return subprocess.call(
        [sys.executable, "-m", "uvicorn", "openmed.api.server:app", "--host", args.host, "--port", str(args.port)]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openmed", description="Medical vision model + RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check the environment").set_defaults(func=cmd_doctor)

    p = sub.add_parser("prepare-data", help="build train/val/test splits")
    p.add_argument("--config", default="configs/chestxray14.yaml")
    p.set_defaults(func=cmd_prepare_data)

    p = sub.add_parser("train", help="fine-tune the vision model")
    p.add_argument("--config", default="configs/chestxray14.yaml")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="evaluate a checkpoint")
    p.add_argument("--config", default="configs/chestxray14.yaml")
    p.add_argument("--checkpoint", default="runs/chestxray14/best.pt")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("gradcam", help="export a Grad-CAM grid")
    p.add_argument("--config", default="configs/chestxray14.yaml")
    p.add_argument("--checkpoint", default="runs/chestxray14/best.pt")
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=8)
    p.set_defaults(func=cmd_gradcam)

    p = sub.add_parser("predict", help="predict a single image")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--gradcam", default="", help="path to save a Grad-CAM overlay")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("ingest", help="embed the corpus")
    p.add_argument("--config", default="configs/rag.yaml")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("query", help="ask the RAG pipeline")
    p.add_argument("--config", default="configs/rag.yaml")
    p.add_argument("-q", "--question", required=True)
    p.add_argument("--finding", default="")
    p.add_argument("--probability", type=float, default=0.86)
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("llama", help="start/stop the llama.cpp server")
    p.add_argument("--stop", action="store_true")
    p.set_defaults(func=cmd_llama)

    p = sub.add_parser("embed-server", help="start/stop the EmbeddingGemma llama.cpp server")
    p.add_argument("--stop", action="store_true")
    p.set_defaults(func=cmd_embed_server)

    p = sub.add_parser("build-gold", help="build the retrieval gold set from the corpus")
    p.add_argument("--config", default="configs/rag.yaml")
    p.add_argument("--out", default=DEFAULT_GOLD_SET)
    p.add_argument("--modes", nargs="+", default=["title", "finding", "negative"], choices=["title", "finding", "negative"])
    p.add_argument("--max-per-finding", type=int, default=None)
    p.set_defaults(func=cmd_build_gold)

    p = sub.add_parser("eval-retrieval", help="score retrieval against the gold set")
    p.add_argument("--config", default="configs/rag.yaml")
    p.add_argument("--gold", default=DEFAULT_GOLD_SET)
    p.add_argument("--out", default="runs/rag/retrieval_metrics.json")
    p.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--lexical", action=argparse.BooleanOptionalAction, default=None)
    p.set_defaults(func=cmd_eval_retrieval)

    p = sub.add_parser("eval-answers", help="judge answer faithfulness with the local LLM")
    p.add_argument("--config", default="configs/rag.yaml")
    p.add_argument("--questions", default="", help="optional newline-delimited questions file")
    p.add_argument("--out", default="runs/rag/faithfulness.json")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_eval_answers)

    p = sub.add_parser("serve", help="run the FastAPI server")
    p.add_argument("--rag-config", default="configs/rag.yaml")
    p.add_argument("--checkpoint", default="runs/chestxray14/best.pt")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
