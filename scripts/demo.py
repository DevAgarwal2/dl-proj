#!/usr/bin/env python3
"""Live OpenMed demo: X-ray -> predictions -> Grad-CAM -> cited explanation.

Run (with the embedding server and the LLM server already up):

    make embed-server && make llama
    python scripts/demo.py                          # uses a bundled sample
    python scripts/demo.py path/to/chest_xray.png
    python scripts/demo.py --finding Cardiomegaly   # skip the image, explain a finding

Prereqs: a trained checkpoint (else it uses the untrained demo backbone) and the
two local servers. Everything is optional and degrades gracefully.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BAR = "─" * 66


def rule(title: str) -> None:
    print(f"\n{BAR}\n  {title}\n{BAR}")


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenMed live demo")
    parser.add_argument("image", nargs="?", default="samples/images/single_000018.png")
    parser.add_argument("--checkpoint", default="runs/chestmnist/best.pt")
    parser.add_argument("--finding", default="", help="explain this finding instead of predicting")
    parser.add_argument("--config", default="configs/rag.yaml")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    from openmed.inference import Predictor

    print("\n  O P E N M E D   —   chest X-ray → explanation  (all on-device)")

    # --- 1. vision -----------------------------------------------------------
    checkpoint = args.checkpoint if Path(args.checkpoint).exists() else None
    rule("STEP 1 · Look at the X-ray")
    if checkpoint:
        print(f"  checkpoint: {args.checkpoint}")
    else:
        print("  ⚠ no checkpoint found — using an UNTRAINED demo backbone (numbers are meaningless)")
    predictor = Predictor(checkpoint, device=args.device)

    finding, probability = args.finding, 0.86
    if not args.finding:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"  image not found: {image_path}")
            return 1
        print(f"  image     : {image_path}")
        findings = predictor.predict(str(image_path), top_k=5)
        print("\n  ranked findings:")
        for item in findings:
            bar = "█" * int(round(item["probability"] * 30))
            print(f"    {item['label']:<20} {item['probability']*100:5.1f}%  {bar}")
        finding, probability = findings[0]["label"], findings[0]["probability"]

        try:
            cam = predictor.predict_with_cam(str(image_path), top_k=1)[0]
            out = Path("outputs"); out.mkdir(exist_ok=True)
            cam_path = out / "demo_gradcam.png"
            cam["overlay"].save(cam_path)
            print(f"\n  Grad-CAM heat-map saved → {cam_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"  (grad-cam skipped: {exc})")

    # --- 2..4. retrieval + explanation --------------------------------------
    rule("STEP 2-4 · Find evidence and explain")
    print(f"  finding: {finding}  ({probability*100:.1f}%)")
    try:
        from openmed.config import load_config
        from openmed.rag.pipeline import OpenMedRAG

        rag = OpenMedRAG.from_config(load_config(args.config))
        explanation = rag.explain_finding(finding, probability)
    except Exception as exc:  # noqa: BLE001
        print(f"\n  ⚠ could not generate an explanation: {exc}")
        print("  start the servers first:  make embed-server   then   make llama")
        return 0

    print("\n  explanation:\n")
    for line in explanation.text.split(". "):
        print(f"    {line.strip().rstrip('.')}.")
    if explanation.citations:
        print("\n  references:")
        for c in explanation.citations:
            print(f"    [{c.index}] {c.title}  (score {c.score:.3f})")
    print(f"\n  sources: {len(explanation.evidence)} retrieved passages · model: {explanation.model}")
    print(f"\n{BAR}\n  Research tool — not medical advice.\n{BAR}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
