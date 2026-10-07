"""Train the twin's risk models on the synthetic cohort and print evaluation metrics.

Usage:  python train.py [--n 20000]
"""
import argparse
import json

from twin.models import MODEL_PATH, TARGETS, train_models

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000, help="synthetic cohort size")
    args = ap.parse_args()
    bundle = train_models(args.n)
    for target, m in bundle["metrics"].items():
        print(f"{TARGETS[target]:32s} AUC={m['auc']:.3f}  Brier={m['brier']:.4f}  prevalence={m['prevalence']:.1%}")
    print(f"saved -> {MODEL_PATH}")
    (MODEL_PATH.parent / "metrics.json").write_text(json.dumps(bundle["metrics"], indent=2))
