"""
Tune the narrow V2.2 public-transfer threshold on the validation split.

This intentionally does not tune broad linear score blends. The only knob here
is how much anchor-conditioned transfer support is required before a public
song is allowed to outrank a supervised SiTunes anchor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v2.inference.recommend import AmbientRecommenderV2


MODELS_DIR = PROJECT_ROOT / "models" / "rebuild"


def objective(metrics: dict[str, float]) -> float | None:
    if metrics["top1_source_max_share"] > 0.85:
        return None
    if metrics["top1_predicted_accept_mean"] < -0.10:
        return None
    return (
        1.5 * metrics["public_transfer_supported_share"]
        - 0.5 * metrics["top1_source_max_share"]
        + 0.25 * metrics["top1_predicted_accept_mean"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune the V2.2 public transfer threshold on validation.")
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    recommender = AmbientRecommenderV2(device=args.device)
    row_indices = recommender.decision_df.index[recommender.decision_df["split"] == "val"].to_numpy(dtype=np.int64)
    thresholds = [0.62, 0.64, 0.66, 0.67, 0.69, 0.72]

    candidates = []
    best = None
    for threshold in thresholds:
        recommender.transfer_support_threshold = float(threshold)
        result = recommender.score_rows(
            row_indices=row_indices,
            top_k=max(50, args.candidate_k),
            candidate_k=args.candidate_k,
        )
        top1_sources, counts = np.unique(result["top1_source"], return_counts=True)
        top1_dist = {str(k): float(v) / float(len(result["top1_source"])) for k, v in zip(top1_sources.tolist(), counts.tolist())}
        metrics = {
            "threshold": float(threshold),
            "public_transfer_supported_share": float(np.mean(result["top1_transfer_supported"])),
            "top1_source_max_share": float(np.max(counts) / float(len(result["top1_source"]))),
            "top1_predicted_accept_mean": float(np.mean(result["top1_accept"])),
            "top1_source_distribution": top1_dist,
        }
        score = objective(metrics)
        metrics["selection_objective"] = None if score is None else float(score)
        candidates.append(metrics)
        if score is not None and (best is None or float(score) > float(best["selection_objective"])):
            best = metrics

    payload = {"best": best, "candidates": candidates}
    (MODELS_DIR / "transfer_threshold_tuning.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
