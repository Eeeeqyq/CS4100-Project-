"""
Readiness checker for the rebuilt V2.2 pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODELS_DIR = PROJECT_ROOT / "models" / "rebuild"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def gate(
    name: str,
    passed: bool,
    actual: float | str,
    target: float | str,
    severity: str,
    rationale: str,
) -> dict[str, object]:
    return {
        "name": name,
        "passed": bool(passed),
        "actual": actual,
        "target": target,
        "severity": severity,
        "rationale": rationale,
    }


def phase(name: str, passed: bool, details: list[str]) -> dict[str, object]:
    return {"name": name, "passed": bool(passed), "details": details}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether V2.2 is ready on its own terms.")
    parser.add_argument("--fail-on-not-ready", action="store_true")
    args = parser.parse_args()

    query_metrics = load_json(MODELS_DIR / "query_tower_metrics.json")
    reranker_metrics = load_json(MODELS_DIR / "reranker_metrics.json")
    offline_eval = load_json(MODELS_DIR / "offline_eval_v2.json")

    primary = offline_eval["primary_metrics"]
    hard_gates = [
        gate(
            name="anchor_query_recall_at_20",
            passed=float(primary["anchor_query_recall_at_20"]) >= 0.20,
            actual=round(float(primary["anchor_query_recall_at_20"]), 4),
            target=">= 0.20",
            severity="hard",
            rationale="Anchor retrieval must recover useful intervention candidates early in the rank list.",
        ),
        gate(
            name="anchor_query_recall_at_50",
            passed=float(primary["anchor_query_recall_at_50"]) >= 0.35,
            actual=round(float(primary["anchor_query_recall_at_50"]), 4),
            target=">= 0.35",
            severity="hard",
            rationale="Anchor retrieval should not rely on extreme depth to find good anchors.",
        ),
        gate(
            name="anchor_query_weighted_recall_at_20",
            passed=float(primary["anchor_query_weighted_recall_at_20"]) >= 0.15,
            actual=round(float(primary["anchor_query_weighted_recall_at_20"]), 4),
            target=">= 0.15",
            severity="hard",
            rationale="Tier-aware retrieval should surface the strongest anchors, not only any acceptable positive.",
        ),
        gate(
            name="anchor_query_weighted_recall_at_50",
            passed=float(primary["anchor_query_weighted_recall_at_50"]) >= 0.25,
            actual=round(float(primary["anchor_query_weighted_recall_at_50"]), 4),
            target=">= 0.25",
            severity="hard",
            rationale="Weighted recall checks that factual and same-song anchors remain reachable under the stricter supervision.",
        ),
        gate(
            name="anchor_rerank_hit_at_10",
            passed=float(primary["anchor_rerank_hit_at_10"]) >= 0.25,
            actual=round(float(primary["anchor_rerank_hit_at_10"]), 4),
            target=">= 0.25",
            severity="hard",
            rationale="Once anchors are retrieved, reranking should usually surface a positive anchor in the top 10.",
        ),
        gate(
            name="anchor_rerank_conditional_mean_rank",
            passed=float(primary["anchor_rerank_conditional_mean_rank"]) <= 6.0,
            actual=round(float(primary["anchor_rerank_conditional_mean_rank"]), 4),
            target="<= 6.0",
            severity="hard",
            rationale="Positive anchors should land near the top after reranking, not remain buried.",
        ),
        gate(
            name="anchor_rerank_weighted_ndcg_at_10",
            passed=float(primary["anchor_rerank_weighted_ndcg_at_10"]) >= 0.20,
            actual=round(float(primary["anchor_rerank_weighted_ndcg_at_10"]), 4),
            target=">= 0.20",
            severity="hard",
            rationale="Weighted NDCG checks that reranking materially improves tier-aware ordering over the query stage under the harder contract.",
        ),
        gate(
            name="benefit_mae",
            passed=float(primary["benefit_mae"]) <= 0.20,
            actual=round(float(primary["benefit_mae"]), 4),
            target="<= 0.20",
            severity="hard",
            rationale="Benefit estimates must remain numerically credible on held-out anchor candidates.",
        ),
        gate(
            name="blended_accept_mae",
            passed=float(primary["blended_accept_mae"]) <= 0.35,
            actual=round(float(primary["blended_accept_mae"]), 4),
            target="<= 0.35",
            severity="hard",
            rationale="Acceptance calibration must be strong enough for the dual-objective story to hold.",
        ),
        gate(
            name="top1_predicted_accept_mean",
            passed=float(primary["top1_predicted_accept_mean"]) >= -0.10,
            actual=round(float(primary["top1_predicted_accept_mean"]), 4),
            target=">= -0.10",
            severity="hard",
            rationale="Top-ranked recommendations should not look systematically unacceptable.",
        ),
        gate(
            name="top1_source_max_share",
            passed=float(primary["top1_source_max_share"]) <= 0.85,
            actual=round(float(primary["top1_source_max_share"]), 4),
            target="<= 0.85",
            severity="hard",
            rationale="The system should not collapse to one catalog after public transfer is enabled.",
        ),
        gate(
            name="public_transfer_supported_share",
            passed=float(primary["public_transfer_supported_share"]) >= 0.05,
            actual=round(float(primary["public_transfer_supported_share"]), 4),
            target=">= 0.05",
            severity="hard",
            rationale="Public-transfer wins should happen sometimes, but only when support is strong enough.",
        ),
    ]

    for goal_name, payload in offline_eval["goal_breakdown"].items():
        rows = int(payload["rows"])
        if rows < 10:
            continue
        hard_gates.append(
            gate(
                name=f"goal_anchor_query_weighted_recall_at_20::{goal_name}",
                passed=float(payload["anchor_query_weighted_recall_at_20"]) >= 0.08,
                actual=round(float(payload["anchor_query_weighted_recall_at_20"]), 4),
                target=">= 0.08",
                severity="hard",
                rationale="Each supported goal needs nontrivial tier-aware anchor retrieval, not only the aggregate average.",
            )
        )

    legacy = offline_eval["legacy_diagnostics"]
    soft_gates = [
        gate(
            name="legacy_exact_song_query_recall_at_50",
            passed=float(legacy["exact_song_query_recall_at_50"]) >= 0.05,
            actual=round(float(legacy["exact_song_query_recall_at_50"]), 4),
            target=">= 0.05",
            severity="soft",
            rationale="Exact-song recovery should still improve, even though it is no longer the primary claim.",
        ),
        gate(
            name="legacy_exact_song_rerank_hit_at_10",
            passed=float(legacy["exact_song_rerank_hit_at_10"]) >= 0.05,
            actual=round(float(legacy["exact_song_rerank_hit_at_10"]), 4),
            target=">= 0.05",
            severity="soft",
            rationale="Legacy song-level imitation remains useful as a sanity check.",
        ),
    ]

    hard_pass = all(item["passed"] for item in hard_gates)
    failed_hard = [item["name"] for item in hard_gates if not item["passed"]]

    phases = [
        phase(
            name="anchor_retrieval",
            passed=all(
                item["passed"]
                for item in hard_gates
                if item["name"].startswith("anchor_query") or item["name"].startswith("goal_anchor_query")
            ),
            details=[
                f"test_recall_at_20={primary['anchor_query_recall_at_20']:.4f}",
                f"test_weighted_recall_at_20={primary['anchor_query_weighted_recall_at_20']:.4f}",
                f"test_recall_at_50={primary['anchor_query_recall_at_50']:.4f}",
                f"query_val_weighted_recall_at_20={query_metrics['val_metrics']['weighted_recall_at_20']:.4f}",
                f"query_val_weighted_ndcg_at_10={query_metrics['val_metrics']['weighted_ndcg_at_10']:.4f}",
            ],
        ),
        phase(
            name="anchor_reranking",
            passed=all(
                item["passed"]
                for item in hard_gates
                if item["name"] in {"anchor_rerank_hit_at_10", "anchor_rerank_conditional_mean_rank", "anchor_rerank_weighted_ndcg_at_10", "benefit_mae", "blended_accept_mae", "top1_predicted_accept_mean"}
            ),
            details=[
                f"test_hit_at_10={primary['anchor_rerank_hit_at_10']:.4f}",
                f"test_conditional_rank={primary['anchor_rerank_conditional_mean_rank']:.4f}",
                f"test_weighted_ndcg_at_10={primary['anchor_rerank_weighted_ndcg_at_10']:.4f}",
                f"test_benefit_mae={primary['benefit_mae']:.4f}",
                f"test_accept_mae={primary['blended_accept_mae']:.4f}",
                f"reranker_test_accept_mae={reranker_metrics['test_metrics']['blended_accept_mae']:.4f}",
            ],
        ),
        phase(
            name="public_transfer",
            passed=all(
                item["passed"]
                for item in hard_gates
                if item["name"] in {"top1_source_max_share", "public_transfer_supported_share"}
            ),
            details=[
                f"public_supported_share={primary['public_transfer_supported_share']:.4f}",
                f"top1_source_max_share={primary['top1_source_max_share']:.4f}",
            ],
        ),
        phase(
            name="legacy_sanity",
            passed=all(item["passed"] for item in soft_gates),
            details=[
                f"legacy_query_recall_at_50={legacy['exact_song_query_recall_at_50']:.4f}",
                f"legacy_hit_at_10={legacy['exact_song_rerank_hit_at_10']:.4f}",
            ],
        ),
        phase(
            name="replacement_ready",
            passed=hard_pass,
            details=["all primary gates must pass before V2.2 can be treated as the main rebuilt path"],
        ),
    ]

    blockers = []
    if any(name.startswith("anchor_query_") for name in failed_hard):
        blockers.append("anchor retrieval is still too weak overall")
    if any(name.startswith("goal_anchor_query_weighted_recall_at_20::") for name in failed_hard):
        blockers.append("at least one supported goal still has weak anchor retrieval")
    if "anchor_rerank_hit_at_10" in failed_hard or "anchor_rerank_conditional_mean_rank" in failed_hard or "anchor_rerank_weighted_ndcg_at_10" in failed_hard:
        blockers.append("anchor reranking is not reliably surfacing positive anchors near the top")
    if "benefit_mae" in failed_hard:
        blockers.append("benefit regression is still too noisy")
    if "blended_accept_mae" in failed_hard or "top1_predicted_accept_mean" in failed_hard:
        blockers.append("acceptance calibration is still too weak")
    if "top1_source_max_share" in failed_hard or "public_transfer_supported_share" in failed_hard:
        blockers.append("public transfer is either too weak or still collapsing to one source")

    summary = {
        "date_checked": str(date.today()),
        "ready": hard_pass,
        "phases": phases,
        "hard_gates": hard_gates,
        "soft_gates": soft_gates,
        "blockers": blockers,
        "notes": [
            "primary evaluation is now stricter anchor retrieval + anchor reranking + support-aware public transfer",
            "weighted anchor metrics are based on positive tiers: factual > same-song > nearby context neighbors",
            "exact-song metrics remain for diagnostics only",
        ],
    }

    (MODELS_DIR / "v2_readiness.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if args.fail_on_not_ready and not hard_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
