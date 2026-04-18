"""
Presentation-friendly evaluation wrapper for the rebuilt V2.2 pipeline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models" / "rebuild"


def run_command(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stderr.strip():
        print(completed.stderr.strip(), file=sys.stderr)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def print_primary_metrics(primary: dict[str, float]) -> None:
    print("Primary Metrics")
    print("-" * 96)
    print(f"{'Anchor query recall@20':<38} {primary['anchor_query_recall_at_20']:.4f}")
    print(f"{'Anchor query recall@50':<38} {primary['anchor_query_recall_at_50']:.4f}")
    print(f"{'Weighted query recall@20':<38} {primary['anchor_query_weighted_recall_at_20']:.4f}")
    print(f"{'Anchor rerank hit@10':<38} {primary['anchor_rerank_hit_at_10']:.4f}")
    print(f"{'Anchor rerank mean rank':<38} {primary['anchor_rerank_conditional_mean_rank']:.4f}")
    print(f"{'Weighted rerank NDCG@10':<38} {primary['anchor_rerank_weighted_ndcg_at_10']:.4f}")
    print(f"{'Benefit MAE':<38} {primary['benefit_mae']:.4f}")
    print(f"{'Blended acceptance MAE':<38} {primary['blended_accept_mae']:.4f}")
    print(f"{'Top-1 predicted acceptance mean':<38} {primary['top1_predicted_accept_mean']:.4f}")
    print(f"{'Public-transfer-supported share':<38} {primary['public_transfer_supported_share']:.4f}")
    print(f"{'Top-1 source max share':<38} {primary['top1_source_max_share']:.4f}")
    print()


def print_goal_breakdown(goal_breakdown: dict[str, dict[str, float]]) -> None:
    print("Per-Goal Summary")
    print("-" * 96)
    print(f"{'Goal':<14} {'Rows':>6} {'WQ@20':>10} {'Hit@10':>10} {'MeanRank':>10}")
    for goal in ["focus", "wind_down", "uplift", "movement"]:
        if goal not in goal_breakdown:
            continue
        payload = goal_breakdown[goal]
        print(
            f"{goal:<14} {int(payload['rows']):>6} "
            f"{float(payload['anchor_query_weighted_recall_at_20']):>10.4f} "
            f"{float(payload['anchor_rerank_hit_at_10']):>10.4f} "
            f"{float(payload['anchor_rerank_conditional_mean_rank']):>10.4f}"
        )
    print()


def save_figures(offline_eval: dict, output_dir: Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Figure export requires matplotlib. Install it with: pip install matplotlib"
        ) from exc

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 19,
            "axes.labelsize": 13,
            "axes.titleweight": "semibold",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in ("v2_key_metrics.png", "v2_legacy_diagnostics.png", "v2_pipeline_overview.png"):
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    created_paths: list[Path] = []

    def _annotate_bars(ax, bars, fmt: str = "{:.3f}", y_pad: float = 0.02) -> None:
        y_top = ax.get_ylim()[1]
        for bar in bars:
            val = float(bar.get_height())
            y = val + y_pad
            va = "bottom"
            color = "#1a1a1a"
            # Keep labels visible even when bars hit the top bound.
            if y > 0.95 * y_top:
                y = val - y_pad * 1.8
                va = "top"
                color = "white"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y,
                fmt.format(val),
                ha="center",
                va=va,
                fontsize=11,
                fontweight="semibold",
                color=color,
            )

    # 1) Top-1 source distribution.
    source_dist = offline_eval.get("top1_source_distribution", {})
    if source_dist:
        ordered = sorted(source_dist.items(), key=lambda item: item[1], reverse=True)
        src_labels = [k for k, _ in ordered]
        src_values = [float(v) for _, v in ordered]
        fig2, ax2 = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
        bars2 = ax2.bar(src_labels, src_values, color="#4C78A8", edgecolor="#2b2b2b", linewidth=0.4)
        ax2.set_title("Top-1 Recommendation Source Mix", pad=12)
        ax2.set_ylabel("Share")
        ax2.set_ylim(0.0, 1.08)
        ax2.grid(axis="y", linestyle="--", alpha=0.28)
        ax2.grid(axis="x", visible=False)
        _annotate_bars(ax2, bars2, fmt="{:.1%}", y_pad=0.03)
        fig2_path = output_dir / "v2_source_mix.png"
        fig2.savefig(fig2_path, dpi=260, facecolor="white")
        plt.close(fig2)
        created_paths.append(fig2_path)

    # 2) Per-goal summary with sample counts.
    goal_breakdown = offline_eval.get("goal_breakdown", {})
    ordered_goals = [goal for goal in ["focus", "wind_down", "uplift", "movement"] if goal in goal_breakdown]
    if ordered_goals:
        fig3, ax3 = plt.subplots(figsize=(11.0, 6.4), constrained_layout=True)
        x_pos = list(range(len(ordered_goals)))
        wq20 = [float(goal_breakdown[goal]["anchor_query_weighted_recall_at_20"]) for goal in ordered_goals]
        hit10 = [float(goal_breakdown[goal]["anchor_rerank_hit_at_10"]) for goal in ordered_goals]
        sample_counts = [int(goal_breakdown[goal]["rows"]) for goal in ordered_goals]
        width = 0.36
        bars31 = ax3.bar(
            [x - width / 2.0 for x in x_pos],
            wq20,
            width=width,
            label="Weighted Query R@20",
            color="#4C78A8",
            edgecolor="#2b2b2b",
            linewidth=0.35,
        )
        bars32 = ax3.bar(
            [x + width / 2.0 for x in x_pos],
            hit10,
            width=width,
            label="Rerank Hit@10",
            color="#F58518",
            edgecolor="#2b2b2b",
            linewidth=0.35,
        )
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f"{goal}\n(n={count})" for goal, count in zip(ordered_goals, sample_counts)])
        ax3.set_ylim(0.0, 1.0)
        ax3.set_ylabel("Score")
        ax3.set_title("Per-Goal Retrieval / Rerank Quality", pad=12)
        ax3.grid(axis="y", linestyle="--", alpha=0.28)
        ax3.grid(axis="x", visible=False)
        ax3.legend(loc="upper right", frameon=True)
        _annotate_bars(ax3, bars31, fmt="{:.3f}", y_pad=0.018)
        _annotate_bars(ax3, bars32, fmt="{:.3f}", y_pad=0.018)
        fig3_path = output_dir / "v2_goal_breakdown.png"
        fig3.savefig(fig3_path, dpi=260, facecolor="white")
        plt.close(fig3)
        created_paths.append(fig3_path)

    return created_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or print the rebuilt V2.2 evaluation summary.")
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--no-rerun", action="store_true", help="Read the latest saved artifacts without rerunning eval.")
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="figures/v2_eval",
        help="Output directory for exported figures.",
    )
    args = parser.parse_args()

    if not args.no_rerun:
        run_command([sys.executable, "src/v2/eval/offline_eval.py", "--split", "test", "--candidate-k", str(args.candidate_k)])
        run_command([sys.executable, "src/v2/eval/check_readiness.py"])

    offline_eval = load_json(MODELS_DIR / "offline_eval_v2.json")
    readiness = load_json(MODELS_DIR / "v2_readiness.json")

    print("=" * 96)
    print("V2.2 OFFLINE EVALUATION")
    print("=" * 96)
    print(f"Readiness: {'READY' if bool(readiness['ready']) else 'NOT READY'}")
    print(f"Rows evaluated: {int(offline_eval['rows'])}")
    print(f"Candidate set size: {int(offline_eval['candidate_k'])}")
    print()

    print_primary_metrics(offline_eval["primary_metrics"])
    print_goal_breakdown(offline_eval["goal_breakdown"])

    print("Top-1 Source Distribution")
    print("-" * 96)
    for source, share in offline_eval["top1_source_distribution"].items():
        print(f"{source:<14} {float(share):.4f}")
    print()

    print("Legacy Diagnostics")
    print("-" * 96)
    legacy = offline_eval["legacy_diagnostics"]
    print(f"{'Exact-song query recall@50':<38} {float(legacy['exact_song_query_recall_at_50']):.4f}")
    print(f"{'Exact-song rerank hit@10':<38} {float(legacy['exact_song_rerank_hit_at_10']):.4f}")
    print(f"{'Exact-song conditional rank':<38} {float(legacy['exact_song_conditional_rank']):.4f}")
    print()

    if readiness["blockers"]:
        print("Blockers")
        print("-" * 96)
        for blocker in readiness["blockers"]:
            print(f"- {blocker}")
        print()

    print("Artifacts")
    print("-" * 96)
    print(f"- {MODELS_DIR / 'offline_eval_v2.json'}")
    print(f"- {MODELS_DIR / 'v2_readiness.json'}")
    figure_dir = (PROJECT_ROOT / args.figures_dir).resolve()
    paths = save_figures(offline_eval, figure_dir)
    print(f"- {figure_dir} ({len(paths)} figures)")
    for path in paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
