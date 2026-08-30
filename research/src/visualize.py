"""Create presentation-ready figures from evaluate.py outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


VARIANTS = ["original", "shuffled", "masked", "first_25", "first_50", "first_75"]
LABELS = {"original": "Original", "shuffled": "Word shuffle", "masked": "Keyword mask", "first_25": "First 25%", "first_50": "First 50%", "first_75": "First 75%"}
COLORS = {"tfidf_lr": "#E76F51", "sbert_lr": "#277DA1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--drops", type=Path, required=True)
    parser.add_argument("--per-genre", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("research/figures"))
    return parser.parse_args()


def style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 300, "axes.titleweight": "bold", "font.family": "DejaVu Sans"})


def plot_overall(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
    x = np.arange(len(VARIANTS)); width = 0.36
    models = sorted(summary["model"].unique())
    for metric, axis in zip(["macro_f1", "micro_f1"], axes):
        for index, model in enumerate(models):
            values = summary.loc[summary.model.eq(model)].set_index("variant").reindex(VARIANTS)[metric]
            axis.bar(x + (index - (len(models)-1)/2)*width, values, width, label=model.replace("_", " + ").upper(), color=COLORS.get(model, "#555555"))
        axis.set_title(metric.replace("_", " ").title())
        axis.set_ylim(0, 1); axis.set_xticks(x, [LABELS[item] for item in VARIANTS], rotation=28, ha="right")
        axis.set_ylabel("F1 score")
    axes[0].legend(frameon=False)
    fig.suptitle("Model performance across six controlled inputs", y=1.02, fontsize=15, fontweight="bold")
    fig.tight_layout(); fig.savefig(output / "fig1_overall_comparison.png", bbox_inches="tight"); plt.close(fig)


def plot_drops(drops: pd.DataFrame, output: Path) -> None:
    order = ["shuffled", "masked", "first_75", "first_50", "first_25"]
    fig, ax = plt.subplots(figsize=(10, 5.5)); x = np.arange(len(order)); width = 0.36
    models = sorted(drops["model"].unique())
    for index, model in enumerate(models):
        values = drops.loc[drops.model.eq(model)].set_index("variant").reindex(order)["macro_f1_drop"]
        bars = ax.bar(x + (index - (len(models)-1)/2)*width, values, width, label=model.replace("_", " + ").upper(), color=COLORS.get(model, "#555555"))
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=9)
    ax.set_xticks(x, [LABELS[item] for item in order]); ax.set_ylabel("Macro-F1 decline from original")
    ax.set_title("Which intervention hurts each model most?"); ax.legend(frameon=False); ax.axhline(0, color="#333333", linewidth=.8)
    fig.tight_layout(); fig.savefig(output / "fig2_perturbation_drop.png", bbox_inches="tight"); plt.close(fig)


def plot_length(summary: pd.DataFrame, output: Path) -> None:
    order = ["first_25", "first_50", "first_75", "original"]
    fig, ax = plt.subplots(figsize=(8, 5.3))
    for model, frame in summary.groupby("model"):
        values = frame.set_index("variant").reindex(order)["macro_f1"]
        ax.plot([25, 50, 75, 100], values, marker="o", markersize=8, linewidth=2.5, label=model.replace("_", " + ").upper(), color=COLORS.get(model, "#555555"))
    ax.set(xlim=(20, 105), ylim=(0, 1), xlabel="Plot text retained (%)", ylabel="Macro-F1", title="How much plot context is needed?")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(output / "fig3_length_curve.png", bbox_inches="tight"); plt.close(fig)


def plot_heatmap(per_genre: pd.DataFrame, output: Path) -> None:
    original = per_genre.loc[per_genre.variant.eq("original")]
    matrix = original.pivot(index="genre_name", columns="model", values="f1").sort_values("sbert_lr", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 8))
    sns.heatmap(matrix, cmap="YlGnBu", vmin=0, vmax=1, annot=True, fmt=".2f", linewidths=.4, linecolor="white", cbar_kws={"label": "F1"}, ax=ax)
    ax.set(xlabel="Model", ylabel="Genre", title="Per-genre F1 on original plots")
    fig.tight_layout(); fig.savefig(output / "fig4_per_genre_heatmap.png", bbox_inches="tight"); plt.close(fig)


def main() -> None:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True); style()
    summary, drops, per_genre = pd.read_csv(args.summary), pd.read_csv(args.drops), pd.read_csv(args.per_genre)
    plot_overall(summary, args.output_dir); plot_drops(drops, args.output_dir); plot_length(summary, args.output_dir); plot_heatmap(per_genre, args.output_dir)
    print(f"Saved four figures to {args.output_dir}")


if __name__ == "__main__":
    main()
