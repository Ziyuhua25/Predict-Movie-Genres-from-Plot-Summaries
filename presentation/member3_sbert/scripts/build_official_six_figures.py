#!/usr/bin/env python3
"""Build the two 16:9 robustness slides from official six-input metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image, ImageOps


BG = "#F7F3EB"
INK = "#203746"
MUTED = "#667680"
LINE = "#D9D4CA"
ORANGE = "#EB5A3B"
TEAL = "#3F8C86"
GOLD = "#D39A31"
BLUE = "#3D7DBD"
WHITE = "#FFFFFF"
PALE_ORANGE = "#FCE8E1"
PALE_TEAL = "#E4F0ED"
PALE_GOLD = "#F7EED8"
PALE_BLUE = "#E5EEF8"

plt.rcParams.update(
    {
        "font.family": "Arial",
        "axes.facecolor": BG,
        "figure.facecolor": BG,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": INK,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def rounded(ax, x, y, w, h, facecolor, edgecolor="none", radius=0.02, lw=1.0):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def header(ax, title, subtitle):
    ax.text(0.055, 0.935, "ROBUSTNESS ANALYSIS", fontsize=10, fontweight="bold", color=ORANGE, va="center")
    ax.add_patch(Rectangle((0.055, 0.907), 0.89, 0.003, transform=ax.transAxes, color=ORANGE, lw=0))
    ax.text(0.055, 0.842, title, fontsize=27, fontweight="bold", color=INK, va="center")
    ax.text(0.055, 0.797, subtitle, fontsize=12.5, color=MUTED, va="center")


def footer(ax, page):
    ax.text(0.055, 0.035, "SBERT + ONE-VS-REST LOGISTIC REGRESSION", fontsize=7.5, color=MUTED)
    ax.text(0.945, 0.035, page, fontsize=8, color=MUTED, ha="right")


def slide_results(metrics: pd.DataFrame, path: Path):
    original = metrics.loc[metrics.variant.eq("original")].iloc[0]
    order = ["shuffled", "masked", "first_75", "first_50", "first_25"]
    labels = ["Shuffle 100%", "Mask 50% cues", "Keep first 75%", "Keep first 50%", "Keep first 25%"]
    colors = [TEAL, ORANGE, GOLD, GOLD, GOLD]
    selected = metrics.set_index("variant").loc[order]

    fig = plt.figure(figsize=(16, 9), dpi=100)
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_axis_off()
    header(
        canvas,
        "SBERT robustness on the official six inputs",
        "Macro-F1 on the same 1,198 movies · fixed encoder, classifier, preprocessing, and threshold (0.25)",
    )

    rounded(canvas, 0.055, 0.19, 0.20, 0.54, WHITE, LINE, 0.018, 1.0)
    canvas.text(0.078, 0.682, "ORIGINAL TEST SET", fontsize=9, fontweight="bold", color=ORANGE)
    canvas.text(0.078, 0.585, f"{original.macro_f1:.3f}", fontsize=39, fontweight="bold", color=BLUE)
    canvas.text(0.078, 0.548, "Macro-F1", fontsize=11, color=MUTED)
    canvas.text(0.078, 0.455, f"{original.micro_f1:.3f}", fontsize=26, fontweight="bold", color=INK)
    canvas.text(0.078, 0.420, "Micro-F1", fontsize=11, color=MUTED)
    canvas.add_patch(Rectangle((0.078, 0.368), 0.13, 0.003, transform=canvas.transAxes, color=LINE, lw=0))
    canvas.text(0.078, 0.315, "Frozen MiniLM encoder", fontsize=10.5, color=INK)
    canvas.text(0.078, 0.275, "18 one-vs-rest LR heads", fontsize=10.5, color=INK)
    canvas.text(0.078, 0.235, "No retraining per condition", fontsize=10.5, color=INK)

    chart = fig.add_axes([0.315, 0.24, 0.625, 0.45])
    y = list(range(len(labels)))[::-1]
    values = selected.macro_f1.to_numpy()
    changes = selected.macro_f1_change_vs_original.to_numpy()
    chart.barh(y, values, color=colors, height=0.58, alpha=0.95)
    chart.axvline(original.macro_f1, color=INK, linestyle=(0, (4, 3)), linewidth=1.4, alpha=0.75)
    chart.text(original.macro_f1 + 0.004, 4.42, "Original 0.548", fontsize=9, color=INK, va="bottom")
    chart.set_xlim(0, 0.62)
    chart.set_ylim(-0.65, 4.7)
    chart.set_yticks(y, labels, fontsize=10.5)
    chart.set_xticks([0, 0.2, 0.4, 0.6])
    chart.set_xlabel("Macro-F1", fontsize=10.5, labelpad=8)
    chart.grid(axis="x", color=LINE, linewidth=0.8, alpha=0.8)
    chart.set_axisbelow(True)
    for spine in chart.spines.values():
        spine.set_visible(False)
    chart.tick_params(axis="y", length=0, pad=10)
    chart.tick_params(axis="x", length=0)
    for yy, value, change in zip(y, values, changes):
        chart.text(value + 0.008, yy, f"{value:.3f}", va="center", fontsize=10.5, fontweight="bold", color=INK)
        chart.text(0.615, yy, f"Δ {change:+.3f}", va="center", ha="right", fontsize=9.5, color=MUTED)

    canvas.text(0.315, 0.725, "WORD ORDER", fontsize=8.5, fontweight="bold", color=TEAL)
    canvas.text(0.49, 0.725, "KEYWORD CUES", fontsize=8.5, fontweight="bold", color=ORANGE)
    canvas.text(0.675, 0.725, "PLOT COVERAGE", fontsize=8.5, fontweight="bold", color=GOLD)
    rounded(canvas, 0.285, 0.105, 0.655, 0.065, PALE_BLUE, "none", 0.012)
    canvas.text(
        0.612,
        0.137,
        "Local word order has little effect; removing cues or plot coverage causes much larger losses.",
        fontsize=11.5,
        fontweight="bold",
        color=INK,
        ha="center",
        va="center",
    )
    footer(canvas, "R1")
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)


def slide_interpretation(metrics: pd.DataFrame, path: Path):
    lookup = metrics.set_index("variant")
    shuffled = lookup.loc["shuffled", "macro_f1_change_vs_original"]
    masked = lookup.loc["masked", "macro_f1_change_vs_original"]
    first_25 = lookup.loc["first_25", "macro_f1_change_vs_original"]

    fig = plt.figure(figsize=(16, 9), dpi=100)
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_axis_off()
    header(canvas, "What do the official interventions reveal about SBERT?", "Controlled inputs isolate which textual signals support genre classification")

    cards = [
        (0.055, TEAL, "01  WORD ORDER", f"{shuffled:+.3f}", "after full local shuffling", "Local reordering barely changes the task signal."),
        (0.365, ORANGE, "02  GENRE CUES", f"{masked:+.3f}", "after official 50% cue masking", "Removing label-related evidence causes a clear decline."),
        (0.675, GOLD, "03  PLOT COVERAGE", f"{first_25:+.3f}", "when only the first 25% remains", "Severe truncation removes the most useful evidence."),
    ]
    for x, color, eyebrow, metric, metric_note, body in cards:
        rounded(canvas, x, 0.485, 0.27, 0.245, WHITE, LINE, 0.018, 1.0)
        canvas.add_patch(Rectangle((x, 0.712), 0.27, 0.018, transform=canvas.transAxes, color=color, lw=0))
        canvas.text(x + 0.025, 0.665, eyebrow, fontsize=9, fontweight="bold", color=color)
        canvas.text(x + 0.025, 0.582, metric, fontsize=29, fontweight="bold", color=INK)
        canvas.text(x + 0.025, 0.546, metric_note, fontsize=9.5, color=MUTED)
        canvas.text(x + 0.025, 0.505, body, fontsize=10.2, color=INK, wrap=True)

    rounded(canvas, 0.055, 0.185, 0.89, 0.235, INK, "none", 0.02)
    canvas.text(0.085, 0.372, "MECHANISTIC INTERPRETATION", fontsize=9, fontweight="bold", color="#F4C554")
    canvas.text(0.085, 0.315, "SBERT behaves like a semantic bag of contextual cues", fontsize=22, fontweight="bold", color=WHITE)
    canvas.text(
        0.085,
        0.258,
        "It tolerates local reordering, but depends on distributed lexical evidence and sufficient narrative coverage.",
        fontsize=12,
        color="#DCE6EA",
    )
    items = [("ORDER", "low task sensitivity", TEAL, 0.022), ("CUES", "moderate sensitivity", ORANGE, 0.057), ("COVERAGE", "highest sensitivity", GOLD, 0.092)]
    for index, (name, desc, color, width) in enumerate(items):
        yy = 0.35 - index * 0.068
        canvas.add_patch(Rectangle((0.66, yy - 0.014), width, 0.022, transform=canvas.transAxes, color=color, lw=0))
        canvas.text(0.79, yy, name, fontsize=9.5, fontweight="bold", color=WHITE, va="center")
        canvas.text(0.865, yy, desc, fontsize=9.5, color="#DCE6EA", va="center")

    rounded(canvas, 0.055, 0.095, 0.89, 0.058, PALE_ORANGE, "none", 0.012)
    canvas.text(
        0.5,
        0.124,
        "Fixed-model perturbation evidence supports this task-level inference; it is not proof of complete language understanding.",
        fontsize=10.5,
        color=INK,
        ha="center",
        va="center",
    )
    footer(canvas, "R2")
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)


def make_preview(paths: list[Path], output: Path):
    thumbnails = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((760, 428), Image.Resampling.LANCZOS)
        thumbnails.append(ImageOps.expand(image, border=2, fill="#CFC9BF"))
    preview = Image.new("RGB", (1600, 490), BG)
    preview.paste(thumbnails[0], (30, 30))
    preview.paste(thumbnails[1], (810, 30))
    preview.save(output, quality=95)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(args.metrics)
    slide_1 = args.output_dir / "06_SBERT_Official_Six_Results.png"
    slide_2 = args.output_dir / "07_SBERT_Mechanistic_Interpretation.png"
    slide_results(metrics, slide_1)
    slide_interpretation(metrics, slide_2)
    make_preview([slide_1, slide_2], args.output_dir / "00_Preview_Official_Robustness_Slides.png")


if __name__ == "__main__":
    main()
