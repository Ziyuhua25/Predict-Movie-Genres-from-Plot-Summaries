"""Evaluate the common six-input movie-genre experiment.

Supply one or more long-format CSV files using the shared schema:
movie_id, model, variant, seed, genre_id, y_true, y_score, y_pred.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss, precision_recall_fscore_support


REQUIRED = {"movie_id", "model", "variant", "seed", "genre_id", "y_true", "y_score", "y_pred"}
VARIANTS = ["original", "shuffled", "masked", "first_25", "first_50", "first_75"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", nargs="+", type=Path, required=True)
    parser.add_argument("--genre-mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("research/figures"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_predictions(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = REQUIRED.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        frames.append(frame)
    if not frames:
        raise ValueError("No prediction files were supplied.")
    predictions = pd.concat(frames, ignore_index=True)
    for column in ("movie_id", "genre_id", "y_true", "y_pred"):
        predictions[column] = pd.to_numeric(predictions[column], errors="raise").astype(int)
    if not predictions["y_true"].isin([0, 1]).all() or not predictions["y_pred"].isin([0, 1]).all():
        raise ValueError("y_true and y_pred must contain only 0 or 1.")
    keys = ["movie_id", "model", "variant", "seed", "genre_id"]
    if predictions.duplicated(keys).any():
        raise ValueError("Duplicate movie/model/variant/seed/genre rows found.")
    unknown = set(predictions["variant"]).difference(VARIANTS)
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")
    return predictions


def to_matrices(group: pd.DataFrame, genre_ids: list[int]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    movie_ids = sorted(group["movie_id"].unique())
    expected = pd.MultiIndex.from_product([movie_ids, genre_ids], names=["movie_id", "genre_id"])
    indexed = group.set_index(["movie_id", "genre_id"]).sort_index()
    if not indexed.index.equals(expected):
        missing = len(expected.difference(indexed.index))
        extra = len(indexed.index.difference(expected))
        raise ValueError(f"{group['model'].iloc[0]} / {group['variant'].iloc[0]} has an incomplete grid (missing={missing}, extra={extra}).")
    y_true = indexed["y_true"].to_numpy().reshape(len(movie_ids), len(genre_ids))
    y_pred = indexed["y_pred"].to_numpy().reshape(len(movie_ids), len(genre_ids))
    return y_true, y_pred, movie_ids


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, average: str, repeats: int, rng: np.random.Generator) -> tuple[float, float]:
    values = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sample = rng.integers(0, len(y_true), size=len(y_true))
        values[index] = f1_score(y_true[sample], y_pred[sample], average=average, zero_division=0)
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def evaluate_group(group: pd.DataFrame, genre_ids: list[int], repeats: int, seed: int) -> tuple[dict[str, object], pd.DataFrame]:
    y_true, y_pred, movie_ids = to_matrices(group, genre_ids)
    rng = np.random.default_rng(seed)
    macro_ci = bootstrap_ci(y_true, y_pred, "macro", repeats, rng)
    micro_ci = bootstrap_ci(y_true, y_pred, "micro", repeats, rng)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    model, variant, run_seed = group[["model", "variant", "seed"]].iloc[0]
    summary = {
        "model": model, "variant": variant, "seed": run_seed, "n_movies": len(movie_ids), "n_genres": len(genre_ids),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1_ci_low": macro_ci[0], "macro_f1_ci_high": macro_ci[1],
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_f1_ci_low": micro_ci[0], "micro_f1_ci_high": micro_ci[1],
        "exact_match": (y_true == y_pred).all(axis=1).mean(),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "mean_predicted_labels": y_pred.sum(axis=1).mean(),
    }
    per_genre = pd.DataFrame({"model": model, "variant": variant, "seed": run_seed, "genre_id": genre_ids,
                              "precision": precision, "recall": recall, "f1": f1, "support": support})
    return summary, per_genre


def add_drops(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, model_summary in summary.groupby("model"):
        original = model_summary.loc[model_summary["variant"].eq("original")]
        if len(original) != 1:
            raise ValueError(f"{model} must have exactly one original condition.")
        baseline = original.iloc[0]
        for _, row in model_summary.iterrows():
            if row["variant"] == "original":
                continue
            rows.append({"model": model, "variant": row["variant"], "macro_f1": row["macro_f1"], "micro_f1": row["micro_f1"],
                         "macro_f1_drop": baseline["macro_f1"] - row["macro_f1"], "micro_f1_drop": baseline["micro_f1"] - row["micro_f1"],
                         "macro_f1_relative_drop_pct": 100 * (baseline["macro_f1"] - row["macro_f1"]) / baseline["macro_f1"],
                         "micro_f1_relative_drop_pct": 100 * (baseline["micro_f1"] - row["micro_f1"]) / baseline["micro_f1"]})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = read_predictions(args.predictions)
    mapping = pd.read_csv(args.genre_mapping)
    if not {"genre_id", "genre_name"}.issubset(mapping.columns):
        raise ValueError("genre mapping must contain genre_id and genre_name")
    genre_ids = mapping["genre_id"].astype(int).tolist()
    summaries, genre_frames = [], []
    for offset, (_, group) in enumerate(predictions.groupby(["model", "variant", "seed"], sort=True)):
        summary, per_genre = evaluate_group(group, genre_ids, args.bootstrap, args.seed + offset)
        summaries.append(summary)
        genre_frames.append(per_genre)
    summary = pd.DataFrame(summaries)
    summary["variant"] = pd.Categorical(summary["variant"], VARIANTS, ordered=True)
    summary = summary.sort_values(["model", "variant"]).reset_index(drop=True)
    per_genre = pd.concat(genre_frames, ignore_index=True).merge(mapping, on="genre_id", how="left")
    drops = add_drops(summary)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    per_genre.to_csv(args.output_dir / "per_genre_metrics.csv", index=False)
    drops.to_csv(args.output_dir / "drop_summary.csv", index=False)
    print(summary[["model", "variant", "macro_f1", "micro_f1"]].to_string(index=False))
    print(f"Saved evaluation tables to {args.output_dir}")


if __name__ == "__main__":
    main()
