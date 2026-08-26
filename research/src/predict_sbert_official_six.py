#!/usr/bin/env python3
"""Evaluate the saved SBERT + One-vs-Rest LR model on the official six inputs.

The encoder, classifier, label order, preprocessing, and threshold are loaded from
the saved Member 3 model and remain fixed. Nothing is refit on perturbed text.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_recall_fscore_support


VARIANTS = ("original", "shuffled", "masked", "first_25", "first_50", "first_75")
REQUIRED_COLUMNS = ("movie_id", "title", "overview", "genre_ids", "split")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-original", type=Path)
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def parse_genre_ids(value: object) -> list[int]:
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    return [int(item) for item in parsed]


def choose_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def validate_inputs(input_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    baseline_ids: list[int] | None = None
    baseline_labels: list[str] | None = None
    for variant in VARIANTS:
        path = input_dir / f"{variant}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing official input: {path}")
        frame = pd.read_csv(path)
        if tuple(frame.columns) != REQUIRED_COLUMNS:
            raise ValueError(f"Unexpected columns in {path.name}: {list(frame.columns)}")
        if len(frame) != 1198 or frame["movie_id"].nunique() != 1198:
            raise ValueError(f"{path.name} must contain 1,198 unique movies")
        if frame["overview"].isna().any() or not frame["split"].eq("test").all():
            raise ValueError(f"Invalid overview or split values in {path.name}")
        ids = frame["movie_id"].astype(int).tolist()
        labels = frame["genre_ids"].astype(str).tolist()
        if baseline_ids is None:
            baseline_ids, baseline_labels = ids, labels
        elif ids != baseline_ids or labels != baseline_labels:
            raise ValueError(f"Rows or labels are not aligned in {path.name}")
        frames[variant] = frame
    return frames


def compact_predictions(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    genre_ids: list[int],
    names_by_id: dict[int, str],
    variant: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, row in frame.reset_index(drop=True).iterrows():
        selected = [genre_ids[j] for j in np.flatnonzero(predicted[index])]
        rows.append(
            {
                "movie_id": int(row["movie_id"]),
                "title": row["title"],
                "model": "sbert_lr",
                "variant": variant,
                "predicted_genre_ids": ";".join(map(str, selected)),
                "predicted_genres": "; ".join(names_by_id[item] for item in selected),
                "predicted_label_count": len(selected),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    long_dir = args.output_dir / "by_variant_long"
    compact_dir = args.output_dir / "by_variant_compact"
    long_dir.mkdir(exist_ok=True)
    compact_dir.mkdir(exist_ok=True)

    config = json.loads((args.model_dir / "model_config.json").read_text(encoding="utf-8"))
    classifier = joblib.load(args.model_dir / "sbert_lr_model.pkl")
    mlb = joblib.load(args.model_dir / "mlb.pkl")
    mapping = pd.read_csv(args.model_dir / "genre_mapping.csv")
    mapping["genre_id"] = mapping["genre_id"].astype(int)

    genre_ids = [int(item) for item in config["genre_ids_in_model_order"]]
    if list(map(int, mlb.classes_)) != genre_ids:
        raise ValueError("Saved MultiLabelBinarizer order does not match model_config.json")
    if mapping["genre_id"].tolist() != genre_ids:
        raise ValueError("genre_mapping.csv order does not match the saved model")
    names_by_id = dict(zip(mapping["genre_id"], mapping["genre_name"]))
    threshold = float(config["threshold"])
    seed = int(config["seed"])

    frames = validate_inputs(args.input_dir)
    device = choose_device()
    print(f"Loading {config['encoder']} on {device}")
    encoder = SentenceTransformer(config["encoder"], device=device)

    summary_rows: list[dict[str, object]] = []
    per_genre_frames: list[pd.DataFrame] = []
    long_frames: list[pd.DataFrame] = []
    compact_frames: list[pd.DataFrame] = []
    original_scores: np.ndarray | None = None
    original_pred: np.ndarray | None = None

    for variant in VARIANTS:
        frame = frames[variant].copy()
        label_lists = frame["genre_ids"].apply(parse_genre_ids)
        y_true = mlb.transform(label_lists)
        texts = frame["overview"].apply(normalize_text).tolist()
        print(f"Encoding and predicting {variant} ({len(frame):,} movies)")
        embeddings = encoder.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        scores = classifier.predict_proba(embeddings)
        predicted = (scores >= threshold).astype(np.int8)
        if variant == "original":
            original_scores, original_pred = scores.copy(), predicted.copy()

        summary_rows.append(
            {
                "variant": variant,
                "source_file": f"{variant}.csv",
                "rows": len(frame),
                "threshold": threshold,
                "macro_f1": f1_score(y_true, predicted, average="macro", zero_division=0),
                "micro_f1": f1_score(y_true, predicted, average="micro", zero_division=0),
                "exact_match": accuracy_score(y_true, predicted),
                "hamming_loss": hamming_loss(y_true, predicted),
                "mean_predicted_labels": predicted.sum(axis=1).mean(),
            }
        )

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, predicted, average=None, zero_division=0
        )
        per_genre_frames.append(
            pd.DataFrame(
                {
                    "variant": variant,
                    "genre_id": genre_ids,
                    "genre_name": [names_by_id[item] for item in genre_ids],
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "support": support,
                }
            )
        )

        n_movies, n_genres = len(frame), len(genre_ids)
        long_frame = pd.DataFrame(
            {
                "movie_id": np.repeat(frame["movie_id"].astype(int).to_numpy(), n_genres),
                "title": np.repeat(frame["title"].fillna("").to_numpy(), n_genres),
                "model": "sbert_lr",
                "variant": variant,
                "seed": seed,
                "threshold": threshold,
                "genre_id": np.tile(genre_ids, n_movies),
                "genre_name": np.tile([names_by_id[item] for item in genre_ids], n_movies),
                "y_true": y_true.reshape(-1),
                "y_score": scores.reshape(-1),
                "y_pred": predicted.reshape(-1),
            }
        )
        long_frame.to_csv(long_dir / f"sbert_{variant}_predictions.csv", index=False)
        long_frames.append(long_frame)

        compact = compact_predictions(frame, predicted, genre_ids, names_by_id, variant)
        compact.to_csv(compact_dir / f"sbert_{variant}_predictions.csv", index=False)
        compact_frames.append(compact)

    summary = pd.DataFrame(summary_rows)
    original = summary.loc[summary["variant"].eq("original")].iloc[0]
    summary["macro_f1_change_vs_original"] = summary["macro_f1"] - original["macro_f1"]
    summary["micro_f1_change_vs_original"] = summary["micro_f1"] - original["micro_f1"]
    summary.to_csv(args.output_dir / "sbert_official_six_metrics.csv", index=False)
    pd.concat(per_genre_frames, ignore_index=True).to_csv(
        args.output_dir / "sbert_official_six_per_genre_metrics.csv", index=False
    )
    pd.concat(long_frames, ignore_index=True).to_csv(
        args.output_dir / "sbert_official_six_predictions_long.csv", index=False
    )
    pd.concat(compact_frames, ignore_index=True).to_csv(
        args.output_dir / "sbert_official_six_predictions_compact.csv", index=False
    )

    verification: dict[str, object] = {
        "official_variant_count": len(VARIANTS),
        "official_variants": list(VARIANTS),
        "movies_per_variant": 1198,
        "genres": len(genre_ids),
        "long_rows_per_variant": 1198 * len(genre_ids),
        "combined_long_rows": len(VARIANTS) * 1198 * len(genre_ids),
        "model_refit_on_perturbations": False,
        "encoder": config["encoder"],
        "classifier": config["classifier"],
        "threshold": threshold,
        "seed": seed,
        "runtime_device": device,
        "original_macro_f1": float(original["macro_f1"]),
        "original_micro_f1": float(original["micro_f1"]),
    }
    if args.reference_original and args.reference_original.exists():
        if original_scores is None or original_pred is None:
            raise RuntimeError("Original predictions were not generated")
        reference = pd.read_csv(args.reference_original)
        if "variant" in reference.columns:
            reference = reference.loc[reference["variant"].eq("original")]
        reference = reference.sort_values(["movie_id", "genre_id"])
        if len(reference) != len(frames["original"]) * len(genre_ids):
            raise ValueError("Reference original has an unexpected number of rows")
        current_index = pd.MultiIndex.from_product(
            [frames["original"]["movie_id"].astype(int), genre_ids],
            names=["movie_id", "genre_id"],
        )
        current_scores = pd.Series(original_scores.reshape(-1), index=current_index).sort_index()
        current_pred = pd.Series(original_pred.reshape(-1), index=current_index).sort_index()
        reference_index = pd.MultiIndex.from_frame(reference[["movie_id", "genre_id"]])
        reference_scores = pd.Series(reference["y_score"].to_numpy(), index=reference_index).sort_index()
        reference_pred = pd.Series(reference["y_pred"].to_numpy(), index=reference_index).sort_index()
        max_difference = float(np.abs(current_scores.to_numpy() - reference_scores.to_numpy()).max())
        mismatches = int((current_pred.to_numpy() != reference_pred.to_numpy()).sum())
        verification.update(
            {
                "reference_original_score_max_abs_difference": max_difference,
                "reference_original_prediction_mismatches": mismatches,
                "reference_original_predictions_identical": mismatches == 0,
            }
        )

    (args.output_dir / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    print("\nOfficial six-input evaluation complete")
    print(summary.to_string(index=False))
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
