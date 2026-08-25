#!/usr/bin/env python3
"""Train the fixed SBERT+LR baseline and predict every perturbation variant.

This intentionally matches Member 3's official experiment:
* sentence-transformers/all-MiniLM-L6-v2 (frozen encoder)
* whitespace-only text normalization
* One-vs-Rest LogisticRegression(max_iter=2000, random_state=42)
* global decision threshold 0.25
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
THRESHOLD = 0.25
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--splits-csv", required=True, type=Path)
    parser.add_argument("--genre-mapping-csv", required=True, type=Path)
    parser.add_argument("--perturbation-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-original", type=Path)
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def parse_genre_ids(value: object) -> list[int]:
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
    else:
        parsed = value
    return [int(item) for item in parsed]


def choose_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def encode(model: SentenceTransformer, texts: pd.Series) -> np.ndarray:
    return model.encode(
        texts.apply(normalize_text).tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def compact_predictions(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    genre_ids: list[int],
    names_by_id: dict[int, str],
    variant: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, row in frame.reset_index(drop=True).iterrows():
        selected_ids = [genre_ids[j] for j in np.flatnonzero(predicted[index])]
        selected_names = [names_by_id[genre_id] for genre_id in selected_ids]
        rows.append(
            {
                "movie_id": int(row["movie_id"]),
                "title": row.get("title", ""),
                "variant": variant,
                "predicted_genre_ids": ";".join(map(str, selected_ids)),
                "predicted_genres": "; ".join(selected_names),
                "predicted_label_count": len(selected_ids),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_variant_dir = args.output_dir / "predictions_by_variant"
    per_variant_dir.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(args.train_csv)
    splits = pd.read_csv(args.splits_csv)
    mapping = pd.read_csv(args.genre_mapping_csv)
    mapping["genre_id"] = mapping["genre_id"].astype(int)
    genre_ids = mapping["genre_id"].tolist()
    names_by_id = dict(zip(mapping["genre_id"], mapping["genre_name"]))

    data = train_raw.merge(splits, on="movie_id", how="inner")
    data["genre_ids"] = data["genre_ids"].apply(parse_genre_ids)
    mlb = MultiLabelBinarizer(classes=genre_ids)
    labels = mlb.fit_transform(data["genre_ids"])
    train_mask = data["split"].eq("train").to_numpy()
    validation_mask = data["split"].eq("validation").to_numpy()

    device = choose_device()
    print(f"Loading {MODEL_NAME} on {device}")
    encoder = SentenceTransformer(MODEL_NAME, device=device)
    train_embeddings = encode(encoder, data.loc[train_mask, "overview"])
    validation_embeddings = encode(encoder, data.loc[validation_mask, "overview"])

    classifier = OneVsRestClassifier(
        LogisticRegression(max_iter=2000, random_state=SEED)
    )
    classifier.fit(train_embeddings, labels[train_mask])

    validation_scores = classifier.predict_proba(validation_embeddings)
    validation_pred = (validation_scores >= THRESHOLD).astype(np.int8)
    validation_metrics = {
        "macro_f1": float(
            f1_score(labels[validation_mask], validation_pred, average="macro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(labels[validation_mask], validation_pred, average="micro", zero_division=0)
        ),
    }

    model_dir = args.output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, model_dir / "sbert_lr_model.pkl")
    joblib.dump(mlb, model_dir / "mlb.pkl")
    (model_dir / "model_config.json").write_text(
        json.dumps(
            {
                "encoder": MODEL_NAME,
                "embedding_dimensions": int(train_embeddings.shape[1]),
                "classifier": "OneVsRestClassifier(LogisticRegression)",
                "logistic_regression_max_iter": 2000,
                "seed": SEED,
                "threshold": THRESHOLD,
                "text_column": "overview",
                "text_preprocessing": "Convert missing values to empty strings, then collapse whitespace only.",
                "encoder_frozen": True,
                "genre_ids_in_model_order": genre_ids,
                "validation_metrics_at_threshold": validation_metrics,
                "runtime_device": device,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    variant_files = sorted(args.perturbation_dir.glob("test_*.csv"))
    original_file = args.perturbation_dir / "test_original.csv"
    variant_files = [original_file] + [path for path in variant_files if path != original_file]
    if len(variant_files) != 10:
        raise ValueError(f"Expected 10 test variants, found {len(variant_files)}")

    summary_rows: list[dict[str, object]] = []
    long_frames: list[pd.DataFrame] = []
    compact_frames: list[pd.DataFrame] = []
    original_scores: np.ndarray | None = None
    original_pred: np.ndarray | None = None

    for path in variant_files:
        variant = path.stem.removeprefix("test_")
        print(f"Encoding and predicting {variant}: {path.name}")
        frame = pd.read_csv(path)
        frame["genre_ids"] = frame["genre_ids"].apply(parse_genre_ids)
        y_true = mlb.transform(frame["genre_ids"])
        embeddings = encode(encoder, frame["overview"])
        scores = classifier.predict_proba(embeddings)
        predicted = (scores >= THRESHOLD).astype(np.int8)
        if variant == "original":
            original_scores = scores.copy()
            original_pred = predicted.copy()

        macro_f1 = f1_score(y_true, predicted, average="macro", zero_division=0)
        micro_f1 = f1_score(y_true, predicted, average="micro", zero_division=0)
        summary_rows.append(
            {
                "variant": variant,
                "source_file": path.name,
                "rows": len(frame),
                "threshold": THRESHOLD,
                "macro_f1": macro_f1,
                "micro_f1": micro_f1,
                "exact_match": accuracy_score(y_true, predicted),
                "hamming_loss": hamming_loss(y_true, predicted),
                "mean_predicted_labels": predicted.sum(axis=1).mean(),
            }
        )

        n_movies = len(frame)
        n_genres = len(genre_ids)
        long_frames.append(
            pd.DataFrame(
                {
                    "movie_id": np.repeat(frame["movie_id"].to_numpy(), n_genres),
                    "title": np.repeat(frame["title"].fillna("").to_numpy(), n_genres),
                    "model": "sbert_lr",
                    "variant": variant,
                    "seed": SEED,
                    "threshold": THRESHOLD,
                    "genre_id": np.tile(genre_ids, n_movies),
                    "genre_name": np.tile([names_by_id[item] for item in genre_ids], n_movies),
                    "y_true": y_true.reshape(-1),
                    "y_score": scores.reshape(-1),
                    "y_pred": predicted.reshape(-1),
                }
            )
        )
        compact = compact_predictions(frame, predicted, genre_ids, names_by_id, variant)
        compact.to_csv(per_variant_dir / f"sbert_{variant}_predictions.csv", index=False)
        compact_frames.append(compact)

    summary = pd.DataFrame(summary_rows)
    original_macro = float(summary.loc[summary["variant"].eq("original"), "macro_f1"].iloc[0])
    original_micro = float(summary.loc[summary["variant"].eq("original"), "micro_f1"].iloc[0])
    summary["macro_f1_change_vs_original"] = summary["macro_f1"] - original_macro
    summary["micro_f1_change_vs_original"] = summary["micro_f1"] - original_micro
    summary.to_csv(args.output_dir / "sbert_perturbation_metrics.csv", index=False)
    pd.concat(compact_frames, ignore_index=True).to_csv(
        args.output_dir / "sbert_all_variants_predictions_compact.csv", index=False
    )
    pd.concat(long_frames, ignore_index=True).to_csv(
        args.output_dir / "sbert_all_variants_predictions_long.csv", index=False
    )

    verification: dict[str, object] = {
        "variant_count": len(variant_files),
        "movies_per_variant": int(summary["rows"].iloc[0]),
        "all_variants_same_row_count": bool(summary["rows"].nunique() == 1),
        "original_macro_f1": original_macro,
        "original_micro_f1": original_micro,
        "validation_macro_f1": validation_metrics["macro_f1"],
        "validation_micro_f1": validation_metrics["micro_f1"],
    }
    if (
        args.reference_original
        and args.reference_original.exists()
        and original_scores is not None
        and original_pred is not None
    ):
        reference = pd.read_csv(args.reference_original)
        reference = reference.sort_values(["movie_id", "genre_id"]).reset_index(drop=True)
        current_ids = pd.read_csv(original_file)["movie_id"].to_numpy()
        current_index = pd.MultiIndex.from_product([current_ids, genre_ids], names=["movie_id", "genre_id"])
        current = pd.Series(original_scores.reshape(-1), index=current_index).sort_index()
        reference_index = pd.MultiIndex.from_frame(reference[["movie_id", "genre_id"]])
        reference_scores = pd.Series(reference["y_score"].to_numpy(), index=reference_index).sort_index()
        difference = np.abs(current.to_numpy() - reference_scores.to_numpy())
        verification["reference_score_max_abs_difference"] = float(difference.max())
        verification["reference_scores_match_at_1e_6"] = bool(difference.max() <= 1e-6)
        reference_pred = pd.Series(reference["y_pred"].to_numpy(), index=reference_index).sort_index()
        current_pred = pd.Series(original_pred.reshape(-1), index=current_index).sort_index()
        pred_mismatches = int((current_pred.to_numpy() != reference_pred.to_numpy()).sum())
        verification["reference_prediction_mismatches"] = pred_mismatches
        verification["reference_predictions_identical"] = pred_mismatches == 0

    (args.output_dir / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    print("\nCompleted")
    print(summary.to_string(index=False))
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
