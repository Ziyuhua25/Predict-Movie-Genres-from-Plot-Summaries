"""Convert Member 2's compact six-input predictions to the shared long schema."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd


VARIANTS = ("original", "shuffled", "masked", "first_25", "first_50", "first_75")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--genre-mapping", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ids(value: object) -> set[int]:
    if pd.isna(value) or str(value).strip() == "":
        return set()
    return {int(item) for item in str(value).replace(";", " ").replace(",", " ").split()}


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train)
    splits = pd.read_csv(args.splits)
    mapping = pd.read_csv(args.genre_mapping)
    test_ids = set(splits.loc[splits["split"].eq("test"), "movie_id"])
    truth = train.loc[train["movie_id"].isin(test_ids), ["movie_id", "genre_ids"]].copy()
    truth["genre_ids"] = truth["genre_ids"].apply(lambda value: set(ast.literal_eval(value)))
    if truth["movie_id"].duplicated().any():
        raise ValueError("Raw train contains duplicate test movie IDs; use Member 1's cleaned data instead.")
    truth_by_id = dict(zip(truth["movie_id"], truth["genre_ids"]))
    genre_ids = mapping["genre_id"].astype(int).tolist()
    rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        path = args.submission_dir / f"{variant}_submission.csv"
        frame = pd.read_csv(path)
        if list(frame.columns) != ["movie_id", "predicted_genre_ids"]:
            raise ValueError(f"Unexpected columns in {path.name}: {list(frame.columns)}")
        if set(frame["movie_id"]) != set(truth_by_id):
            raise ValueError(f"{path.name} does not contain exactly the shared test movie IDs")
        for movie_id, predicted in frame[["movie_id", "predicted_genre_ids"]].itertuples(index=False):
            predicted_ids, actual_ids = ids(predicted), truth_by_id[movie_id]
            for genre_id in genre_ids:
                rows.append({"movie_id": movie_id, "model": "tfidf_lr", "variant": variant, "seed": args.seed,
                             "genre_id": genre_id, "y_true": int(genre_id in actual_ids), "y_score": np.nan,
                             "y_pred": int(genre_id in predicted_ids)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {len(rows):,} long-format TF-IDF decisions to {args.output}")


if __name__ == "__main__":
    main()
