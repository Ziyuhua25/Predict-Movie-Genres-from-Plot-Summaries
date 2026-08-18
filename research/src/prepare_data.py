"""Clean the movie-genre training data and create reproducible experiment splits.

The script deliberately uses only the Python standard library so every team member
can regenerate the shared artifacts without a machine-specific environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("train", "validation", "test")
WHITESPACE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=root / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "processed")
    parser.add_argument("--splits-path", type=Path, default=root / "data" / "splits.csv")
    parser.add_argument(
        "--genre-mapping-path", type=Path, default=root / "data" / "genre_mapping.csv"
    )
    parser.add_argument(
        "--report-path", type=Path, default=root / "data" / "DATA_QUALITY.md"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser.parse_args()


def validate_ratios(args: argparse.Namespace) -> dict[str, float]:
    ratios = {
        "train": args.train_ratio,
        "validation": args.validation_ratio,
        "test": args.test_ratio,
    }
    if any(value <= 0 for value in ratios.values()):
        raise ValueError("All split ratios must be positive.")
    if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Split ratios must sum to 1.0.")
    return ratios


def normalize_text(value: str | None) -> str:
    """Normalize spacing only; retain case, punctuation, and all lexical content."""
    return WHITESPACE.sub(" ", (value or "").strip())


def parse_genre_ids(value: str | None, known_ids: set[int]) -> tuple[int, ...]:
    """Parse Kaggle's JSON-like label field and reject unknown genre identifiers."""
    text = (value or "").strip()
    if not text:
        return ()
    try:
        parsed = json.loads(text)
        raw_ids = parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        raw_ids = re.findall(r"\d+", text)

    try:
        labels = tuple(sorted({int(item) for item in raw_ids}))
    except (TypeError, ValueError):
        return ()
    return labels if labels and set(labels).issubset(known_ids) else ()


def read_genre_mapping(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        mapping = {int(row["id"]): row["name"].strip() for row in csv.DictReader(handle)}
    if not mapping:
        raise ValueError(f"No genre mapping rows found in {path}")
    return mapping


def read_candidates(train_path: Path, known_ids: set[int]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    with train_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"movie_id", "title", "overview", "genre_ids"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{train_path} must contain {sorted(required)}")
        for row_number, row in enumerate(reader, start=2):
            movie_id = (row.get("movie_id") or "").strip()
            overview_raw = row.get("overview") or ""
            overview_clean = normalize_text(overview_raw)
            labels = parse_genre_ids(row.get("genre_ids"), known_ids)
            if not movie_id:
                rejected.append({"row_number": str(row_number), "movie_id": "", "reason": "missing_movie_id"})
            elif not overview_clean:
                rejected.append({"row_number": str(row_number), "movie_id": movie_id, "reason": "missing_overview"})
            elif not labels:
                rejected.append({"row_number": str(row_number), "movie_id": movie_id, "reason": "invalid_or_missing_labels"})
            else:
                candidates.append(
                    {
                        "row_number": row_number,
                        "movie_id": movie_id,
                        "title": normalize_text(row.get("title")),
                        "overview_raw": overview_raw,
                        "overview_clean": overview_clean,
                        "genre_ids": labels,
                    }
                )
    return candidates, rejected


def clean_candidates(
    candidates: list[dict[str, Any]], rejected: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Remove duplicate rows and conflicting duplicated records to prevent leakage."""
    removal_counts: Counter[str] = Counter(item["reason"] for item in rejected)
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_id[row["movie_id"]].append(row)

    unique_ids: list[dict[str, Any]] = []
    for movie_id, rows in by_id.items():
        signatures = {(row["overview_clean"], row["genre_ids"]) for row in rows}
        if len(signatures) > 1:
            for row in rows:
                rejected.append(
                    {"row_number": str(row["row_number"]), "movie_id": movie_id, "reason": "conflicting_movie_id"}
                )
            removal_counts["conflicting_movie_id"] += len(rows)
        else:
            unique_ids.append(rows[0])
            for row in rows[1:]:
                rejected.append(
                    {"row_number": str(row["row_number"]), "movie_id": movie_id, "reason": "duplicate_movie_id"}
                )
                removal_counts["duplicate_movie_id"] += 1

    by_overview: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in unique_ids:
        by_overview[row["overview_clean"]].append(row)

    cleaned: list[dict[str, Any]] = []
    for rows in by_overview.values():
        label_sets = {row["genre_ids"] for row in rows}
        if len(label_sets) > 1:
            for row in rows:
                rejected.append(
                    {
                        "row_number": str(row["row_number"]),
                        "movie_id": row["movie_id"],
                        "reason": "conflicting_duplicate_overview",
                    }
                )
            removal_counts["conflicting_duplicate_overview"] += len(rows)
        else:
            cleaned.append(rows[0])
            for row in rows[1:]:
                rejected.append(
                    {
                        "row_number": str(row["row_number"]),
                        "movie_id": row["movie_id"],
                        "reason": "duplicate_overview",
                    }
                )
                removal_counts["duplicate_overview"] += 1
    return cleaned, removal_counts


def allocate_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    counts = {split: math.floor(total * ratio) for split, ratio in ratios.items()}
    remainder = total - sum(counts.values())
    for split in sorted(ratios, key=lambda name: (total * ratios[name]) % 1, reverse=True)[:remainder]:
        counts[split] += 1
    return counts


def multilabel_split(
    rows: list[dict[str, Any]], ratios: dict[str, float], seed: int
) -> dict[str, str]:
    """Iterative multi-label stratification with exact sizes and deterministic ties."""
    rng = random.Random(seed)
    split_sizes = allocate_counts(len(rows), ratios)
    label_totals = Counter(label for row in rows for label in row["genre_ids"])
    desired_labels = {
        split: {label: label_totals[label] * ratios[split] for label in label_totals}
        for split in SPLITS
    }
    desired_sizes = {split: float(split_sizes[split]) for split in SPLITS}
    remaining = set(range(len(rows)))
    label_to_rows = {
        label: {index for index, row in enumerate(rows) if label in row["genre_ids"]}
        for label in label_totals
    }
    assignments: dict[str, str] = {}

    def choose_split(label: int | None) -> str:
        available = [split for split in SPLITS if desired_sizes[split] > 0]
        if label is not None:
            largest_need = max(desired_labels[split][label] for split in available)
            available = [split for split in available if desired_labels[split][label] == largest_need]
        largest_capacity = max(desired_sizes[split] for split in available)
        available = [split for split in available if desired_sizes[split] == largest_capacity]
        return rng.choice(available)

    def assign(index: int, split: str) -> None:
        row = rows[index]
        assignments[row["movie_id"]] = split
        remaining.remove(index)
        desired_sizes[split] -= 1
        for label in row["genre_ids"]:
            desired_labels[split][label] -= 1
            label_to_rows[label].discard(index)

    # Allocate the least frequent remaining label first, as in iterative
    # stratification, to protect rare labels in all three splits.
    while remaining:
        active_labels = [label for label, indexes in label_to_rows.items() if indexes]
        if not active_labels:
            for index in sorted(remaining, key=lambda item: rows[item]["movie_id"]):
                assign(index, choose_split(None))
            break
        smallest = min(len(label_to_rows[label]) for label in active_labels)
        label = rng.choice(sorted(label for label in active_labels if len(label_to_rows[label]) == smallest))
        indexes = list(label_to_rows[label])
        rng.shuffle(indexes)
        for index in indexes:
            if index in remaining:
                assign(index, choose_split(label))
    return assignments


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    raw_count: int,
    cleaned: list[dict[str, Any]],
    removal_counts: Counter[str],
    genre_mapping: dict[int, str],
    source_genre_count: int,
    excluded_labels: list[str],
    assignments: dict[str, str],
    seed: int,
    ratios: dict[str, float],
) -> None:
    label_counts = Counter(label for row in cleaned for label in row["genre_ids"])
    text_lengths = sorted(len(row["overview_clean"].split()) for row in cleaned)
    cardinality = Counter(len(row["genre_ids"]) for row in cleaned)
    split_counts = Counter(assignments.values())
    split_label_counts = {
        split: Counter(
            label for row in cleaned if assignments[row["movie_id"]] == split for label in row["genre_ids"]
        )
        for split in SPLITS
    }

    def percentile(p: float) -> int:
        index = max(0, min(len(text_lengths) - 1, math.ceil(p * len(text_lengths)) - 1))
        return text_lengths[index]

    lines = [
        "# EDA and Data Quality Report",
        "",
        "Generated by `research/src/prepare_data.py` from `research/data/raw/train.csv`.",
        "",
        "## Cleaning Policy",
        "",
        "- Keep both `overview_raw` and whitespace-normalized `overview_clean`; no lexical content is removed.",
        "- Drop rows missing a movie ID, non-empty overview, or valid genre IDs from `movies_genres.csv`.",
        "- For repeated movie IDs, keep one exact duplicate but drop every row when overview or labels conflict.",
        "- For repeated cleaned overviews across movie IDs, keep one matching-label record; drop all conflicting-label groups. This prevents text leakage across splits.",
        "",
        "## Dataset Summary",
        "",
        f"- Raw training rows: {raw_count:,}",
        f"- Usable, de-duplicated rows: {len(cleaned):,}",
        f"- Removed rows: {raw_count - len(cleaned):,}",
        f"- Source genre mapping classes: {source_genre_count}",
        f"- Trainable genre classes: {len(genre_mapping)}",
        f"- Label cardinality (mean labels/movie): {sum(len(row['genre_ids']) for row in cleaned) / len(cleaned):.2f}",
        f"- Overview length in words (p25 / median / p75): {percentile(.25)} / {percentile(.50)} / {percentile(.75)}",
        "",
        "## Zero-Support Labels",
        "",
    ]
    if excluded_labels:
        lines.append(
            "The following source labels have no usable training examples and are excluded "
            "from `genre_mapping.csv`, so model training does not create an all-zero target: "
            + ", ".join(excluded_labels)
            + "."
        )
    else:
        lines.append("All source labels have usable training examples.")

    lines.extend([
        "",
        "## Removed Rows",
        "",
        "| Reason | Rows |",
        "| --- | ---: |",
    ])
    lines.extend(f"| {reason} | {count:,} |" for reason, count in sorted(removal_counts.items()))
    if not removal_counts:
        lines.append("| None | 0 |")

    lines.extend(["", "## Label Cardinality", "", "| Labels per movie | Movies |", "| ---: | ---: |"])
    lines.extend(f"| {size} | {count:,} |" for size, count in sorted(cardinality.items()))

    lines.extend(["", "## Genre Distribution", "", "| Genre ID | Genre | Movies |", "| ---: | --- | ---: |"])
    lines.extend(
        f"| {genre_id} | {genre_mapping[genre_id]} | {label_counts[genre_id]:,} |"
        for genre_id in sorted(genre_mapping)
    )

    lines.extend(
        [
            "",
            "## Fixed Experimental Split",
            "",
            f"Seed: `{seed}`. Target proportions: train {ratios['train']:.0%}, validation {ratios['validation']:.0%}, test {ratios['test']:.0%}.",
            "`splits.csv` is produced with deterministic iterative multi-label stratification: rare labels are assigned first, and each row goes to the split with the greatest remaining need for that label while exact split sizes are enforced.",
            "",
            "| Split | Movies |",
            "| --- | ---: |",
        ]
    )
    lines.extend(f"| {split} | {split_counts[split]:,} |" for split in SPLITS)
    lines.extend(["", "### Per-Genre Split Counts", "", "| Genre | Train | Validation | Test |", "| --- | ---: | ---: | ---: |"])
    lines.extend(
        f"| {genre_mapping[genre_id]} ({genre_id}) | {split_label_counts['train'][genre_id]:,} | {split_label_counts['validation'][genre_id]:,} | {split_label_counts['test'][genre_id]:,} |"
        for genre_id in sorted(genre_mapping)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ratios = validate_ratios(args)
    genre_mapping = read_genre_mapping(args.raw_dir / "movies_genres.csv")
    candidates, rejected = read_candidates(args.raw_dir / "train.csv", set(genre_mapping))
    raw_count = len(candidates) + len(rejected)
    cleaned, removal_counts = clean_candidates(candidates, rejected)
    if len(cleaned) < len(SPLITS):
        raise ValueError("Too few usable rows to create all experimental splits.")

    source_genre_count = len(genre_mapping)
    observed_labels = {label for row in cleaned for label in row["genre_ids"]}
    excluded_label_ids = sorted(set(genre_mapping) - observed_labels)
    excluded_labels = [f"{genre_mapping[label]} ({label})" for label in excluded_label_ids]
    if excluded_label_ids:
        print(
            "Excluded zero-support genres from the trainable mapping: "
            + ", ".join(excluded_labels)
        )
    genre_mapping = {label: name for label, name in genre_mapping.items() if label in observed_labels}
    assignments = multilabel_split(cleaned, ratios, args.seed)
    processed_rows = [
        {
            "movie_id": row["movie_id"],
            "title": row["title"],
            "overview_raw": row["overview_raw"],
            "overview_clean": row["overview_clean"],
            "genre_ids": json.dumps(row["genre_ids"]),
            "genre_names": json.dumps([genre_mapping[label] for label in row["genre_ids"]]),
        }
        for row in sorted(cleaned, key=lambda item: int(item["movie_id"]))
    ]
    write_csv(
        args.output_dir / "cleaned_train.csv",
        ["movie_id", "title", "overview_raw", "overview_clean", "genre_ids", "genre_names"],
        processed_rows,
    )
    write_csv(args.output_dir / "dropped_rows.csv", ["row_number", "movie_id", "reason"], rejected)
    write_csv(
        args.splits_path,
        ["movie_id", "split"],
        [
            {"movie_id": movie_id, "split": assignments[movie_id]}
            for movie_id in sorted(assignments, key=int)
        ],
    )
    write_csv(
        args.genre_mapping_path,
        ["genre_id", "genre_name"],
        [
            {"genre_id": genre_id, "genre_name": genre_mapping[genre_id]}
            for genre_id in sorted(genre_mapping)
        ],
    )
    write_report(
        args.report_path,
        raw_count,
        cleaned,
        removal_counts,
        genre_mapping,
        source_genre_count,
        excluded_labels,
        assignments,
        args.seed,
        ratios,
    )
    print(f"Cleaned {raw_count:,} raw rows to {len(cleaned):,} usable rows.")
    print(f"Wrote {args.splits_path}, {args.genre_mapping_path}, and {args.report_path}.")


if __name__ == "__main__":
    main()
