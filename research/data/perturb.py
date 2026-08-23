#!/usr/bin/env python3
"""Create reproducible perturbed movie-plot test sets.

The script accepts either normal CSV files or GitHub CSV pages saved as HTML.
It selects the labeled internal test set from splits.csv/splits.html, then creates
word-shuffle, keyword-mask, and text-truncation variants without changing IDs,
labels, row order, or any non-text source column.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


# Alphanumeric words stay intact (for example, "1950s" is one token). This is
# important because a shuffle must preserve the exact word-token count.
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_github_csv_html(path: Path) -> list[dict[str, str]]:
    html = path.read_text(encoding="utf-8")
    match = re.search(
        r'<script type="application/json" data-target="react-app\.embeddedData">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError(f"Could not find GitHub CSV data in {path}")
    payload = json.loads(match.group(1))
    route = payload["payload"]["codeViewBlobRoute"]
    rows = route.get("csv")
    if not rows:
        raw_lines = route.get("rawLines", [])
        rows = list(csv.reader(raw_lines))
    header = rows[0]
    return [dict(zip(header, row)) for row in rows[1:]]


def read_table(path: Path) -> list[dict[str, str]]:
    return read_github_csv_html(path) if path.suffix.lower() in {".html", ".htm"} else read_csv(path)


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stable_rng(seed: int, movie_id: str, method: str, strength: float) -> random.Random:
    key = f"{seed}|{movie_id}|{method}|{strength:.8f}".encode("utf-8")
    number = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return random.Random(number)


def replace_spans(text: str, replacements: list[tuple[int, int, str]]) -> str:
    pieces: list[str] = []
    cursor = 0
    for start, end, value in sorted(replacements):
        pieces.append(text[cursor:start])
        pieces.append(value)
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def shuffle_words(text: str, ratio: float, rng: random.Random, window_size: int = 10) -> str:
    """Shuffle a proportion of words inside fixed local windows.

    Punctuation and whitespace stay in their original positions. Selected words
    are cyclically rotated, avoiding a no-op permutation whenever possible.
    """
    matches = list(WORD_RE.finditer(text))
    replacements: list[tuple[int, int, str]] = []
    for start in range(0, len(matches), window_size):
        window = matches[start : start + window_size]
        if len(window) < 2:
            continue
        count = int(round(len(window) * ratio))
        if count < 2:
            count = 2
        count = min(count, len(window))
        chosen = sorted(rng.sample(range(len(window)), count))
        words = [window[i].group(0) for i in chosen]
        offset = rng.randrange(1, len(words))
        rotated = words[offset:] + words[:offset]
        for local_idx, new_word in zip(chosen, rotated):
            match = window[local_idx]
            replacements.append((match.start(), match.end(), new_word))
    return replace_spans(text, replacements)


def parse_genre_ids(value: str) -> set[str]:
    return set(re.findall(r"\d+", value or ""))


def keyword_pattern(keyword: str) -> re.Pattern[str]:
    parts = keyword.strip().split()
    body = r"\s+".join(re.escape(part) for part in parts)
    # Do not match a keyword as only one component of a hyphenated/possessive word.
    boundary_chars = r"A-Za-z0-9'’-"
    return re.compile(rf"(?<![{boundary_chars}]){body}(?![{boundary_chars}])", flags=re.IGNORECASE)


def mask_keywords(
    text: str,
    ratio: float,
    genre_ids: set[str],
    keywords: list[dict[str, str]],
    scope: str = "label-specific",
) -> str:
    """Mask the highest-weight non-overlapping matched keyword occurrences."""
    relevant = keywords
    if scope == "label-specific" and genre_ids:
        relevant = [row for row in keywords if row["genre_id"] in genre_ids]

    candidates: list[tuple[float, int, int, str]] = []
    for row in relevant:
        keyword = row["keyword"].strip()
        if not keyword:
            continue
        weight = float(row.get("weight", 0) or 0)
        for match in keyword_pattern(keyword).finditer(text):
            candidates.append((weight, match.start(), match.end(), keyword))

    # Highest-weight and longest phrases win when occurrences overlap.
    candidates.sort(key=lambda item: (-item[0], -(item[2] - item[1]), item[1]))
    non_overlapping: list[tuple[float, int, int, str]] = []
    for candidate in candidates:
        _, start, end, _ = candidate
        if any(not (end <= chosen[1] or start >= chosen[2]) for chosen in non_overlapping):
            continue
        non_overlapping.append(candidate)

    if not non_overlapping:
        return text
    count = min(len(non_overlapping), max(1, math.ceil(len(non_overlapping) * ratio)))
    selected = non_overlapping[:count]
    return replace_spans(text, [(start, end, "[MASK]") for _, start, end, _ in selected])


def truncate_text(text: str, keep_ratio: float) -> str:
    matches = list(WORD_RE.finditer(text))
    if not matches:
        return text
    keep = min(len(matches), max(1, math.ceil(len(matches) * keep_ratio)))
    if keep == len(matches):
        return text
    # Retain punctuation and spaces following the last kept word, up to the next word.
    return text[: matches[keep].start()].rstrip()


def ratio_label(ratio: float) -> str:
    return str(int(round(ratio * 100)))


def validate_rows(original: list[dict[str, str]], perturbed: list[dict[str, str]], text_column: str, id_column: str) -> None:
    if len(original) != len(perturbed):
        raise AssertionError("Row count changed")
    if [row[id_column] for row in original] != [row[id_column] for row in perturbed]:
        raise AssertionError("Movie IDs or row order changed")
    for before, after in zip(original, perturbed):
        for column in before:
            if column != text_column and before[column] != after[column]:
                raise AssertionError(f"Non-text column changed: {column}")


def perturb_dataset(
    rows: list[dict[str, str]],
    method: str,
    ratio: float,
    text_column: str,
    id_column: str,
    label_column: str,
    seed: int,
    keywords: list[dict[str, str]],
    keyword_scope: str,
    window_size: int,
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in rows:
        new_row = dict(row)
        text = row.get(text_column, "") or ""
        if method == "shuffle":
            rng = stable_rng(seed, row[id_column], method, ratio)
            new_row[text_column] = shuffle_words(text, ratio, rng, window_size)
        elif method == "mask":
            new_row[text_column] = mask_keywords(
                text,
                ratio,
                parse_genre_ids(row.get(label_column, "")),
                keywords,
                keyword_scope,
            )
        elif method == "truncate":
            new_row[text_column] = truncate_text(text, ratio)
        else:
            raise ValueError(f"Unknown perturbation method: {method}")
        output.append(new_row)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Labeled source CSV containing plot text")
    parser.add_argument("--splits", required=True, type=Path, help="splits.csv or a saved GitHub splits.html")
    parser.add_argument("--genre-mapping", required=True, type=Path, help="genre_mapping.csv or saved GitHub HTML")
    parser.add_argument("--keywords", required=True, type=Path, help="TF-IDF keyword CSV")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--text-column", default="overview")
    parser.add_argument("--id-column", default="movie_id")
    parser.add_argument("--label-column", default="genre_ids")
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-ratios", nargs="+", type=float, default=[0.25, 0.50, 1.00])
    parser.add_argument("--mask-ratios", nargs="+", type=float, default=[0.10, 0.30, 0.50])
    parser.add_argument("--truncate-ratios", nargs="+", type=float, default=[0.25, 0.50, 0.75])
    parser.add_argument("--shuffle-window", type=int, default=10)
    parser.add_argument("--keyword-scope", choices=["label-specific", "all"], default="label-specific")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for ratio in args.shuffle_ratios + args.mask_ratios + args.truncate_ratios:
        if not 0 < ratio <= 1:
            raise ValueError(f"Ratios must be in (0, 1], got {ratio}")

    source_rows = read_csv(args.data)
    split_rows = read_table(args.splits)
    mapping_rows = read_table(args.genre_mapping)
    keyword_rows = read_csv(args.keywords)

    required_source = {args.id_column, args.text_column, args.label_column}
    if not source_rows or not required_source.issubset(source_rows[0]):
        raise ValueError(f"Source data must contain columns: {sorted(required_source)}")
    if not split_rows or not {args.id_column, "split"}.issubset(split_rows[0]):
        raise ValueError(f"Split data must contain {args.id_column} and split")
    if not mapping_rows or not {"genre_id", "genre_name"}.issubset(mapping_rows[0]):
        raise ValueError("Genre mapping must contain genre_id and genre_name")
    if not keyword_rows or not {"genre_id", "keyword", "weight"}.issubset(keyword_rows[0]):
        raise ValueError("Keyword data must contain genre_id, keyword, and weight")

    split_by_id = {row[args.id_column]: row["split"] for row in split_rows}
    test_rows = [dict(row, split=split_by_id[row[args.id_column]]) for row in source_rows if split_by_id.get(row[args.id_column]) == args.split_name]
    expected = sum(row["split"] == args.split_name for row in split_rows)
    if len(test_rows) != expected:
        raise ValueError(f"Expected {expected} '{args.split_name}' rows but found {len(test_rows)} in source data")
    if len({row[args.id_column] for row in test_rows}) != len(test_rows):
        raise ValueError("Duplicate movie IDs found in selected rows")
    if any(not (row.get(args.text_column) or "").strip() for row in test_rows):
        raise ValueError("Selected test set contains missing plot text")

    mapping_ids = {row["genre_id"] for row in mapping_rows}
    unknown_keyword_ids = {row["genre_id"] for row in keyword_rows} - mapping_ids
    if unknown_keyword_ids:
        raise ValueError(f"Keyword file contains unknown genre IDs: {sorted(unknown_keyword_ids)}")

    output_dir = args.output_dir
    data_dir = output_dir / "data"
    metadata_dir = output_dir / "metadata"
    fieldnames = list(source_rows[0].keys()) + ["split"]
    write_csv(data_dir / "test_original.csv", test_rows, fieldnames)
    write_csv(metadata_dir / "splits.csv", split_rows, [args.id_column, "split"])
    write_csv(metadata_dir / "genre_mapping.csv", mapping_rows, ["genre_id", "genre_name"])
    write_csv(metadata_dir / "keywords.csv", keyword_rows, ["genre_id", "keyword", "weight"])

    audit_rows: list[dict[str, object]] = []
    example_rows: list[dict[str, object]] = []
    configurations = [
        *(('shuffle', ratio) for ratio in args.shuffle_ratios),
        *(('mask', ratio) for ratio in args.mask_ratios),
        *(('truncate', ratio) for ratio in args.truncate_ratios),
    ]

    for method, ratio in configurations:
        perturbed = perturb_dataset(
            test_rows, method, ratio, args.text_column, args.id_column,
            args.label_column, args.seed, keyword_rows, args.keyword_scope,
            args.shuffle_window,
        )
        validate_rows(test_rows, perturbed, args.text_column, args.id_column)
        if method == "shuffle" and any(
            Counter(WORD_RE.findall(before[args.text_column])) != Counter(WORD_RE.findall(after[args.text_column]))
            for before, after in zip(test_rows, perturbed)
        ):
            raise AssertionError("Word shuffle changed the word-token multiset")

        # Re-run to prove deterministic output.
        repeated = perturb_dataset(
            test_rows, method, ratio, args.text_column, args.id_column,
            args.label_column, args.seed, keyword_rows, args.keyword_scope,
            args.shuffle_window,
        )
        reproducible = [r[args.text_column] for r in perturbed] == [r[args.text_column] for r in repeated]
        if not reproducible:
            raise AssertionError(f"Non-reproducible output for {method} {ratio}")

        filename = f"test_{method}_{ratio_label(ratio)}.csv"
        write_csv(data_dir / filename, perturbed, fieldnames)

        changed = [before[args.text_column] != after[args.text_column] for before, after in zip(test_rows, perturbed)]
        original_words = [len(WORD_RE.findall(row[args.text_column])) for row in test_rows]
        output_words = [len(WORD_RE.findall(row[args.text_column])) for row in perturbed]
        mask_count = sum(row[args.text_column].count("[MASK]") for row in perturbed)
        audit_rows.append({
            "file": filename,
            "method": method,
            "strength": ratio,
            "seed": args.seed,
            "rows": len(perturbed),
            "changed_rows": sum(changed),
            "changed_row_pct": round(100 * sum(changed) / len(changed), 2),
            "mean_original_words": round(sum(original_words) / len(original_words), 2),
            "mean_output_words": round(sum(output_words) / len(output_words), 2),
            "mask_tokens": mask_count,
            "reproducible": reproducible,
        })

        example_indexes = [i for i, value in enumerate(changed) if value][:2]
        for index in example_indexes:
            example_rows.append({
                args.id_column: test_rows[index][args.id_column],
                "method": method,
                "strength": ratio,
                "original_overview": test_rows[index][args.text_column],
                "perturbed_overview": perturbed[index][args.text_column],
            })

    write_csv(
        output_dir / "perturbation_audit.csv",
        audit_rows,
        ["file", "method", "strength", "seed", "rows", "changed_rows", "changed_row_pct", "mean_original_words", "mean_output_words", "mask_tokens", "reproducible"],
    )
    write_csv(
        output_dir / "perturbation_examples.csv",
        example_rows,
        [args.id_column, "method", "strength", "original_overview", "perturbed_overview"],
    )

    split_counts = Counter(row["split"] for row in split_rows)
    summary = {
        "source_rows": len(source_rows),
        "selected_split": args.split_name,
        "selected_rows": len(test_rows),
        "split_counts": dict(split_counts),
        "genre_count": len(mapping_rows),
        "keyword_rows": len(keyword_rows),
        "keyword_scope": args.keyword_scope,
        "seed": args.seed,
        "shuffle_window": args.shuffle_window,
        "generated_files": [row["file"] for row in audit_rows],
    }
    (output_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
