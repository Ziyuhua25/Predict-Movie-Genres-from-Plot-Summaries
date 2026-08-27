# Movie Plot Perturbation Package

This package creates reproducible perturbed versions of the labeled internal test set for the movie-genre classification project.

## Data used

- Source text and labels: `train.csv`
- Fixed split: `splits.html` / `splits.csv`
- Label names: `genre_mapping.html` / `genre_mapping.csv`
- Masking vocabulary: Member 2's `best_keywords.csv`
- Selected evaluation set: 1,198 labeled rows where `split == test`

The 2,000-row Kaggle `test.csv` is not used for robustness scoring because its labels are unavailable.

## Perturbations

### Word shuffle

Words are shuffled inside local windows of 10 words. Punctuation, whitespace, IDs, labels, row order, and all other columns are preserved. Generated strengths: 25%, 50%, and 100%.

### Keyword mask

The script reads the 180 ranked TF-IDF keywords supplied by Member 2. By default, it considers keywords associated with each sample's ground-truth genres, selects non-overlapping matches by coefficient weight, and replaces the configured proportion with `[MASK]`. Generated strengths: 10%, 30%, and 50%.

To mask matches from the union of all 18 genre vocabularies instead, run with `--keyword-scope all`.

### Text truncation

The script retains the beginning of each plot by word count. Generated retained proportions: 25%, 50%, and 75%.

## Reproduce the provided datasets

```bash
python3 perturb.py \
  --data /path/to/train.csv \
  --splits /path/to/splits.html \
  --genre-mapping /path/to/genre_mapping.html \
  --keywords /path/to/best_keywords.csv \
  --output-dir /path/to/perturbation_package \
  --seed 42
```

Both ordinary CSV files and GitHub CSV pages saved as HTML are accepted for the split and genre mapping.

## Generated files

```text
perturbation_package/
├── perturb.py
├── README.md
├── generation_summary.json
├── perturbation_audit.csv
├── perturbation_examples.csv
├── metadata/
│   ├── splits.csv
│   ├── genre_mapping.csv
│   └── keywords.csv
└── data/
    ├── test_original.csv
    ├── test_shuffle_25.csv
    ├── test_shuffle_50.csv
    ├── test_shuffle_100.csv
    ├── test_mask_10.csv
    ├── test_mask_30.csv
    ├── test_mask_50.csv
    ├── test_truncate_25.csv
    ├── test_truncate_50.csv
    └── test_truncate_75.csv
```

Every dataset has the same 1,198 rows and the same columns:

```text
movie_id, title, overview, genre_ids, split
```

Only `overview` changes in perturbed files.

## Validation

The script automatically verifies:

- identical row count;
- identical movie IDs and row order;
- unchanged labels and non-text columns;
- no missing plot text in the selected test set;
- keyword IDs belong to the shared 18-label mapping;
- identical output when regenerated with the same seed.

See `perturbation_audit.csv` for observed changed-row coverage and output lengths. See `perturbation_examples.csv` for presentation-ready before/after examples.

## Recommended evaluation

Run both the TF-IDF and Sentence-BERT models on `test_original.csv` and every perturbed file. Compare paired predictions using Micro-F1, Macro-F1, and:

```text
Delta F1 = F1(original) - F1(perturbed)
```
