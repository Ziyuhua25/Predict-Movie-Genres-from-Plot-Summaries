# Official six prediction inputs

These are the six fixed 1,198-row internal-test inputs shared by the TF-IDF and Sentence-BERT experiments.

| Condition | File | Definition |
|---|---|---|
| original | `original.csv` | Unchanged plot text |
| shuffled | `shuffled.csv` | 100% of words shuffled inside local 10-word windows |
| masked | `masked.csv` | Highest-weight 50% of matched label-specific TF-IDF keyword occurrences replaced with `[MASK]` |
| first_25 | `first_25.csv` | First 25% of plot words retained |
| first_50 | `first_50.csv` | First 50% retained |
| first_75 | `first_75.csv` | First 75% retained |

All six files contain the same rows and columns in the same order:

```text
movie_id,title,overview,genre_ids,split
```

Use only `overview` as model input. Keep the already-trained model, preprocessing, label order, and selected decision threshold fixed across all six conditions. Do not refit a model on perturbed inputs.

Supporting files:

- `best_keywords.csv`: corrected TF-IDF keywords, 18 genres x 10 keywords.
- `genre_mapping.csv`: shared mapping for all 18 genres.

Return one prediction CSV for each condition plus Macro-F1 and Micro-F1 computed against `genre_ids`.
