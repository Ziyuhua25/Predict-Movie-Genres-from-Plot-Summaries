# Data

Kaggle 的 `train.csv`、`test.csv`、`movies_genres.csv` 和 `sample_submission.csv` 已集中放在 `research/data/raw/`。后续生成的大型数据文件受 `.gitignore` 保护，不再上传 GitHub。

全组只使用 1 号生成的 cleaned data、`splits.csv` 和 genre mapping。清洗保留原始文本列 `overview_raw`，另建仅压缩空白字符的 `overview_clean`，并在 `processed/dropped_rows.csv` 记录删除原因。

`splits.csv` 至少包含：

```text
movie_id, split
```

其中 `split` 只能是 `train`、`validation` 或 `test`。

## Reproducing the shared dataset

From the repository root, run:

```powershell
python research/src/prepare_data.py --seed 42
```

This creates local (ignored) files in `processed/`: `cleaned_train.csv` and
`dropped_rows.csv`. It also regenerates the versioned, shared artifacts
`splits.csv`, `genre_mapping.csv`, and `DATA_QUALITY.md`. The split is 70/15/15
with deterministic iterative multi-label stratification; do not regenerate it with
a different seed during comparative experiments.
