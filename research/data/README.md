# Data

Kaggle 的 `train.csv`、`test.csv`、`movies_genres.csv` 和 `sample_submission.csv` 已集中放在 `research/data/raw/`。后续生成的大型数据文件受 `.gitignore` 保护，不再上传 GitHub。

全组只使用 1 号生成的 cleaned data、`splits.csv` 和 genre mapping。清洗必须保留原始文本列 `overview_raw`，另建 `overview_clean`，并记录删除样本的原因。

`splits.csv` 至少包含：

```text
movie_id, split
```

其中 `split` 只能是 `train`、`validation` 或 `test`。
