# Predictions

所有模型预测使用统一 schema：

```text
movie_id, model, variant, seed, genre_id, y_true, y_score, y_pred
```

- `model`：`tfidf_lr` 或 `sbert_lr`
- `variant`：`original`、`shuffled`、`masked`、`first_25`、`first_50`、`first_75`
- `y_true`、`y_pred`：0 或 1
- `y_score`：用于阈值与评价的连续分数

预测文件默认不提交，避免仓库膨胀；由 5 号统一收集并生成汇总结果。

