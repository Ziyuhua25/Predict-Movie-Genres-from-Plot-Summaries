# Research workspace

所有研究工作按功能放在本目录，避免在仓库根目录堆放文件。

```text
data/
  raw/          Kaggle 原始 CSV
  processed/    本地生成的清洗数据，不提交
src/            可复用的数据、模型、扰动、评估代码
notebooks/      EDA 与实验 notebook
predictions/    统一格式的模型预测
figures/        汇报用结果图表
```

成员与目录的主要对应关系：

| 成员 | 主要目录 |
| --- | --- |
| 1号 | `data/`、`src/prepare_data.py`、`notebooks/` |
| 2号 | `src/train_tfidf.py`、`predictions/` |
| 3号 | `src/train_sbert.py`、`predictions/` |
| 4号 | `src/perturbations.py`、`../presentation/` |
| 5号 | `src/evaluate.py`、`src/visualize.py`、`figures/` |
