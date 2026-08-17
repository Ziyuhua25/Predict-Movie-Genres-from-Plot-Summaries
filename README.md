# Movie Genre Research

课程研究项目，数据来自 Kaggle 的 [Predict Movie Genres from Plot Summaries](https://www.kaggle.com/competitions/predict-movie-genres-from-plot-summaries/overview)。本项目不再以竞赛排名为目标，而是研究：

> **Keywords or Semantics? What Do Movie Genre Classifiers Learn from Plot Summaries?**

我们对比 TF-IDF 词汇表示与 Sentence-BERT 语义表示，并通过词序打乱、关键词遮蔽和文本截断，分析模型究竟依赖关键词、词序还是更完整的剧情语义。

## 五人并行工作线

| 成员 | 工作线 | 主要交付 |
| --- | --- | --- |
| 1号 | 数据质量与实验划分 | `prepare_data.py`、`splits.csv`、EDA 与数据质量说明 |
| 2号 | TF-IDF 关键词模型 | 训练脚本、统一预测、关键词与错误案例 |
| 3号 | Sentence-BERT 语义模型 | embeddings、训练脚本、统一预测与语义案例 |
| 4号 | 输入扰动与 PPT 总负责 | 六类测试输入、扰动质检、最终 PPT 与讲稿 |
| 5号 | 评估统计与 Q&A 总负责 | 统一指标、结果图表、Q&A 题库与 backup slides |

完整职责与验收标准见 [团队分工方案](team/TEAM_PLAN.md) 和 [Word 版本](team/五人分工方案.docx)。

## 每日协作方式

1. 从 `main` 创建自己的分支，禁止直接把实验代码推到 `main`。
2. 每天至少 push 一次；未完成也可提交能运行的中间版本。
3. 每个阶段通过 Pull Request 汇报，并填写“完成内容、输出文件、结果、阻塞项、复现方法”。
4. 每个人只修改自己负责的模块；公共接口修改必须先在 Issue 中说明。
5. 模型统一使用 1 号提供的 cleaned data、`splits.csv` 和 genre mapping。

建议分支名：

```text
member1-data
member2-tfidf
member3-sbert
member4-perturbation-ppt
member5-evaluation-qa
```

详细 Git 操作见 [协作规则](team/CONTRIBUTING.md)，当前进度见 [项目进度](team/STATUS.md)。

## 仓库结构

```text
research/       数据、代码、notebooks、预测与图表
team/           五人分工、项目进度与协作规则
presentation/   PPT、speaker notes 与 Q&A 材料
.github/        Issue 和 Pull Request 模板
```

研究目录的二级分类见 [research/README.md](research/README.md)。

## 统一预测接口

所有模型预测至少包含以下字段：

```text
movie_id, model, variant, seed, genre_id, y_true, y_score, y_pred
```

`variant` 统一使用：`original`、`shuffled`、`masked`、`first_25`、`first_50`、`first_75`。具体要求见 [预测接口](research/predictions/README.md)。

## 数据规则

仓库现有 CSV 已统一归入 `research/data/raw/`。后续不要继续提交大型生成数据、模型权重或 embeddings；清洗和划分规则见 [数据说明](research/data/README.md)。
