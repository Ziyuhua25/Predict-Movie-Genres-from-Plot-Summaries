# 3号成员：Sentence-BERT 模块大纲

## 模块定位

- 建议放在总 PPT 的“模型实验”部分，位于 TF-IDF 基线之后、扰动实验与综合结论之前。
- 建议讲解时长：2–3 分钟。
- 核心结论：Sentence-BERT 是一个稳定、可复现的语义基线；当前测试集 Macro-F1 为 0.548，但尚不能仅凭分类结果证明模型真正依赖深层语义。
- 这 4 页均可直接拆入总 PPT；如果总时长紧张，可保留第 2、3、4 页，把封面页删除。

## 第 1 页：Sentence-BERT 语义模型

**页面任务：** 交代研究问题和最重要结果。

**建议讲稿：**

> 3号成员负责 Sentence-BERT 语义模型。我们的目标不是堆叠复杂模型，而是建立一个可复现的语义基线，检验电影类型预测能否超越局部关键词、利用剧情的整体语义。模型在测试集上的 Macro-F1 为 0.548。

**要保留的限定：** 正式扰动实验尚未完成，因此不能说“已经证明模型理解语义”。

## 第 2 页：方法与实验控制

**流程：** 剧情摘要 → `all-MiniLM-L6-v2` → 384 维句向量 → One-vs-Rest Logistic Regression → 阈值 0.25。

**实验配置：**

| 项目 | 数值 |
|---|---:|
| 训练集 | 5,594 |
| 验证集 | 1,199 |
| 测试集 | 1,198 |
| 标签数 | 18 |
| 随机种子 | 42 |
| 逻辑回归 `max_iter` | 2,000 |

**建议讲稿：**

> 为了让不同文本表示能够公平比较，我们固定数据划分和分类头，只更换文本表示。SBERT 将每条摘要压缩为 384 维句向量，再使用 One-vs-Rest 逻辑回归完成多标签分类。阈值 0.25 在验证集上选择，并锁定用于测试集。

## 第 3 页：正式实验结果

| 数据集 | Macro-F1 | Micro-F1 |
|---|---:|---:|
| Validation | 0.5519 | 0.6249 |
| Test | 0.5476 | 0.6244 |

**建议讲稿：**

> 验证集和测试集结果非常接近，Macro-F1 只从 0.552 下降到 0.548，Micro-F1 均约为 0.624，说明结果较稳定。Micro-F1 高于 Macro-F1，说明常见类型更容易被识别，长尾或少数类型仍然是主要短板。

**总 PPT 后续可补：** 等其他成员结果齐全后，在这一页旁边增加 TF-IDF、其他模型的同口径 Macro-F1 对比。不要在没有统一数据划分和阈值策略时直接横向比较。

## 第 4 页：成功、失败与机理假设

**成功案例：** `Labyrinth`

- 真实标签：Adventure、Family、Fantasy
- 模型预测：Adventure、Family、Fantasy
- 解释：营救任务、奇幻世界与家庭关系组成一致的整体情节模式。

**失败案例：** `Brahms: The Boy II`

- 真实标签：Horror、Mystery、Thriller
- 模型预测：Comedy、Drama、Family
- 解释：摘要中的恐怖线索较含蓄，模型可能过度依赖家庭语境或训练数据中的共现模式。

**建议讲稿：**

> 成功案例表明完整情节结构可以形成稳定语义信号；失败案例说明缺少显式类型线索时，模型仍可能被表面语境带偏。所以当前最稳妥的结论是：SBERT 是稳定语义基线，而不是已经被证明的“语义理解模型”。下一步应使用词序打乱、关键词遮蔽和文本截断实验检验模型依赖的信息。

## 与总 PPT 成员的交接说明

1. 结果图中的数值可以直接引用，建议保留 3 位小数：Macro-F1 `0.548`、Micro-F1 `0.624`。
2. 与其他模型比较时必须统一：同一训练/验证/测试划分、同一 18 标签集合、同一指标定义，并说明阈值如何选择。
3. 当前只有 `original` 版本结果，不要把词序打乱、关键词遮蔽或截断写成已经完成的实验。
4. 若总 PPT 只能留一页，优先保留“第 3 页结果图”，并在右下角补一句“定性案例显示语义模式有效，但含蓄摘要仍会误导模型”。
5. 建议总 PPT 的最终研究结论使用保守措辞：`SBERT improves semantic representation / provides a stable semantic baseline`，避免写成 `SBERT understands plot semantics`。

## 数据与代码来源

- `origin/member3-sbert: research/notebooks/04_sbert_official_experiment.ipynb`
- `origin/member3-sbert: research/notebooks/03_sbert_sanity_check.ipynb`
- `origin/member3-sbert: research/src/train_sbert.py`
- 预测明细：`sbert_original_predictions.csv`，21,564 行 × 8 列。
