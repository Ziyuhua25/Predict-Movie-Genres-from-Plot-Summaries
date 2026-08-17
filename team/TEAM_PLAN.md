# 五人最终分工

## 1号：数据质量与实验划分

解析多标签，处理空标签、无剧情、重复和冲突样本；固定 train/validation/test；完成 EDA。全组只使用其提交的 cleaned data、`splits.csv` 和 genre mapping。

## 2号：TF-IDF关键词模型

训练 One-vs-Rest TF-IDF + Logistic Regression，比较 unigram/bigram，调阈值，提取各类型关键词，并完成成功与失败案例分析。

## 3号：Sentence-BERT语义模型

使用 `all-MiniLM-L6-v2` 等轻量句向量生成 embeddings，训练相同分类头与阈值流程，输出格式必须与 2 号完全一致。

## 4号：输入扰动与PPT总负责

实现 word shuffle、keyword mask 和文本 25%/50%/75% 截断；检查可复现性和文本质量。同时建立 PPT 模板、整合全组 slides、统一图表风格并组织彩排。

## 5号：评估统计与Q&A总负责

建立统一评价脚本，计算 Macro-F1、Micro-F1、per-genre F1、性能下降和置信区间；生成最终图表。维护 Q&A 题库、问题路由和 backup slides。

## 并行原则

- Day 1 起五条工作线同时启动。
- 没有真实输入时使用小样本、mock keywords 或 dummy predictions 开发。
- 2号与3号使用相同数据和分类接口；4号只提供统一扰动版本；5号只接收统一预测 schema。
- 研究结论必须回答“模型学到了什么”，而不只是“哪个模型分数最高”。

