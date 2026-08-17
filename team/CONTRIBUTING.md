# 协作与提交规则

## 第一次开始工作

```bash
git clone https://github.com/Ziyuhua25/Predict-Movie-Genres-from-Plot-Summaries.git
cd Predict-Movie-Genres-from-Plot-Summaries
git switch -c memberN-topic
```

把 `memberN-topic` 替换成 README 中对应的分支名。

## 每日同步

```bash
git status
git add <本次修改的文件>
git commit -m "简短说明完成了什么"
git push -u origin memberN-topic
```

不要使用 `git add .` 盲目提交；先确认没有 CSV、embeddings、模型权重和临时文件。

## Pull Request 规则

- 一个 PR 只解决一个明确任务。
- 标题建议：`[成员N] 动词 + 交付内容`。
- PR 必须填写仓库模板，并提供可复现命令。
- 合并前至少由另一位成员检查一次。
- 发生冲突时先同步 `main`，不要覆盖其他成员代码。

```bash
git fetch origin
git rebase origin/main
```

## 公共接口

- 数据划分只能由 1 号维护；其他成员不得自行重新划分。
- 预测文件必须遵守 `research/predictions/README.md` 的 schema。
- 图表必须包含标题、坐标含义、指标名称与一句话 takeaway。
- 每个随机实验记录 `seed`、模型版本和关键超参数。

## 完成定义

任务只有同时满足以下条件才算完成：代码可运行、输出已保存、复现方法已写、结果已解释、PR 已通过检查。
