"""
evaluate.py
5号负责：统一评估脚本
计算 Macro-F1、Micro-F1、per-genre F1、性能下降与置信区间

用法：
    python research/src/evaluate.py --predictions_dir research/predictions --output_dir research/figures
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import f1_score, precision_score, recall_score


def compute_metrics(df_group):
    """
    对一个 (model, variant) 组的预测计算指标
    多标签：每个 genre_id 单独计算，然后聚合
    """
    # 获取所有唯一的 genre_id
    genres = sorted(df_group['genre_id'].unique())

    y_true_list = []
    y_pred_list = []
    y_score_list = []

    # 按 genre_id 分组，重组为矩阵形式
    for genre in genres:
        gdf = df_group[df_group['genre_id'] == genre].sort_values('movie_id')
        y_true_list.append(gdf['y_true'].values)
        y_pred_list.append(gdf['y_pred'].values)
        y_score_list.append(gdf['y_score'].values)

    # 转置：行=样本，列=genre
    Y_true = np.array(y_true_list).T  # (n_samples, n_genres)
    Y_pred = np.array(y_pred_list).T
    Y_score = np.array(y_score_list).T

    # Micro-F1：全局计算
    micro_f1 = f1_score(Y_true.ravel(), Y_pred.ravel(), average='micro', zero_division=0)

    # Macro-F1：每个genre的F1取平均
    per_genre_f1 = []
    for i in range(Y_true.shape[1]):
        f1 = f1_score(Y_true[:, i], Y_pred[:, i], zero_division=0)
        per_genre_f1.append(f1)

    macro_f1 = np.mean(per_genre_f1)

    # 每个genre的精确率和召回率
    per_genre_precision = []
    per_genre_recall = []
    for i in range(Y_true.shape[1]):
        p = precision_score(Y_true[:, i], Y_pred[:, i], zero_division=0)
        r = recall_score(Y_true[:, i], Y_pred[:, i], zero_division=0)
        per_genre_precision.append(p)
        per_genre_recall.append(r)

    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'per_genre_f1': per_genre_f1,
        'per_genre_precision': per_genre_precision,
        'per_genre_recall': per_genre_recall,
        'n_samples': Y_true.shape[0],
        'n_genres': Y_true.shape[1]
    }


def compute_drop(original_f1, perturbed_f1):
    """计算性能下降百分比"""
    if original_f1 == 0:
        return 0.0
    return (original_f1 - perturbed_f1) / original_f1 * 100.0


def bootstrap_confidence_interval(y_true, y_pred, n_bootstrap=1000, confidence=0.95):
    """
    用bootstrap计算Macro-F1的置信区间
    """
    rng = np.random.RandomState(42)
    n_samples = y_true.shape[0]
    bootstrapped_f1s = []

    for _ in range(n_bootstrap):
        indices = rng.randint(0, n_samples, n_samples)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        per_genre_f1 = []
        for i in range(y_true.shape[1]):
            f1 = f1_score(y_true_boot[:, i], y_pred_boot[:, i], zero_division=0)
            per_genre_f1.append(f1)
        bootstrapped_f1s.append(np.mean(per_genre_f1))

    alpha = 1 - confidence
    lower = np.percentile(bootstrapped_f1s, alpha/2 * 100)
    upper = np.percentile(bootstrapped_f1s, (1 - alpha/2) * 100)
    return lower, upper


def main():
    parser = argparse.ArgumentParser(description='统一评估脚本')
    parser.add_argument('--predictions_dir', type=str, 
                        default='research/predictions',
                        help='预测文件所在目录')
    parser.add_argument('--output_dir', type=str, 
                        default='research/figures',
                        help='输出图表目录')
    parser.add_argument('--genre_mapping', type=str,
                        default='research/data/genre_mapping.csv',
                        help='genre映射文件')
    args = parser.parse_args()

    pred_dir = Path(args.predictions_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读取所有预测文件
    all_files = list(pred_dir.glob('*.csv'))
    if not all_files:
        print(f"错误：在 {pred_dir} 下没有找到CSV文件！")
        print("请确认预测文件已放入该目录。")
        sys.exit(1)

    print(f"找到 {len(all_files)} 个预测文件")

    # 读取并合并
    dfs = []
    for f in all_files:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"  加载: {f.name}, 行数: {len(df)}")

    all_preds = pd.concat(dfs, ignore_index=True)

    # 检查必要列
    required_cols = ['movie_id', 'model', 'variant', 'seed', 'genre_id', 'y_true', 'y_score', 'y_pred']
    missing = [c for c in required_cols if c not in all_preds.columns]
    if missing:
        print(f"错误：预测文件缺少列: {missing}")
        sys.exit(1)

    # 读取genre映射
    genre_names = {}
    if Path(args.genre_mapping).exists():
        genre_map = pd.read_csv(args.genre_mapping)
        if 'genre_id' in genre_map.columns and 'genre_name' in genre_map.columns:
            genre_names = dict(zip(genre_map['genre_id'], genre_map['genre_name']))
        print(f"加载genre映射: {len(genre_names)} 个genre")
    else:
        print(f"警告：未找到genre映射文件 {args.genre_mapping}")

    # 按 (model, variant) 分组评估
    results = []
    for (model, variant), group in all_preds.groupby(['model', 'variant']):
        metrics = compute_metrics(group)
        results.append({
            'model': model,
            'variant': variant,
            'micro_f1': metrics['micro_f1'],
            'macro_f1': metrics['macro_f1'],
            'n_samples': metrics['n_samples'],
            'n_genres': metrics['n_genres']
        })
        print(f"\n{model} | {variant}:")
        print(f"  Micro-F1: {metrics['micro_f1']:.4f}")
        print(f"  Macro-F1: {metrics['macro_f1']:.4f}")
        print(f"  样本数: {metrics['n_samples']}, Genre数: {metrics['n_genres']}")

    results_df = pd.DataFrame(results)

    # 计算性能下降（相对于original）
    print("\n" + "="*60)
    print("性能下降分析（相对于 original）")
    print("="*60)

    drop_results = []
    for model in results_df['model'].unique():
        model_df = results_df[results_df['model'] == model]
        original_row = model_df[model_df['variant'] == 'original']
        if len(original_row) == 0:
            print(f"警告：{model} 缺少 original 结果，跳过下降计算")
            continue

        orig_macro = original_row['macro_f1'].values[0]
        orig_micro = original_row['micro_f1'].values[0]

        for _, row in model_df.iterrows():
            if row['variant'] == 'original':
                continue
            macro_drop = compute_drop(orig_macro, row['macro_f1'])
            micro_drop = compute_drop(orig_micro, row['micro_f1'])
            drop_results.append({
                'model': model,
                'variant': row['variant'],
                'macro_f1_drop_pct': macro_drop,
                'micro_f1_drop_pct': micro_drop,
                'macro_f1': row['macro_f1'],
                'micro_f1': row['micro_f1']
            })
            print(f"{model} | {row['variant']}: Macro-F1下降 {macro_drop:.2f}%")

    # 保存汇总结果
    summary_path = out_dir / 'summary.csv'
    results_df.to_csv(summary_path, index=False)
    print(f"\n汇总结果已保存: {summary_path}")

    if drop_results:
        drop_df = pd.DataFrame(drop_results)
        drop_path = out_dir / 'drop_summary.csv'
        drop_df.to_csv(drop_path, index=False)
        print(f"下降分析已保存: {drop_path}")

    # 保存per-genre详细结果
    per_genre_results = []
    for (model, variant), group in all_preds.groupby(['model', 'variant']):
        genres = sorted(group['genre_id'].unique())
        metrics = compute_metrics(group)
        for i, genre in enumerate(genres):
            genre_name = genre_names.get(genre, f"genre_{genre}")
            per_genre_results.append({
                'model': model,
                'variant': variant,
                'genre_id': genre,
                'genre_name': genre_name,
                'f1': metrics['per_genre_f1'][i],
                'precision': metrics['per_genre_precision'][i],
                'recall': metrics['per_genre_recall'][i]
            })

    per_genre_df = pd.DataFrame(per_genre_results)
    per_genre_path = out_dir / 'per_genre_metrics.csv'
    per_genre_df.to_csv(per_genre_path, index=False)
    print(f"Per-genre结果已保存: {per_genre_path}")

    print("\n评估完成！请运行 visualize.py 生成图表。")


if __name__ == '__main__':
    main()
