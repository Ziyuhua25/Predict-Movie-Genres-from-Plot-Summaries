"""
visualize.py
5号负责：生成4种最终图表

用法：
    python research/src/visualize.py --summary research/figures/summary.csv \
        --drop_summary research/figures/drop_summary.csv \
        --per_genre research/figures/per_genre_metrics.csv \
        --output_dir research/figures
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置样式
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.dpi'] = 150

# 统一配色
COLORS = {
    'tfidf_lr': '#E74C3C',      # 红色
    'sbert_lr': '#3498DB',       # 蓝色
    'original': '#2ECC71',      # 绿色
    'shuffled': '#E67E22',       # 橙色
    'masked': '#9B59B6',         # 紫色
    'first_25': '#F39C12',       # 黄色
    'first_50': '#1ABC9C',      # 青色
    'first_75': '#34495E',      # 深蓝灰
}


def plot_overall_comparison(summary_df, out_dir):
    """
    图1: 整体对比图（TF-IDF vs SBERT，6种输入）
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    variants = ['original', 'shuffled', 'masked', 'first_25', 'first_50', 'first_75']
    summary_df['variant'] = pd.Categorical(summary_df['variant'], categories=variants, ordered=True)
    summary_df = summary_df.sort_values('variant')

    # Macro-F1 对比
    ax1 = axes[0]
    x = np.arange(len(variants))
    width = 0.35

    for i, model in enumerate(sorted(summary_df['model'].unique())):
        model_df = summary_df[summary_df['model'] == model].sort_values('variant')
        color = COLORS.get(model, '#333333')
        ax1.bar(x + i*width, model_df['macro_f1'], width, 
                alpha=0.85, label=model, color=color, edgecolor='white', linewidth=0.5)

    ax1.set_ylabel('Macro-F1', fontsize=12)
    ax1.set_title('Macro-F1: TF-IDF vs Sentence-BERT', fontsize=13, fontweight='bold')
    ax1.set_xticks(x + width/2)
    ax1.set_xticklabels(variants, rotation=30, ha='right')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=summary_df[summary_df['variant']=='original']['macro_f1'].mean(), 
                color='gray', linestyle='--', alpha=0.5, label='avg original')

    # Micro-F1 对比
    ax2 = axes[1]
    for i, model in enumerate(sorted(summary_df['model'].unique())):
        model_df = summary_df[summary_df['model'] == model].sort_values('variant')
        color = COLORS.get(model, '#333333')
        ax2.bar(x + i*width, model_df['micro_f1'], width, 
                alpha=0.85, label=model, color=color, edgecolor='white', linewidth=0.5)

    ax2.set_ylabel('Micro-F1', fontsize=12)
    ax2.set_title('Micro-F1: TF-IDF vs Sentence-BERT', fontsize=13, fontweight='bold')
    ax2.set_xticks(x + width/2)
    ax2.set_xticklabels(variants, rotation=30, ha='right')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / 'fig1_overall_comparison.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"保存: {out_path}")
    plt.close()


def plot_perturbation_drop(drop_df, out_dir):
    """
    图2: 扰动下降图（相对于original的性能下降）
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    variants = ['shuffled', 'masked', 'first_25', 'first_50', 'first_75']
    drop_df['variant'] = pd.Categorical(drop_df['variant'], categories=variants, ordered=True)
    drop_df = drop_df.sort_values('variant')

    x = np.arange(len(variants))
    width = 0.35

    for i, model in enumerate(sorted(drop_df['model'].unique())):
        model_df = drop_df[drop_df['model'] == model].sort_values('variant')
        color = COLORS.get(model, '#333333')
        bars = ax.bar(x + i*width, model_df['macro_f1_drop_pct'], width, 
                      label=model, color=color, alpha=0.85, edgecolor='white', linewidth=0.5)
        # 在柱子上标注数值
        for bar, val in zip(bars, model_df['macro_f1_drop_pct']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Macro-F1 Drop (%)', fontsize=12)
    ax.set_title('Performance Drop Under Input Perturbations', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(variants, rotation=30, ha='right')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / 'fig2_perturbation_drop.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"保存: {out_path}")
    plt.close()


def plot_length_curve(summary_df, out_dir):
    """
    图3: 文本长度曲线（first_25/50/75 vs original）
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    length_variants = ['first_25', 'first_50', 'first_75', 'original']
    length_labels = ['25%', '50%', '75%', '100%']

    for model in sorted(summary_df['model'].unique()):
        model_df = summary_df[summary_df['model'] == model]
        color = COLORS.get(model, '#333333')
        f1_values = []
        for v in length_variants:
            row = model_df[model_df['variant'] == v]
            if len(row) > 0:
                f1_values.append(row['macro_f1'].values[0])
            else:
                f1_values.append(np.nan)

        ax.plot(length_labels, f1_values, marker='o', linewidth=2.5, 
                markersize=10, label=model, color=color, markeredgecolor='white', markeredgewidth=1.5)

    ax.set_xlabel('Text Length Retained', fontsize=12)
    ax.set_ylabel('Macro-F1', fontsize=12)
    ax.set_title('Effect of Text Truncation on Genre Classification', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / 'fig3_length_curve.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"保存: {out_path}")
    plt.close()


def plot_per_genre_heatmap(per_genre_df, out_dir):
    """
    图4: per-genre 热力图
    """
    # 只选 original 做展示
    pivot_df = per_genre_df[per_genre_df['variant'] == 'original'].copy()

    if len(pivot_df) == 0:
        print("跳过per-genre热力图（缺少original数据）")
        return

    # 创建 genre x model 的矩阵
    if 'genre_name' in pivot_df.columns and pivot_df['genre_name'].notna().any():
        idx_col = 'genre_name'
    else:
        idx_col = 'genre_id'

    heatmap_data = pivot_df.pivot_table(
        index=idx_col, 
        columns='model', 
        values='f1'
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'F1 Score'},
                linewidths=0.5, linecolor='white')
    ax.set_title('Per-Genre F1 Score (Original Input)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Genre', fontsize=12)

    plt.tight_layout()
    out_path = out_dir / 'fig4_per_genre_heatmap.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"保存: {out_path}")
    plt.close()


def plot_per_genre_drop(per_genre_df, out_dir):
    """
    图5: per-genre 性能下降热力图（shuffled vs original）
    """
    orig = per_genre_df[per_genre_df['variant'] == 'original'][['model', 'genre_id', 'genre_name', 'f1']].rename(columns={'f1': 'f1_orig'})
    shuf = per_genre_df[per_genre_df['variant'] == 'shuffled'][['model', 'genre_id', 'f1']].rename(columns={'f1': 'f1_shuf'})

    if len(orig) == 0 or len(shuf) == 0:
        print("跳过per-genre下降图（缺少shuffled数据）")
        return

    merged = orig.merge(shuf, on=['model', 'genre_id'])
    merged['drop'] = (merged['f1_orig'] - merged['f1_shuf']) / merged['f1_orig'] * 100
    merged['drop'] = merged['drop'].fillna(0)

    if 'genre_name' in merged.columns and merged['genre_name'].notna().any():
        idx_col = 'genre_name'
    else:
        idx_col = 'genre_id'

    heatmap_data = merged.pivot_table(index=idx_col, columns='model', values='drop')

    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='Reds', 
                vmin=0, ax=ax, cbar_kws={'label': 'F1 Drop (%)'},
                linewidths=0.5, linecolor='white')
    ax.set_title('Per-Genre F1 Drop: Shuffled vs Original', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Genre', fontsize=12)

    plt.tight_layout()
    out_path = out_dir / 'fig5_per_genre_drop_heatmap.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"保存: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='可视化脚本')
    parser.add_argument('--summary', type=str, 
                        default='research/figures/summary.csv',
                        help='summary.csv路径')
    parser.add_argument('--drop_summary', type=str,
                        default='research/figures/drop_summary.csv',
                        help='drop_summary.csv路径')
    parser.add_argument('--per_genre', type=str,
                        default='research/figures/per_genre_metrics.csv',
                        help='per_genre_metrics.csv路径')
    parser.add_argument('--output_dir', type=str,
                        default='research/figures',
                        help='输出目录')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读取数据
    if not Path(args.summary).exists():
        print(f"错误：找不到 {args.summary}")
        print("请先运行 evaluate.py 生成汇总结果！")
        return

    summary_df = pd.read_csv(args.summary)
    print(f"加载汇总结果: {len(summary_df)} 行")

    # 图1: 整体对比
    plot_overall_comparison(summary_df, out_dir)

    # 图2: 扰动下降
    if Path(args.drop_summary).exists():
        drop_df = pd.read_csv(args.drop_summary)
        plot_perturbation_drop(drop_df, out_dir)
    else:
        print("跳过扰动下降图（缺少drop_summary.csv）")

    # 图3: 长度曲线
    plot_length_curve(summary_df, out_dir)

    # 图4: per-genre热力图
    if Path(args.per_genre).exists():
        per_genre_df = pd.read_csv(args.per_genre)
        plot_per_genre_heatmap(per_genre_df, out_dir)
        plot_per_genre_drop(per_genre_df, out_dir)
    else:
        print("跳过per-genre热力图（缺少per_genre_metrics.csv）")

    print(f"\n所有图表已保存到: {out_dir}")


if __name__ == '__main__':
    main()
