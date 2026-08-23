"""
dummy_prediction_generator.py
生成模拟预测数据，用于测试 evaluate.py 和 visualize.py
不依赖真实模型预测，格式与统一接口一致。
"""

import pandas as pd
import numpy as np
import ast
from pathlib import Path

# 配置
SEED = 42
np.random.seed(SEED)

# 路径
DATA_DIR = Path("research/data")
PRED_DIR = Path("research/predictions")
PRED_DIR.mkdir(parents=True, exist_ok=True)

# 加载数据
df = pd.read_csv(DATA_DIR / "cleaned_train.csv")
df['genre_ids'] = df['genre_ids'].apply(ast.literal_eval)

genre_mapping = pd.read_csv(DATA_DIR / "genre_mapping.csv")
valid_genre_ids = sorted(genre_mapping['genre_id'].values)

splits = pd.read_csv(DATA_DIR / "splits.csv")
test_ids = set(splits[splits['split'] == 'test']['movie_id'].values)
test_df = df[df['movie_id'].isin(test_ids)].copy()

# 为每个 movie 生成模拟预测
variants = ['original', 'shuffled', 'masked', 'first_25', 'first_50', 'first_75']
models = ['tfidf_lr', 'sbert_lr']

for model in models:
    for variant in variants:
        records = []
        for _, row in test_df.iterrows():
            mid = row['movie_id']
            true_genres = set(row['genre_ids'])

            for gid in valid_genre_ids:
                y_true = 1 if gid in true_genres else 0

                # 模拟预测：original 最准，perturbation 越重越差
                base_score = np.random.beta(2, 2)  # 基础随机分数

                if variant == 'original':
                    noise = np.random.normal(0, 0.1)
                elif variant == 'shuffled':
                    noise = np.random.normal(-0.05, 0.15)
                elif variant == 'masked':
                    noise = np.random.normal(-0.08, 0.15)
                elif variant == 'first_75':
                    noise = np.random.normal(-0.02, 0.12)
                elif variant == 'first_50':
                    noise = np.random.normal(-0.06, 0.14)
                elif variant == 'first_25':
                    noise = np.random.normal(-0.12, 0.18)

                y_score = np.clip(base_score + noise, 0.001, 0.999)

                # 用 0.25 阈值生成预测
                y_pred = 1 if y_score >= 0.25 else 0

                records.append({
                    'movie_id': mid,
                    'model': model,
                    'variant': variant,
                    'seed': SEED,
                    'genre_id': gid,
                    'y_true': y_true,
                    'y_score': round(y_score, 6),
                    'y_pred': y_pred
                })

        pred_df = pd.DataFrame(records)
        output_path = PRED_DIR / f"{model}_{variant}_predictions.csv"
        pred_df.to_csv(output_path, index=False)
        print(f"Generated: {output_path} ({len(pred_df)} rows)")

print("\nAll dummy predictions generated!")
