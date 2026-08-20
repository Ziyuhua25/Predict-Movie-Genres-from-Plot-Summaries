import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# =========================
# 配置：定义要测试的输入版本
# =========================

def clean_text(text, stem=False):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        if stem:
            text = ' '.join([w.rstrip('ing').rstrip('ed').rstrip('s') for w in text.split()])
        return text
    return ""

def build_input_text(df, columns, clean=True, stem=False):
    texts = []
    for col in columns:
        col_text = df[col].fillna("")
        if clean:
            col_text = col_text.apply(clean_text, stem=stem)
        texts.append(col_text)
    return pd.Series([" ".join(t) for t in zip(*texts)])

def run_experiment(train_df, test_df, input_columns, version_name, clean=True, stem=False):
    print(f"\n{'='*60}")
    print(f"运行版本: {version_name}")
    print(f"输入列: {input_columns}")
    print(f"清洗: {clean}, 词干: {stem}")
    print('='*60)

    train_input = build_input_text(train_df, input_columns, clean=clean, stem=stem)
    test_input = build_input_text(test_df, input_columns, clean=clean, stem=stem)

    def parse_genres(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            x = x.replace("[", "").replace("]", "").replace(",", " ")
            return [int(i) for i in x.split() if i.strip()]
        return []

    train_df['genre_list'] = train_df['genre_ids'].apply(parse_genres)
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(train_df['genre_list'])
    genre_ids = mlb.classes_

    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=10000, min_df=2, max_df=0.8, stop_words='english')
    X = tfidf.fit_transform(train_input)
    print(f"   TF-IDF 矩阵: {X.shape}")

    X_train, X_val, y_train, y_val, train_idx, val_idx = train_test_split(
        X, y, train_df.index, test_size=0.2, random_state=42
    )

    model = OneVsRestClassifier(LogisticRegression(max_iter=1000, solver='liblinear'))
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)
    best_threshold = 0.5
    best_f1 = 0
    for th in [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        pred = (val_prob >= th).astype(int)
        score = f1_score(y_val, pred, average='macro')
        if score > best_f1:
            best_f1 = score
            best_threshold = th

    y_val_pred = (val_prob >= best_threshold).astype(int)
    val_f1 = f1_score(y_val, y_val_pred, average='macro')
    print(f"   最佳阈值: {best_threshold:.2f}, 验证集 F1: {val_f1:.4f}")

    X_test = tfidf.transform(test_input)
    test_prob = model.predict_proba(X_test)
    test_pred = (test_prob >= best_threshold).astype(int)
    predicted_ids = mlb.inverse_transform(test_pred)

    sub = pd.DataFrame({
        "movie_id": test_df["movie_id"],
        "genre_ids": [" ".join(map(str, ids)) for ids in predicted_ids]
    })
    sub.to_csv(f"submission_{version_name}.csv", index=False)

    features = np.array(tfidf.get_feature_names_out())
    keywords = {}
    for i, g in enumerate(genre_ids):
        coef = model.estimators_[i].coef_[0]
        top_idx = np.argsort(coef)[-10:][::-1]
        keywords[g] = [features[j] for j in top_idx if coef[j] > 0]

    return {
        "version": version_name,
        "input_columns": input_columns,
        "clean": clean,
        "stem": stem,
        "val_f1": val_f1,
        "best_threshold": best_threshold,
        "keywords": keywords,
        "val_idx": val_idx,
        "y_val": y_val,
        "y_val_pred": y_val_pred,
        "train_df": train_df,
        "genre_ids": genre_ids,
        "submission_file": f"submission_{version_name}.csv"
    }

# =========================
# 主程序
# =========================

print("="*60)
print("TF-IDF 多版本实验")
print("="*60)

# 加载数据（修改为当前目录）
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 原始文本列（用于不清洗版本）
train['overview_raw'] = train['overview'].fillna("")
test['overview_raw'] = test['overview'].fillna("")
train['title_raw'] = train['title'].fillna("")
test['title_raw'] = test['title'].fillna("")

experiments = [
    {"columns": ["overview"], "name": "V1_overview", "clean": True, "stem": False},
    {"columns": ["title"], "name": "V2_title", "clean": True, "stem": False},
    {"columns": ["title", "overview"], "name": "V3_title+overview", "clean": True, "stem": False},
    {"columns": ["overview_raw"], "name": "V4_overview_raw", "clean": False, "stem": False},
    {"columns": ["title_raw", "overview_raw"], "name": "V5_title+overview_raw", "clean": False, "stem": False},
    {"columns": ["overview"], "name": "V6_overview_stem", "clean": True, "stem": True},
]

results = []
for exp in experiments:
    res = run_experiment(train, test, exp["columns"], exp["name"], clean=exp["clean"], stem=exp["stem"])
    results.append(res)

# 汇总对比结果
print("\n" + "="*60)
print("汇总结果对比")
print("="*60)

summary = pd.DataFrame([{
    "Version": r["version"],
    "Input": " + ".join(r["input_columns"]),
    "Clean": r["clean"],
    "Stem": r["stem"],
    "Val F1": round(r["val_f1"], 4)
} for r in results])

print(summary.to_string(index=False))
summary.to_csv("version_comparison.csv", index=False)

best = max(results, key=lambda x: x["val_f1"])
print(f"\n最佳版本: {best['version']} (F1={best['val_f1']:.4f})")

# =========================
# 对最佳版本输出详细分析
# =========================

print("\n" + "="*60)
print(f"最佳版本 ({best['version']}) 的详细分析")
print("="*60)

val_original = train.loc[best['val_idx'], 'overview'].values

correct_mask = (best['y_val_pred'] == best['y_val']).all(axis=1)
correct_indices = [i for i, flag in enumerate(correct_mask) if flag]

print("\n成功案例（预测完全正确）:")
for i in range(min(3, len(correct_indices))):
    idx = correct_indices[i]
    true_ids = best['genre_ids'][best['y_val'][idx].astype(bool)]
    pred_ids = best['genre_ids'][best['y_val_pred'][idx].astype(bool)]
    print(f"\n样本 {i+1}:")
    print(f"  文本: {val_original[idx][:150]}...")
    print(f"  真实: {', '.join(map(str, true_ids))}")
    print(f"  预测: {', '.join(map(str, pred_ids))}")

wrong_mask = (best['y_val_pred'] * best['y_val']).sum(axis=1) == 0
wrong_indices = [i for i, flag in enumerate(wrong_mask) if flag]

print("\n失败案例（预测完全错误）:")
for i in range(min(3, len(wrong_indices))):
    idx = wrong_indices[i]
    true_ids = best['genre_ids'][best['y_val'][idx].astype(bool)]
    pred_ids = best['genre_ids'][best['y_val_pred'][idx].astype(bool)]
    print(f"\n样本 {i+1}:")
    print(f"  文本: {val_original[idx][:150]}...")
    print(f"  真实: {', '.join(map(str, true_ids))}")
    print(f"  预测: {', '.join(map(str, pred_ids))}")

print("\n各类型 Top 关键词（最佳版本）:")
for genre, words in best['keywords'].items():
    print(f"  {genre}: {', '.join(words[:5])}")

print("\n全部完成！生成文件:")
print("  - version_comparison.csv")
print("  - submission_*.csv (各版本)")