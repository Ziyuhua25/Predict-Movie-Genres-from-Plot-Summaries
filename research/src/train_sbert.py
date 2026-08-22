"""
Sentence-BERT semantic model.

Pipeline:
1. Load official split data
2. Generate SBERT embeddings
3. Train One-vs-Rest Logistic Regression
4. Tune validation threshold
5. Generate predictions
"""

from pathlib import Path
import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from sentence_transformers import SentenceTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
ROOT_DIR = Path(__file__).resolve().parents[2]
from sklearn.metrics import f1_score
import numpy as np

DATA_DIR = ROOT_DIR / "research" / "data"
RAW_DIR = DATA_DIR / "raw"
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

def load_data():

    train_raw = pd.read_csv(
        RAW_DIR / "train.csv"
    )

    splits = pd.read_csv(
        DATA_DIR / "splits.csv"
    )

    genre_mapping = pd.read_csv(
        DATA_DIR / "genre_mapping.csv"
    )

    df = train_raw.merge(
        splits,
        on="movie_id",
        how="inner"
    )

    return df, genre_mapping

def parse_genre_ids(x):

    if isinstance(x, str):
        return ast.literal_eval(x)

    return x

def build_labels(df, genre_mapping):

    df = df.copy()

    df["genre_ids"] = (
        df["genre_ids"]
        .apply(parse_genre_ids)
    )

    genre_ids = (
        genre_mapping["genre_id"]
        .tolist()
    )

    genre_names = (
        genre_mapping["genre_name"]
        .tolist()
    )

    mlb = MultiLabelBinarizer(
        classes=genre_ids
    )

    Y = mlb.fit_transform(
        df["genre_ids"]
    )


    train_mask = (
        df["split"] == "train"
    )

    val_mask = (
        df["split"] == "validation"
    )

    test_mask = (
        df["split"] == "test"
    )


    y_train = Y[train_mask]
    y_val = Y[val_mask]
    y_test = Y[test_mask]


    return (
        y_train,
        y_val,
        y_test,
        genre_ids,
        genre_names
    )

def normalize_text(text):

    if pd.isna(text):
        return ""

    return " ".join(
        str(text).split()
    )

def encode_embeddings(df):

    model = SentenceTransformer(
        MODEL_NAME,
        device=DEVICE
    )


    train_texts = (
        df[df["split"] == "train"]["overview"]
        .apply(normalize_text)
        .tolist()
    )

    val_texts = (
        df[df["split"] == "validation"]["overview"]
        .apply(normalize_text)
        .tolist()
    )

    test_texts = (
        df[df["split"] == "test"]["overview"]
        .apply(normalize_text)
        .tolist()
    )


    train_embeddings = model.encode(
        train_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    val_embeddings = model.encode(
        val_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    test_embeddings = model.encode(
        test_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )


    return (
        train_embeddings,
        val_embeddings,
        test_embeddings
    )

def train_classifier(
    train_embeddings,
    y_train
):

    print("Start training classifier...")


    classifier = OneVsRestClassifier(
        LogisticRegression(
            max_iter=2000,
            random_state=42
        )
    )

    classifier.fit(
        train_embeddings,
        y_train
    )

    print("Classifier training finished.")

    return classifier

def tune_threshold(
    classifier,
    val_embeddings,
    y_val
):

    val_scores = classifier.predict_proba(
        val_embeddings
    )

    thresholds = np.arange(
        0.1,
        0.81,
        0.05
    )

    results = []

    for threshold in thresholds:

        preds = (
            val_scores >= threshold
        ).astype(int)

        macro = f1_score(
            y_val,
            preds,
            average="macro",
            zero_division=0
        )

        micro = f1_score(
            y_val,
            preds,
            average="micro",
            zero_division=0
        )

        results.append(
            {
                "threshold": threshold,
                "macro_f1": macro,
                "micro_f1": micro
            }
        )

    results_df = pd.DataFrame(results)

    best_row = results_df.loc[
        results_df["macro_f1"].idxmax()
    ]

    return (
        float(best_row["threshold"]),
        results_df,
        val_scores
    )

def evaluate_test(
    classifier,
    test_embeddings,
    y_test,
    threshold
):

    test_scores = classifier.predict_proba(
        test_embeddings
    )

    test_pred = (
        test_scores >= threshold
    ).astype(int)


    macro_f1 = f1_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0
    )

    micro_f1 = f1_score(
        y_test,
        test_pred,
        average="micro",
        zero_division=0
    )


    return (
        test_scores,
        test_pred,
        macro_f1,
        micro_f1
    )

def save_predictions(
    test_df,
    test_scores,
    test_pred,
    y_test,
    genre_ids
):

    import numpy as np

    test_movie_ids = (
        test_df["movie_id"]
        .to_numpy()
    )

    n_movies = len(test_movie_ids)
    n_genres = len(genre_ids)


    prediction_df = pd.DataFrame({

        "movie_id": np.repeat(
            test_movie_ids,
            n_genres
        ),

        "model": "sbert_lr",

        "variant": "original",

        "seed": 42,

        "genre_id": np.tile(
            genre_ids,
            n_movies
        ),

        "y_true": y_test.reshape(-1),

        "y_score": test_scores.reshape(-1),

        "y_pred": test_pred.reshape(-1)
    })


    output_dir = (
        ROOT_DIR /
        "research" /
        "predictions"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    output_path = (
        output_dir /
        "sbert_original_predictions.csv"
    )


    prediction_df.to_csv(
        output_path,
        index=False
    )


    return output_path, prediction_df

def main():

    df, genre_mapping = load_data()

    print("Dataset:")
    print(df.shape)

    print("\nGenre mapping:")
    print(genre_mapping.shape)


    (
        y_train,
        y_val,
        y_test,
        genre_ids,
        genre_names
    ) = build_labels(
        df,
        genre_mapping
    )


    print("\nLabels:")
    print("y_train:", y_train.shape)
    print("y_val:", y_val.shape)
    print("y_test:", y_test.shape)

    print("\nGenres:")
    print(len(genre_ids))
    (
        train_embeddings,
        val_embeddings,
        test_embeddings
    ) = encode_embeddings(df)


    print("\nEmbeddings:")
    print(
        "train:",
        train_embeddings.shape
    )

    print(
        "val:",
        val_embeddings.shape
    )

    print(
        "test:",
        test_embeddings.shape
    )
    
    classifier = train_classifier(
        train_embeddings,
        y_train
    )

    print("\nClassifier:")
    print(classifier)
        
    (
        best_threshold,
        threshold_results,
        val_scores
    ) = tune_threshold(
        classifier,
        val_embeddings,
        y_val
    )


    print("\nThreshold:")
    print(
        "Best threshold:",
        best_threshold
    )

    print(
        "Best Macro-F1:",
        threshold_results.loc[
            threshold_results["threshold"]
            == best_threshold,
            "macro_f1"
        ].values[0]
    )
    
    (
        test_scores,
        test_pred,
        test_macro,
        test_micro
    ) = evaluate_test(
        classifier,
        test_embeddings,
        y_test,
        best_threshold
    )


    print("\nTest:")
    print("Macro-F1:", test_macro)
    print("Micro-F1:", test_micro)

    output_path, prediction_df = save_predictions(
        df[df["split"] == "test"],
        test_scores,
        test_pred,
        y_test,
        genre_ids
    )

    print("\nPrediction saved:")
    print(output_path)

    print(
        prediction_df.shape
    )
if __name__ == "__main__":
    main()