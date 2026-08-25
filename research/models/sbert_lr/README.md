# Saved SBERT + LR Model

- `sbert_lr_model.pkl`: trained 18-head One-vs-Rest Logistic Regression model
- `mlb.pkl`: fitted `MultiLabelBinarizer` and model class order
- `model_config.json`: encoder, preprocessing, threshold, seed, and validation metrics
- `genre_mapping.csv`: genre IDs and names in the project mapping

The Sentence-BERT encoder weights are not duplicated in this repository. Load
`sentence-transformers/all-MiniLM-L6-v2` through Sentence-Transformers, then use
the saved linear classifier and threshold recorded in the configuration.
