# Saved SBERT + LR model

- `sbert_lr_model.pkl`: 18 fitted One-vs-Rest Logistic Regression heads
- `mlb.pkl`: fitted multi-label encoder and label order
- `model_config.json`: encoder, threshold, seed, preprocessing, and validation metrics
- `genre_mapping.csv`: genre IDs and names in model order

These artifacts are loaded unchanged by
`research/src/predict_sbert_official_six.py`. Do not refit the model on any of
the perturbed official inputs.
