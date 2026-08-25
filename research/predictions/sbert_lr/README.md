# SBERT Predictions

- `by_variant/`: one compact file per original or perturbed test variant
- `sbert_all_variants_predictions_compact.csv`: one row per movie and variant,
  with predicted genre IDs and names
- `sbert_all_variants_predictions_long.csv`: one row per movie, genre, and
  variant, including `y_true`, `y_score`, and `y_pred`

The long file follows the shared evaluation schema and contains 215,640 rows:
1,198 movies × 18 genres × 10 variants.
