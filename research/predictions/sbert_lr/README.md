# Official six-input SBERT predictions

- `by_variant_long/`: one evaluator-ready prediction file for each official
  condition; each has 21,564 rows (1,198 movies × 18 genres)
- `by_variant_compact/`: one readable movie-level file per condition
- `sbert_official_six_predictions_long.csv`: all six long-format files combined,
  with 129,384 rows
- `sbert_official_six_predictions_compact.csv`: all six compact files combined,
  with 7,188 rows

The evaluator-ready schema includes `movie_id`, `model`, `variant`, `seed`,
`genre_id`, `y_true`, `y_score`, and `y_pred`.
