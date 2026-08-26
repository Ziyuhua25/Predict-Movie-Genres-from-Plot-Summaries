# Official six-input SBERT results

- `sbert_official_six_metrics.csv`: aggregate metrics and changes from original
- `sbert_official_six_per_genre_metrics.csv`: precision, recall, F1, and support
  for every genre under every condition
- `verification.json`: input dimensions, fixed-model settings, and reproduction
  checks

The regenerated original predictions have zero label mismatches against the
earlier original export. The maximum score difference is below `3e-8`.
