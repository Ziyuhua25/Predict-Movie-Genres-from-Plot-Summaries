# Member 3: Sentence-BERT Semantic Model

This workstream studies whether multi-label movie-genre predictions are driven
by sentence-level semantics or by surface lexical cues.

## Technical stack

- Python: NumPy, Pandas, scikit-learn
- PyTorch and Sentence-Transformers
- Git and GitHub collaboration
- Sentence-BERT text representations
- One-vs-Rest Logistic Regression
- Data visualization and experimental result communication

## Model

- Frozen encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Representation: one 384-dimensional embedding per plot summary
- Classifier: 18 One-vs-Rest Logistic Regression heads
- Seed: `42`
- Global threshold selected on validation: `0.25`
- Text preprocessing: collapse whitespace only; no lexical deletion

## Verified results

| Evaluation | Macro-F1 | Micro-F1 |
|---|---:|---:|
| Validation, threshold 0.25 | 0.552 | 0.625 |
| Fixed test set, threshold 0.25 | 0.548 | 0.624 |

The default validation threshold of 0.50 produced Macro-F1 0.442. Selecting
0.25 increased validation Macro-F1 by 0.110 without changing the encoder or
classifier.

## Perturbation results

All variants use the same 1,198 test movies and the same frozen model.

| Variant | Macro-F1 | Change vs. original |
|---|---:|---:|
| Original | 0.548 | 0.000 |
| Shuffle 25% | 0.538 | -0.010 |
| Shuffle 50% | 0.528 | -0.019 |
| Shuffle 100% | 0.535 | -0.012 |
| Mask 10% | 0.479 | -0.068 |
| Mask 30% | 0.466 | -0.082 |
| Mask 50% | 0.451 | -0.097 |
| Keep 75% of the plot | 0.519 | -0.029 |
| Keep 50% of the plot | 0.491 | -0.057 |
| Keep 25% of the plot | 0.420 | -0.128 |

## Interpretation

The model is relatively robust to local word-order changes, but it is more
sensitive to the removal of genre-related cues and plot coverage. The evidence
supports a limited task-level interpretation: SBERT captures what happens in a
plot more reliably than how the story is framed. It should not be treated as
proof of complete semantic or genre understanding.

## Repository locations

- Training pipeline: `research/src/train_sbert.py`
- Perturbation prediction pipeline: `research/src/run_sbert_perturbations.py`
- Saved classifier and label encoder: `research/models/sbert_lr/`
- Predictions: `research/predictions/sbert_lr/`
- Metrics and verification: `research/results/sbert_lr/`
- Slides, figures, and scripts: `presentation/member3_sbert/`

## Reproduction

Install the packages listed in `research/requirements-sbert.txt`. The
perturbation runner takes explicit input paths:

```bash
python research/src/run_sbert_perturbations.py \
  --train-csv PATH/TO/train.csv \
  --splits-csv research/data/splits.csv \
  --genre-mapping-csv research/data/genre_mapping.csv \
  --perturbation-dir PATH/TO/perturbation/data \
  --output-dir PATH/TO/output
```

The saved configuration in `research/models/sbert_lr/model_config.json` records
the encoder, threshold, seed, preprocessing, label order, and validation metrics.
