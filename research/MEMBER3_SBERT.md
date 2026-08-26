# Member 3: Sentence-BERT Semantic Model

This workstream asks a mechanism-oriented question: does SBERT classify movie
genres from narrative meaning, or does it mainly depend on lexical cues and the
amount of plot evidence available?

## Fixed model

- Frozen encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Representation: one 384-dimensional embedding per plot summary
- Classifier: 18 One-vs-Rest Logistic Regression heads
- Seed: `42`
- Validation-selected global threshold: `0.25`
- Preprocessing: collapse whitespace only; do not delete lexical content

One-vs-Rest is suitable because a movie can have several genres. Each head
independently estimates whether one genre is present, and the threshold converts
all 18 probabilities into a multi-label prediction.

## Main performance

| Evaluation | Macro-F1 | Micro-F1 |
|---|---:|---:|
| Validation, threshold 0.25 | 0.552 | 0.625 |
| Official original test input | 0.548 | 0.624 |

The default validation threshold of 0.50 produced Macro-F1 0.442. Selecting
0.25 improved it by 0.110 without changing the representation or classifier.

## Official six-input evaluation

All conditions contain the same 1,198 movies. The already-trained encoder,
classifier, preprocessing, label order, and threshold remain fixed; the model is
never refit on a perturbed input.

| Condition | Intervention | Macro-F1 | Micro-F1 | Macro change |
|---|---|---:|---:|---:|
| Original | Unchanged plot | 0.548 | 0.624 | 0.000 |
| Shuffled | All words shuffled within local 10-word windows | 0.535 | 0.623 | -0.012 |
| Masked | Official highest-weight 50% TF-IDF cue masking | 0.451 | 0.572 | -0.097 |
| First 75% | Retain the first 75% of plot words | 0.519 | 0.606 | -0.029 |
| First 50% | Retain the first 50% of plot words | 0.491 | 0.579 | -0.057 |
| First 25% | Retain the first 25% of plot words | 0.420 | 0.526 | -0.128 |

## Mechanistic interpretation

Full local shuffling changes Macro-F1 by only -0.012, while official cue
masking changes it by -0.097 and severe truncation by -0.128. At this task
level, SBERT is therefore much less dependent on exact local word order than on
distributed label-related evidence and sufficient narrative coverage.

The cautious conclusion is that SBERT behaves like a semantic bag of
contextual cues for this task. This is perturbation evidence about model
behavior, not proof of complete language or genre understanding.

## Reproduce the official evaluation

Install `research/requirements-sbert.txt`, then run:

```bash
python research/src/predict_sbert_official_six.py \
  --input-dir research/data/official_six_inputs \
  --model-dir research/models/sbert_lr \
  --output-dir PATH/TO/output
```

The runner validates all six inputs and loads the saved model; it does not train
or tune anything. Aggregate metrics are in `research/results/sbert_lr/`, and
movie-level/label-level predictions are in `research/predictions/sbert_lr/`.
