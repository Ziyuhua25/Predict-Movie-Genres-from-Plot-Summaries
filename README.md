# Movie Genre Research

Course research project based on the Kaggle dataset **Predict Movie Genres from Plot Summaries**.

## Research question

**Keywords or Semantics? What Do Movie Genre Classifiers Learn from Plot Summaries?**

We compare lexical and semantic text representations, then use controlled input perturbations to study whether genre classifiers rely mainly on keywords, word order, or broader plot semantics.

## Parallel workstreams

1. Data audit, shared splits, and baselines
2. TF-IDF lexical model and keyword analysis
3. Sentence-BERT semantic model
4. Input perturbations: word shuffling, keyword masking, and truncation
5. Evaluation, statistics, visualization, and error analysis

## Planned repository structure

```text
data/           local data and shared split IDs (raw data is not committed)
src/            reusable data, model, perturbation, and evaluation code
notebooks/      exploratory and model notebooks
predictions/    standardized model outputs
figures/        presentation-ready figures
presentation/   slides and speaker notes
```

## Dataset

[Predict Movie Genres from Plot Summaries](https://www.kaggle.com/competitions/predict-movie-genres-from-plot-summaries/overview)

