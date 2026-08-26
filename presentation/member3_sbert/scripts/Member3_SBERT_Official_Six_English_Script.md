# Member 3 — Official Six-Input English Script

Approximate speaking time: 5.5–6.5 minutes.

## Opening

After the lexical baseline, I will introduce our semantic model and ask a more
fundamental question: does SBERT understand the narrative, or does it still
depend mainly on surface textual cues?

## Slide 1 — Why Sentence-BERT?

Standard BERT produces one contextual representation for each token, so a full
plot still needs pooling or task-specific fine-tuning to become one vector. We
used `all-MiniLM-L6-v2`, a Sentence-BERT model that maps each plot summary to a
384-dimensional embedding.

For example, “A detective investigates a mysterious murder” and “A police
officer tries to solve a killing” use different words but express similar
meaning. Our hypothesis is that sentence embeddings should capture narrative
meaning beyond exact genre keywords.

## Slide 2 — Pipeline and One-vs-Rest

We collapse whitespace but do not remove lexical content. The frozen MiniLM
encoder creates one vector per plot, followed by 18 One-vs-Rest Logistic
Regression heads with seed 42.

We chose One-vs-Rest because this is a multi-label problem: one movie may be
Action, Comedy, and Drama at the same time. Each head answers one independent
question—whether its genre is present—and all positive heads can be returned.

We selected one global threshold on validation. If the three probabilities are
0.61, 0.43, and 0.69, a threshold of 0.50 misses the middle label, but 0.25
recovers all three. This is important under label imbalance.

## Slide 3 — Main result

At threshold 0.50, validation Macro-F1 was 0.442. At the selected threshold of
0.25, it rose to 0.552, an absolute improvement of 0.110 without changing the
encoder or classifier.

On the official original test input, the result was 0.548 Macro-F1 and 0.624
Micro-F1. The validation and test scores are close, suggesting that the
threshold choice is stable. It also shows that the decision rule matters, not
only the representation.

## Slide 4 — One success case

For *Next Goal Wins*, the plot describes an underdog soccer team changing from
perennial losers into winners. The model correctly predicts Comedy and Drama.
It can use the narrative trajectory—failure, transformation, and victory—even
without explicit genre words. Success is strongest when narrative meaning and
recognizable genre cues support the same prediction.

## Slide 5 — One failure case

For *History of the World: Part I*, the model detects historical content but
misses the comedic framing, predicting Drama and History instead of Comedy.
This suggests that semantic similarity is not identical to genre understanding:
SBERT may capture what happens more reliably than the tone in which it happens.

This failure motivates the controlled intervention study.

## Slide 6 — Official six-input robustness results

We use the same 1,198 movies under six official input conditions. The trained
encoder, classifier, preprocessing, label order, and threshold are fixed. We do
not refit anything on perturbed text.

The original Macro-F1 is 0.548. After all words are shuffled within local
10-word windows, it remains 0.535, a decrease of only 0.012. Under the official
50-percent masking of the strongest label-specific TF-IDF cue occurrences, it
falls to 0.451, a decrease of 0.097.

Truncation produces a graded decline: keeping the first 75 percent gives 0.519,
keeping 50 percent gives 0.491, and keeping only 25 percent gives 0.420. The
largest decrease is therefore 0.128.

## Slide 7 — Mechanistic interpretation

The interventions reveal an asymmetry. The model is relatively insensitive to
exact local word order, but it depends strongly on distributed lexical evidence
and sufficient plot coverage.

For this task, SBERT behaves like a semantic bag of contextual cues. This does
not mean that it has no semantic ability, and it is not proof of complete
language understanding. It is a cautious, task-level inference from fixed-model
perturbation evidence.

The key contribution is therefore not only a classification score. We identify
which textual signals support the prediction and where the model's apparent
understanding remains limited.

## Closing

In summary, SBERT provides a strong semantic baseline, threshold selection
substantially improves multi-label prediction, and the official interventions
show much higher sensitivity to cue removal and narrative coverage than to local
word order. I will now hand over to the final comparison and conclusion.
