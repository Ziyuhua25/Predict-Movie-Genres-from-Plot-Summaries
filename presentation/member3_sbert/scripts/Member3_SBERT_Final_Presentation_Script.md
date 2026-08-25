# Member 3 — Final SBERT Presentation Script

Approximate speaking time: 5.5–6.5 minutes

## Opening transition

After examining the lexical baseline, I will now introduce our semantic model and ask a more fundamental question: does the model understand the overall narrative, or does it still rely mainly on surface-level textual cues?

## Slide 1 — Why use a sentence-level encoder?

Traditional BERT produces a contextual representation for every token. This is useful for fine-tuning, but a complete plot still needs pooling or task-specific training before it becomes one vector.

We therefore used Sentence-BERT. Specifically, `all-MiniLM-L6-v2` converts each plot summary into one 384-dimensional embedding that can be reused by a lightweight classifier.

For example, “A detective investigates a mysterious murder” and “A police officer tries to solve a killing” use different words but express very similar meanings. SBERT should represent them as semantically close.

Our hypothesis was that sentence embeddings could capture narrative meaning beyond exact genre keywords.

## Slide 2 — Experimental pipeline

To test this hypothesis, we built a controlled and reproducible pipeline.

First, we normalized whitespace without removing any lexical content. Second, the frozen MiniLM encoder transformed every plot into a 384-dimensional vector. Third, we trained 18 one-versus-rest logistic-regression classifiers, using seed 42. Finally, we selected one global prediction threshold on the validation set.

The example below shows why the threshold matters. With scores of 0.61 for Action, 0.43 for Comedy, and 0.69 for Drama, a threshold of 0.50 misses Comedy. At 0.25, all three labels are recovered.

This is important because the task is both multi-label and imbalanced.

## Slide 3 — Main SBERT results

Threshold tuning produced a substantial improvement.

With the default threshold of 0.50, validation Macro-F1 was 0.442. With the selected threshold of 0.25, it increased to 0.552—an absolute improvement of 0.110 without changing the encoder or the classifier.

On the locked test set, the model achieved 0.548 Macro-F1 and 0.624 Micro-F1. The validation and test Macro-F1 scores were nearly identical, which suggests that the selected threshold was stable rather than overfitted to the validation set.

This result also shows that the decision rule can matter almost as much as the representation itself.

## Slide 4 — Success examples

We then examined individual predictions to understand when the semantic model succeeds.

For *Next Goal Wins*, the plot describes an underdog soccer team changing from perennial losers into winners. The model correctly predicted both Comedy and Drama. It could use the full trajectory of failure, transformation, and victory, even without explicit genre words.

For *Labyrinth*, the model correctly predicted Adventure, Family, and Fantasy. Here, the rescue mission, the labyrinth, and the fantasy setting all reinforce one another.

These cases suggest that performance is strongest when narrative meaning and recognizable genre cues point in the same direction.

## Slide 5 — Failure mechanisms

However, semantic similarity is not the same as genre understanding.

For *History of the World: Part I*, the model recognized historical content but missed the comedic framing. For *We’re No Angels*, it focused on escape, police, and crime-related events, but failed to identify the comedic tone. For *Brahms: The Boy II*, family-related cues distracted it from the horror framing.

These errors suggest that SBERT captures what happens in the plot more reliably than how the story is framed.

This observation motivated our next experiment. We tested the same trained model under three controlled perturbations: local word-order shuffling, genre-keyword masking, and text truncation.

## Slide 6 — Robustness results

The robustness results reveal three different patterns. All tests used the same 1,198 movies, the same frozen encoder, the same classifier, and the same threshold. We did not retrain the model for any perturbation.

First, local word-order shuffling had only a small effect. Macro-F1 remained between 0.528 and 0.538, compared with the original score of 0.548. Even the largest observed decrease was only 0.019.

Second, keyword masking caused a clearer and nearly monotonic decline. Macro-F1 fell to 0.479 with 10 percent masking, 0.466 with 30 percent, and 0.451 with 50 percent.

Third, truncation produced the largest loss. Keeping 75 percent of the plot gave 0.519, keeping 50 percent gave 0.491, and keeping only 25 percent reduced Macro-F1 to 0.420—a decrease of 0.128.

So, word order had relatively little influence, while genre cues and overall plot coverage were much more important.

## Slide 7 — Mechanistic interpretation

Together, these interventions give us a more precise interpretation of the model.

At the task level, SBERT behaves like a semantic bag of contextual cues. It is highly robust to local reordering, suggesting low sensitivity to exact word order. However, it depends more strongly on distributed lexical evidence, and it becomes especially vulnerable when most of the narrative is removed.

Our conclusion is not that SBERT has no linguistic understanding. Rather, sentence-level semantics alone do not guarantee a complete understanding of genre, particularly when genre depends on tone, framing, or implicit conventions.

Therefore, the main value of this experiment is not only the final score. It shows which types of information the semantic model actually uses and where its understanding remains limited.

## Closing transition

To summarize, SBERT provides a strong and reproducible semantic baseline, threshold tuning substantially improves multi-label prediction, and the perturbation tests reveal that the model is more robust to word order than to the removal of genre cues or narrative coverage. I will now hand over to the next section for the overall model comparison and final conclusions.
