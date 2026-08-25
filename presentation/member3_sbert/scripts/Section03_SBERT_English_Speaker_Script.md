# Section 03 — Sentence-BERT Semantic Model

Estimated speaking time: **4–5 minutes**
Role in the full presentation: **third section, after the lexical/TF-IDF baseline and before robustness or perturbation analysis**

## Transition from the previous speaker

> The previous section showed what a lexical model can learn from explicit words and phrases. Building on that baseline, my section asks a different question: can a sentence-level encoder capture the broader meaning of a plot summary, rather than relying mainly on exact genre keywords?

## Slide 3.1 — Why use a sentence-level encoder?

> Standard BERT produces contextual representations for individual tokens. That is powerful, but a complete movie plot still needs pooling or task-specific fine-tuning before it becomes a single representation.
>
> Sentence-BERT is designed for this purpose. In our experiment, `all-MiniLM-L6-v2` converts each plot summary into one 384-dimensional embedding. We then reuse that embedding with a lightweight classifier.
>
> The small example on the right shows the basic idea. “A detective investigates a mysterious murder” and “A police officer tries to solve a killing” use different words, but their meanings are very similar. A sentence embedding should place them close together.
>
> However, movie genre classification is harder than ordinary semantic similarity. A text can be about history without being a History movie, and it can describe a dangerous event while presenting it as a comedy. This distinction becomes important in our case analysis.

## Slide 3.2 — A controlled and reproducible semantic baseline

> Our pipeline has four stages. First, we apply only whitespace normalization, so we do not remove meaningful words from the plot summaries. Second, we use the frozen MiniLM Sentence-BERT encoder to produce a 384-dimensional vector. Third, we train eighteen one-versus-rest logistic regression classifiers, one for each genre. Finally, we select one global probability threshold on the validation set.
>
> The probability example explains why threshold selection matters. Suppose the model outputs 0.61 for Action, 0.43 for Comedy, and 0.69 for Drama. With the default threshold of 0.5, Comedy is removed even though the score is still meaningful. With a threshold of 0.25, all three labels are retained.
>
> This is a multi-label and imbalanced task, so 0.5 is not a theoretically privileged value. We tune the threshold on validation data and lock it before testing. This avoids using the test set for model selection.

## Slide 3.3 — Official results

> Threshold tuning produced one of the clearest findings in this experiment. On the official validation split, the default threshold of 0.5 gives a Macro-F1 of 0.442. After selecting a threshold of 0.25, Macro-F1 increases to 0.552. That is an absolute improvement of approximately 0.110.
>
> We then locked the threshold and evaluated the unseen test set. The final test Macro-F1 is 0.548, and the Micro-F1 is 0.624.
>
> The validation and test results are almost identical, so this appears to be a stable baseline rather than validation overfitting. The higher Micro-F1 also suggests that common genres are easier to recognise, while rare genres remain more difficult.
>
> At this stage, I would describe Sentence-BERT as a reliable semantic baseline. I would not yet claim that it fully understands genre semantics.

## Slide 3.4 — What does the semantic model get right?

> The first exact-match example is *Next Goal Wins*. The plot describes a football team changing from repeated losers into winners. The model correctly predicts Comedy and Drama. This example is useful because the narrative trajectory itself provides information; the overview does not simply list explicit genre words.
>
> The second exact-match example is *Labyrinth*. Sarah must solve a labyrinth and rescue her brother within thirteen hours. The model exactly predicts Adventure, Family, and Fantasy. Here, the quest structure, family relationship, and fantasy setting all reinforce one another.
>
> These examples suggest that the model works best when broad narrative meaning and strong genre cues point in the same direction.

## Slide 3.5 — What does the model still miss?

> The failure cases are more useful for understanding the mechanism.
>
> First, *History of the World: Part I* mentions the Roman Empire, the French Revolution, and the Spanish Inquisition. The model predicts Drama and History, but the real genre is Comedy. The model captures what the film is about, but topic semantics are not the same as genre semantics.
>
> Second, *We're No Angels* contains escaped convicts, a police blockade, and a border crossing. These events sound like Action or Thriller, which is what the model predicts. But the actual genres are Comedy and Crime. The model recognises the events more easily than the comic tone in which those events are presented.
>
> Finally, the short overview of *Brahms: The Boy II* mainly mentions a family, a young son, and a life-like doll. It contains very little explicit horror evidence. The model predicts Comedy, Drama, and Family instead of Horror, Mystery, and Thriller. This illustrates a basic limitation: an embedding cannot recover information that is absent from the input text.
>
> Therefore, our mechanism-level conclusion is that Sentence-BERT captures what happens in a plot more reliably than how the story is framed. To test whether it truly relies on compositional meaning, the next step is to compare performance after word-order shuffling, genre-keyword masking, and text truncation.

## Transition to the next speaker

> In summary, Sentence-BERT gives us a stable semantic baseline, but the case analysis shows that topic, events, tone, and genre are not interchangeable. This motivates the next section, where we test the models under controlled textual perturbations.

## Short version if time is limited

> We encoded each plot summary with `all-MiniLM-L6-v2`, producing a 384-dimensional Sentence-BERT vector, and trained eighteen one-versus-rest logistic regression classifiers. Validation-based threshold tuning increased Macro-F1 from 0.442 to 0.552, and the locked test result was 0.548 Macro-F1 and 0.624 Micro-F1. Exact-match cases show that SBERT can use narrative structure, but failures reveal that topic is not genre, events are not tone, and missing genre cues cannot be recovered from the input. Therefore, SBERT is a stable semantic baseline, while controlled perturbation tests are still needed to examine what information it truly uses.

## Likely Q&A

### Why did you not fine-tune BERT end to end?

> We wanted a reproducible and computationally manageable semantic baseline. Using a frozen encoder also makes the comparison cleaner because the downstream classifier remains simple and interpretable. End-to-end fine-tuning would be a valid extension, but it would introduce more hyperparameters and a greater risk of overfitting on this dataset.

### Why is the threshold 0.25 instead of 0.5?

> This is an imbalanced multi-label problem. The default value of 0.5 was too conservative and missed plausible labels. We selected 0.25 using validation Macro-F1 and then locked it before evaluating the test set.

### Does the score prove that SBERT understands semantics?

> No. The score shows that SBERT provides a stable semantic representation, but it does not identify which textual information the model uses. That requires controlled perturbation experiments such as word-order shuffling, keyword masking, and truncation.

### Why can TF-IDF still be competitive?

> Some genres have strong lexical cues, such as “spaceship”, “alien”, or “galaxy” for science fiction. In those cases, sparse keyword features may already contain much of the useful signal. SBERT should be most valuable when meaning is expressed through paraphrase or broader narrative structure.

## Verified experiment facts

- Encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimension: `384`
- Classifier: One-vs-Rest Logistic Regression
- Random seed: `42`
- Training / validation / test: `5,594 / 1,199 / 1,198`
- Number of genres: `18`
- Selected threshold: `0.25`
- Validation Macro-F1 / Micro-F1: `0.5519 / 0.6249`
- Test Macro-F1 / Micro-F1: `0.5476 / 0.6244`
