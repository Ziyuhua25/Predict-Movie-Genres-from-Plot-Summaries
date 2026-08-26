# Member 3 presentation materials — official six-input version

## Figures

The `figures/` directory contains seven English 16:9 PNG slides:

1. Why use Sentence-BERT?
2. SBERT pipeline and threshold selection
3. Main SBERT results
4. Exact-match success examples
5. Failure mechanisms
6. Official six-input robustness results
7. Mechanistic interpretation

Slides 6 and 7 were regenerated from
`research/results/sbert_lr/sbert_official_six_metrics.csv`. They replace the
earlier non-official multi-level perturbation figures.

## Slides and scripts

- `slides/Section03_SBERT_English_Insertable_Slides.pptx`: editable core SBERT
  section (slides 1–5); PNG slides 6–7 can be inserted directly afterward
- `scripts/Member3_SBERT_Official_Six_English_Script.md`: final English script
- `scripts/Member3_SBERT_Official_Six_Chinese_Script.md`: matching Chinese script
- `scripts/build_official_six_figures.py`: reproducible generator for slides 6–7
