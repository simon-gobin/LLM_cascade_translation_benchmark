# Assignment 2 - Experiment Tracking Notes

## Goal

Keep a clear record of the experiment ideas for the Irish-to-English speech translation assignment and reuse them later in the report.

## Confirmed Evaluation Setup

- Use `iwslt2023_ga-eng/dev` for evaluation.
- Reason: `iwslt2026_ga-eng/test-2026` is a blind test set without reference translations.
- Report must state:
  - dataset split used
  - exact data source
  - that the 2023 dev split comes from the linked submodule

## Baseline System

- ASR: `openai/whisper-small`
- MT: `facebook/nllb-200-distilled-600M`
- Pipeline: audio -> ASR transcript -> MT English translation

## Important Limitation To Mention

- In this local copy of the dataset, we have:
  - audio files
  - English references
  - metadata in `stamped.tsv`
- We do not appear to have gold Irish transcript text as a plain reference file.
- This means BLEU and chrF++ can be computed for English translation output, but not ASR-vs-gold-Irish transcript accuracy in the usual way.

## Why A Weak Whisper Baseline Is Still Useful

- A weak baseline does not ruin the assignment.
- It gives useful material for:
  - error analysis
  - discussion of low-resource challenges
  - discussion of error propagation from ASR to MT
  - comparison with improved systems

## Planned Experiments

### Experiment 1 - Baseline

- ASR: `whisper-small`
- MT: `nllb-200-distilled-600M`
- Evaluate with BLEU, chrF++, coverage, repetition rate

### Experiment 2 - Better ASR

- Replace `whisper-small` with `whisper-medium`
- Keep the same MT model
- Goal: isolate the effect of improved ASR quality

### Experiment 3 - Better MT

- Keep the baseline ASR
- Replace NLLB with another GA->EN translation model if feasible
- Goal: isolate the effect of MT quality

### Experiment 4 - Direct Speech Translation

- Try a model that handles speech-to-English translation directly, if feasible
- Goal: compare cascaded ASR->MT against end-to-end speech translation

## Suggested Evaluation Strategy

- First debug on 50 examples
- Then scale to a larger subset
- If stable, evaluate on the full dev split
- Always compare systems on the same subset when reporting benchmark results

## Metrics To Report

- BLEU
- chrF++
- Coverage
- Repetition rate

## Error Analysis Ideas

- ASR substitutions
- ASR deletions
- ASR hallucinations
- MT untranslated tokens
- MT mistranslations
- MT over-generation
- Error propagation from ASR into MT

## Report Framing Ideas

- Improving ASR should reduce downstream MT errors.
- Better MT cannot fully recover from a poor transcript.
- A direct speech translation system may avoid some cascade errors, but may also be less robust depending on model support and data coverage.
- Any model limitation around Irish language support in Whisper should be documented clearly.

## Practical Next Steps

1. Finalize the baseline pipeline.
2. Run baseline on a small subset.
3. Compute baseline metrics.
4. Inspect outputs manually.
5. Choose one ASR improvement and one MT comparison.
6. If time allows, test one direct speech translation model.
7. Build the final benchmark table for the report.
