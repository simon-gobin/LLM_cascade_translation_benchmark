# Benchmark Sources

This file collects the main implementation and model sources used for the benchmark script and notebook experiments, so they can be cited later in the final LaTeX report.

## Dataset

- IWSLT 2023/2024 Irish-English shared task data repository:
  https://github.com/shashwatup9k/iwslt2023_ga-eng
- IWSLT 2026 Irish-English repository (blind 2026 test set context):
  https://github.com/shashwatup9k/iwslt2026_ga-eng

## ASR Models

- OpenAI Whisper Small:
  https://huggingface.co/openai/whisper-small
- OpenAI Whisper Medium:
  https://huggingface.co/openai/whisper-medium
- Qwen Qwen3-ASR-1.7B:
  https://huggingface.co/Qwen/Qwen3-ASR-1.7B

## MT Models

- Meta NLLB-200 Distilled 600M:
  https://huggingface.co/facebook/nllb-200-distilled-600M
- Meta NLLB-200 1.3B:
  https://huggingface.co/facebook/nllb-200-1.3B
- Helsinki-NLP OPUS-MT Irish to English:
  https://huggingface.co/Helsinki-NLP/opus-mt-ga-en

## Metrics

- SacreBLEU:
  https://github.com/mjpost/sacrebleu

## Notes for Report Writing

- Whisper implementation in the notebook and `benchmark.py` was adapted from the Hugging Face model-card usage examples.
- NLLB and OPUS-MT loading patterns were adapted from Hugging Face model usage conventions.
- Qwen3-ASR integration was based on the quickstart instructions from the Hugging Face model card.
- BLEU and chrF++ are computed with SacreBLEU for reproducibility.
