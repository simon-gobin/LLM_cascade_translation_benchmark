"""
Reusable pipeline utilities extracted from the notebook.

These helpers mirror the notebook architecture:
- load_audio
- transcribe_with_whisper
- translate_with_nllb
- speech_to_text_translate
- evaluation helpers

Sources:
- Whisper model usage:
  https://huggingface.co/openai/whisper-small
  https://huggingface.co/openai/whisper-medium
- NLLB model usage:
  https://huggingface.co/facebook/nllb-200-distilled-600M
  https://huggingface.co/facebook/nllb-200-1.3B
- SacreBLEU:
  https://github.com/mjpost/sacrebleu
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import sacrebleu


TARGET_SR = 16000


def load_audio(filepath: str | Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """
    Load an audio file, resample to target_sr, and return mono float32 waveform.
    """
    waveform, sr = librosa.load(filepath, sr=target_sr, mono=True)
    waveform = waveform.astype(np.float32)
    return waveform, sr


def transcribe_with_whisper(audio_path, processor, model, task: str = "transcribe") -> str:
    """
    Transcribe a single audio file with Whisper.

    Notes:
    - We intentionally do not force language='ga' here because that setting was
      not reliably available in the tested Whisper setup.
    - This mirrors the notebook implementation.
    """
    waveform, sr = load_audio(audio_path)
    inputs = processor(
        waveform,
        sampling_rate=sr,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_features = inputs.input_features.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    predicted_ids = model.generate(
        input_features,
        attention_mask=attention_mask,
        task=task,
    )
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription.strip()


def translate_with_nllb(
    text: str,
    tokenizer,
    model,
    src_lang: str = "gle_Latn",
    tgt_lang: str = "eng_Latn",
    max_new_tokens: int = 256,
) -> str:
    """
    Translate Irish text to English using NLLB-style tokenizers/models.
    """
    if not text or not text.strip():
        return ""

    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_new_tokens=max_new_tokens,
    )
    translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return translation.strip()


def speech_to_text_translate(
    audio_path,
    asr_processor,
    asr_model,
    mt_tokenizer,
    mt_model,
    src_lang: str = "gle_Latn",
    tgt_lang: str = "eng_Latn",
) -> tuple[str, str]:
    """
    Full cascaded speech translation: audio -> ASR transcript -> English translation.
    """
    irish_transcript = transcribe_with_whisper(audio_path, asr_processor, asr_model)
    english_translation = translate_with_nllb(
        irish_transcript,
        mt_tokenizer,
        mt_model,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )
    return irish_transcript, english_translation


def compute_bleu(hypotheses: list[str], references: list[str]) -> float:
    return sacrebleu.corpus_bleu(hypotheses, [references]).score


def compute_chrf(hypotheses: list[str], references: list[str]) -> float:
    return sacrebleu.corpus_chrf(hypotheses, [references], word_order=2).score


def compute_coverage(hypotheses: list[str]) -> float:
    n_empty = sum(1 for h in hypotheses if not h.strip())
    return 1.0 - (n_empty / max(len(hypotheses), 1))


def compute_repetition_rate(hypotheses: list[str], threshold: float = 0.3) -> float:
    flagged = 0
    for hypothesis in hypotheses:
        words = hypothesis.split()
        if len(words) < 4:
            continue
        freqs = Counter(words)
        if freqs.most_common(1)[0][1] / len(words) > threshold:
            flagged += 1
    return flagged / max(len(hypotheses), 1)
