#!/usr/bin/env python3
"""
Benchmark runner for the Irish -> English speech translation assignment.

Designed for the local repo / Colab workflow:
1. clone the repo
2. install the required packages
3. run this script with one or more experiments

Supported ideas:
- Whisper ASR baselines / upgrades
- optional Qwen ASR experiment if qwen-asr is installed
- NLLB and Marian/OPUS-MT translation models
- transcript caching so MT comparisons are faster

Primary implementation / model sources used while designing this script:
- Whisper model usage and model cards:
  https://huggingface.co/openai/whisper-small
  https://huggingface.co/openai/whisper-medium
- NLLB model cards:
  https://huggingface.co/facebook/nllb-200-distilled-600M
  https://huggingface.co/facebook/nllb-200-1.3B
- OPUS-MT Irish->English:
  https://huggingface.co/Helsinki-NLP/opus-mt-ga-en
- Qwen ASR:
  https://huggingface.co/Qwen/Qwen3-ASR-1.7B
- Evaluation metrics:
  SacreBLEU / chrF++ via https://github.com/mjpost/sacrebleu

Example:
    python benchmark.py --dataset-root iwslt2023_ga-eng --experiments baseline whisper_medium opus_mt_ga_en --max-samples 100
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from pipeline_utils import (
    compute_bleu,
    compute_chrf,
    compute_coverage,
    compute_repetition_rate,
    load_audio,
    transcribe_with_whisper,
)


DEFAULT_RESULTS_DIR = "results/benchmark_runs"
DEFAULT_DATASET_ROOT = "iwslt2023_ga-eng"


@dataclass
class Sample:
    index: int
    audio_path: Path
    audio_relpath: str
    reference: str


@dataclass
class ExperimentConfig:
    name: str
    asr_type: str
    asr_model_id: str
    mt_model_id: str
    src_lang: str = "gle_Latn"
    tgt_lang: str = "eng_Latn"
    mt_batch_size: int = 8


EXPERIMENTS: dict[str, ExperimentConfig] = {
    "baseline": ExperimentConfig(
        name="baseline",
        asr_type="whisper",
        asr_model_id="openai/whisper-small",
        mt_model_id="facebook/nllb-200-distilled-600M",
        mt_batch_size=8,
    ),
    "whisper_medium": ExperimentConfig(
        name="whisper_medium",
        asr_type="whisper",
        asr_model_id="openai/whisper-medium",
        mt_model_id="facebook/nllb-200-distilled-600M",
        mt_batch_size=8,
    ),
    "opus_mt_ga_en": ExperimentConfig(
        name="opus_mt_ga_en",
        asr_type="whisper",
        asr_model_id="openai/whisper-small",
        mt_model_id="Helsinki-NLP/opus-mt-ga-en",
        mt_batch_size=16,
    ),
    "nllb_1_3b": ExperimentConfig(
        name="nllb_1_3b",
        asr_type="whisper",
        asr_model_id="openai/whisper-small",
        mt_model_id="facebook/nllb-200-1.3B",
        mt_batch_size=4,
    ),
    "qwen_asr_1_7b": ExperimentConfig(
        name="qwen_asr_1_7b",
        asr_type="qwen",
        asr_model_id="Qwen/Qwen3-ASR-1.7B",
        mt_model_id="facebook/nllb-200-distilled-600M",
        mt_batch_size=8,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run speech translation benchmarks.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(DEFAULT_DATASET_ROOT),
        help="Path to iwslt2023_ga-eng",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(DEFAULT_RESULTS_DIR),
        help="Directory for cached transcripts, translations, and metrics",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["baseline"],
        choices=sorted(EXPERIMENTS.keys()),
        help="Experiments to run",
    )
    parser.add_argument("--max-samples", type=int, default=50, help="Number of dev samples to run")
    parser.add_argument("--start-index", type=int, default=0, help="Start index in the dev split")
    parser.add_argument("--device", default=None, help="cuda, cpu, mps, or auto-detect if omitted")
    parser.add_argument("--mt-batch-size", type=int, default=None, help="Override MT batch size")
    parser.add_argument("--force-rerun-asr", action="store_true", help="Ignore cached transcripts")
    parser.add_argument("--force-rerun-mt", action="store_true", help="Ignore cached translations")
    parser.add_argument("--skip-existing", action="store_true", help="Skip experiments with existing metrics.json")
    return parser.parse_args()


def choose_device(explicit_device: str | None) -> str:
    if explicit_device:
        return explicit_device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_dev_samples(dataset_root: Path, start_index: int, max_samples: int) -> list[Sample]:
    dev_root = dataset_root / "dev"
    wav_dir = dev_root / "wav"
    trans_file = dev_root / "stamped.tsv"
    ref_file = dev_root / "txt" / "dev.eng"

    if not wav_dir.exists():
        raise FileNotFoundError(f"Missing wav dir: {wav_dir}")
    if not trans_file.exists():
        raise FileNotFoundError(f"Missing metadata file: {trans_file}")
    if not ref_file.exists():
        raise FileNotFoundError(f"Missing reference file: {ref_file}")

    with trans_file.open(encoding="utf-8") as f:
        audio_relpaths = [line.strip().split("\t")[0] for line in f if line.strip()]

    with ref_file.open(encoding="utf-8") as f:
        references = [line.strip() for line in f if line.strip()]

    if len(audio_relpaths) != len(references):
        raise ValueError(
            f"Metadata/reference size mismatch: {len(audio_relpaths)} vs {len(references)}"
        )

    end_index = min(start_index + max_samples, len(references))
    samples: list[Sample] = []
    for idx in range(start_index, end_index):
        relpath = audio_relpaths[idx]
        audio_path = dataset_root / "dev" / relpath
        if not audio_path.exists():
            raise FileNotFoundError(f"Missing audio file for sample {idx}: {audio_path}")
        samples.append(
            Sample(
                index=idx,
                audio_path=audio_path,
                audio_relpath=relpath,
                reference=references[idx],
            )
        )
    return samples


class WhisperASR:
    def __init__(self, model_id: str, device: str):
        self.model_id = model_id
        self.device = device
        # Source adapted from Hugging Face Whisper usage examples:
        # https://huggingface.co/openai/whisper-small
        # https://huggingface.co/openai/whisper-medium
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device).eval()
        self.model.generation_config.forced_decoder_ids = None
        self.model.generation_config.task = "transcribe"
        self.model.generation_config.language = None

    def transcribe(self, audio_path: Path) -> str:
        return transcribe_with_whisper(
            audio_path=audio_path,
            processor=self.processor,
            model=self.model,
        )


class QwenASR:
    def __init__(self, model_id: str, device: str):
        self.model_id = model_id
        self.device = device
        # Source adapted from the Qwen3-ASR model card quickstart:
        # https://huggingface.co/Qwen/Qwen3-ASR-1.7B
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise ImportError(
                "Qwen ASR requires the `qwen-asr` package. Install it first, for example: "
                "`pip install -U qwen-asr`"
            ) from exc
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = Qwen3ASRModel.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device,
            max_inference_batch_size=1,
            max_new_tokens=256,
        )

    def transcribe(self, audio_path: Path) -> str:
        results = self.model.transcribe(audio=str(audio_path), language=None)
        if not results:
            return ""
        return results[0].text.strip()


class MTTranslator:
    def __init__(self, model_id: str, device: str, src_lang: str, tgt_lang: str):
        self.model_id = model_id
        self.device = device
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        # Source adapted from Hugging Face model usage for:
        # - NLLB: https://huggingface.co/facebook/nllb-200-distilled-600M
        # - NLLB: https://huggingface.co/facebook/nllb-200-1.3B
        # - Marian OPUS-MT: https://huggingface.co/Helsinki-NLP/opus-mt-ga-en
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device).eval()
        self.is_nllb = "nllb" in model_id.lower()
        self.is_marian = "opus-mt" in model_id.lower()

    def translate_batch(self, texts: list[str], max_new_tokens: int = 256) -> list[str]:
        if not texts:
            return []

        clean_texts = [t if t and t.strip() else "" for t in texts]
        non_empty_indices = [i for i, t in enumerate(clean_texts) if t.strip()]
        outputs = [""] * len(clean_texts)
        if not non_empty_indices:
            return outputs

        batch_inputs = [clean_texts[i] for i in non_empty_indices]
        with torch.no_grad():
            if self.is_nllb:
                self.tokenizer.src_lang = self.src_lang
                encoded = self.tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True)
                encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
                forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)
                generated = self.model.generate(
                    **encoded,
                    forced_bos_token_id=forced_bos_token_id,
                    max_new_tokens=max_new_tokens,
                )
            else:
                encoded = self.tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True)
                encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
                generate_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
                if self.is_marian:
                    generate_kwargs["renormalize_logits"] = True
                generated = self.model.generate(**encoded, **generate_kwargs)

        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        for target_index, text in zip(non_empty_indices, decoded):
            outputs[target_index] = text.strip()
        return outputs


def build_asr_runner(config: ExperimentConfig, device: str):
    if config.asr_type == "whisper":
        return WhisperASR(config.asr_model_id, device)
    if config.asr_type == "qwen":
        return QwenASR(config.asr_model_id, device)
    raise ValueError(f"Unsupported ASR type: {config.asr_type}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write((line or "").replace("\n", " ").strip() + "\n")


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def run_asr(
    asr_runner: Any,
    samples: list[Sample],
    transcript_cache_path: Path,
    force_rerun: bool,
) -> tuple[list[str], list[float], list[int]]:
    if transcript_cache_path.exists() and not force_rerun:
        payload = json.loads(transcript_cache_path.read_text(encoding="utf-8"))
        return payload["transcripts"], payload.get("asr_times", []), payload.get("failed_indices", [])

    transcripts: list[str] = []
    asr_times: list[float] = []
    failed_indices: list[int] = []

    for sample in tqdm(samples, desc="ASR", leave=False):
        start = time.time()
        try:
            transcript = asr_runner.transcribe(sample.audio_path)
        except Exception as exc:
            print(f"[ASR ERROR] sample={sample.index} file={sample.audio_path.name}: {exc}")
            transcript = ""
            failed_indices.append(sample.index)
        asr_times.append(time.time() - start)
        transcripts.append(transcript)

    write_json(
        transcript_cache_path,
        {
            "transcripts": transcripts,
            "asr_times": asr_times,
            "failed_indices": failed_indices,
        },
    )
    return transcripts, asr_times, failed_indices


def batched(seq: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(seq), batch_size):
        yield seq[start : start + batch_size]


def run_mt(
    translator: MTTranslator,
    transcripts: list[str],
    translation_cache_path: Path,
    force_rerun: bool,
    batch_size: int,
) -> list[str]:
    if translation_cache_path.exists() and not force_rerun:
        payload = json.loads(translation_cache_path.read_text(encoding="utf-8"))
        return payload["translations"]

    translations: list[str] = []
    for chunk in tqdm(list(batched(transcripts, batch_size)), desc="MT", leave=False):
        translations.extend(translator.translate_batch(chunk))

    write_json(translation_cache_path, {"translations": translations})
    return translations


def evaluate_outputs(translations: list[str], references: list[str]) -> dict[str, Any]:
    valid_hyps: list[str] = []
    valid_refs: list[str] = []
    for hyp, ref in zip(translations, references):
        if hyp.strip() and ref.strip():
            valid_hyps.append(hyp)
            valid_refs.append(ref)

    if valid_hyps:
        bleu = compute_bleu(valid_hyps, valid_refs)
        chrf = compute_chrf(valid_hyps, valid_refs)
    else:
        bleu = 0.0
        chrf = 0.0

    return {
        "samples_evaluated": len(valid_hyps),
        "bleu": bleu,
        "chrf_pp": chrf,
        "coverage": compute_coverage(translations),
        "repetition_rate": compute_repetition_rate(translations),
    }


def append_summary_row(summary_csv_path: Path, row: dict[str, Any]) -> None:
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "asr_model",
        "mt_model",
        "max_samples",
        "samples_evaluated",
        "bleu",
        "chrf_pp",
        "coverage",
        "repetition_rate",
        "avg_asr_time_sec",
        "failed_count",
    ]
    write_header = not summary_csv_path.exists()
    with summary_csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_experiment(
    config: ExperimentConfig,
    samples: list[Sample],
    results_dir: Path,
    device: str,
    force_rerun_asr: bool,
    force_rerun_mt: bool,
    mt_batch_size_override: int | None,
) -> dict[str, Any]:
    exp_dir = results_dir / config.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    transcript_cache_path = exp_dir / "transcripts.json"
    translation_cache_path = exp_dir / "translations.json"
    metrics_path = exp_dir / "metrics.json"
    refs_path = exp_dir / "references.txt"
    hyp_path = exp_dir / "hypotheses.txt"
    transcript_txt_path = exp_dir / "transcripts.txt"

    references = [sample.reference for sample in samples]
    write_lines(refs_path, references)

    asr_runner = build_asr_runner(config, device)
    transcripts, asr_times, failed_indices = run_asr(
        asr_runner=asr_runner,
        samples=samples,
        transcript_cache_path=transcript_cache_path,
        force_rerun=force_rerun_asr,
    )
    write_lines(transcript_txt_path, transcripts)

    translator = MTTranslator(
        model_id=config.mt_model_id,
        device=device,
        src_lang=config.src_lang,
        tgt_lang=config.tgt_lang,
    )
    mt_batch_size = mt_batch_size_override or config.mt_batch_size
    translations = run_mt(
        translator=translator,
        transcripts=transcripts,
        translation_cache_path=translation_cache_path,
        force_rerun=force_rerun_mt,
        batch_size=mt_batch_size,
    )
    write_lines(hyp_path, translations)

    metrics = evaluate_outputs(translations, references)
    metrics.update(
        {
            "experiment": config.name,
            "device": device,
            "dataset_root": str(results_dir),
            "asr_model": config.asr_model_id,
            "mt_model": config.mt_model_id,
            "max_samples": len(samples),
            "avg_asr_time_sec": float(np.mean(asr_times)) if asr_times else 0.0,
            "failed_count": len(failed_indices),
            "failed_indices": failed_indices,
        }
    )
    write_json(metrics_path, metrics)
    return metrics


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    samples = load_dev_samples(args.dataset_root, args.start_index, args.max_samples)
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = results_dir / "summary.csv"

    print(f"Device: {device}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Samples: {len(samples)} (start_index={args.start_index})")
    print(f"Experiments: {', '.join(args.experiments)}")

    all_metrics: list[dict[str, Any]] = []
    for exp_name in args.experiments:
        config = EXPERIMENTS[exp_name]
        metrics_path = results_dir / config.name / "metrics.json"
        if args.skip_existing and metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            print(f"[skip-existing] {exp_name}")
        else:
            print(f"\n=== Running experiment: {exp_name} ===")
            metrics = run_experiment(
                config=config,
                samples=samples,
                results_dir=results_dir,
                device=device,
                force_rerun_asr=args.force_rerun_asr,
                force_rerun_mt=args.force_rerun_mt,
                mt_batch_size_override=args.mt_batch_size,
            )
        all_metrics.append(metrics)
        append_summary_row(
            summary_csv_path,
            {
                "experiment": metrics["experiment"],
                "asr_model": metrics["asr_model"],
                "mt_model": metrics["mt_model"],
                "max_samples": metrics["max_samples"],
                "samples_evaluated": metrics["samples_evaluated"],
                "bleu": f"{metrics['bleu']:.2f}",
                "chrf_pp": f"{metrics['chrf_pp']:.2f}",
                "coverage": f"{metrics['coverage']:.4f}",
                "repetition_rate": f"{metrics['repetition_rate']:.4f}",
                "avg_asr_time_sec": f"{metrics['avg_asr_time_sec']:.2f}",
                "failed_count": metrics["failed_count"],
            },
        )
        print(json.dumps(metrics, indent=2))

    write_json(results_dir / "latest_run.json", all_metrics)
    print(f"\nSaved summary CSV to: {summary_csv_path}")


if __name__ == "__main__":
    main()
