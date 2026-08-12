"""Fine-tune Whisper on a local CSV/audio dataset.

Expected CSV columns:
  audio_file,transcription

The split is created before feature extraction so evaluation examples are not
used for training. WER is therefore measured on a held-out set.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import evaluate
import numpy as np
import pandas as pd
import torch
import torchaudio
from datasets import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("whisper-output"))
    parser.add_argument("--model", default="openai/whisper-medium")
    parser.add_argument("--language", default="japanese")
    parser.add_argument("--eval-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    return parser.parse_args()


def load_manifest(csv_path: Path, audio_root: Path) -> Dataset:
    frame = pd.read_csv(csv_path)
    required = {"audio_file", "transcription"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    frame = frame.loc[:, ["audio_file", "transcription"]].dropna()
    frame["audio_file"] = frame["audio_file"].map(
        lambda value: str((audio_root / str(value)).resolve())
    )
    missing_audio = [path for path in frame["audio_file"] if not Path(path).is_file()]
    if missing_audio:
        preview = ", ".join(missing_audio[:3])
        raise FileNotFoundError(f"Audio files not found (first three): {preview}")
    if len(frame) < 2:
        raise ValueError("At least two valid examples are required for a held-out split")
    return Dataset.from_pandas(frame, preserve_index=False)


def load_audio(path: str, target_rate: int = 16_000) -> np.ndarray:
    waveform, sample_rate = torchaudio.load(path)
    waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_rate)
    return waveform.squeeze(0).numpy()


@dataclass
class SpeechSeq2SeqCollator:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        audio_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(audio_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main() -> None:
    args = parse_args()
    if not 0 < args.eval_size < 1:
        raise ValueError("--eval-size must be between 0 and 1")

    processor = WhisperProcessor.from_pretrained(
        args.model, language=args.language, task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.generation_config.language = args.language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    dataset = load_manifest(args.csv, args.audio_root)
    split = dataset.train_test_split(test_size=args.eval_size, seed=args.seed)

    def prepare(example: dict[str, Any]) -> dict[str, Any]:
        audio = load_audio(example["audio_file"])
        inputs = processor.feature_extractor(
            audio, sampling_rate=16_000, return_tensors="np"
        )
        return {
            "input_features": inputs.input_features[0],
            "labels": processor.tokenizer(example["transcription"]).input_ids,
        }

    processed = split.map(
        prepare,
        remove_columns=split["train"].column_names,
        desc="Extracting Whisper features",
    )
    metric = evaluate.load("wer")

    def compute_metrics(prediction: Any) -> dict[str, float]:
        prediction_ids = prediction.predictions
        label_ids = np.array(prediction.label_ids, copy=True)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        predictions = processor.tokenizer.batch_decode(
            prediction_ids, skip_special_tokens=True
        )
        references = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )
        return {"wer": 100 * metric.compute(predictions=predictions, references=references)}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        evaluation_strategy="steps",
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=500,
        eval_steps=500,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        seed=args.seed,
    )
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=processed["train"],
        eval_dataset=processed["test"],
        data_collator=SpeechSeq2SeqCollator(
            processor=processor,
            decoder_start_token_id=model.config.decoder_start_token_id,
        ),
        compute_metrics=compute_metrics,
        tokenizer=processor.tokenizer,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_metrics("eval", metrics)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    processor.save_pretrained(args.output_dir)
    print(f"Saved model and held-out evaluation metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
