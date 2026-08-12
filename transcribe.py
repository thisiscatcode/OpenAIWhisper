"""Transcribe long audio with a fine-tuned Whisper checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from pydub import AudioSegment
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", type=Path, default=Path("whisper-output"))
    parser.add_argument("--language", default="japanese")
    parser.add_argument("--chunk-seconds", type=int, default=30)
    return parser.parse_args()


def load_audio(path: Path) -> torch.Tensor:
    audio = AudioSegment.from_file(path)
    audio = audio.set_frame_rate(16_000).set_channels(1).set_sample_width(2)
    return torch.tensor(audio.get_array_of_samples(), dtype=torch.float32) / (2**15)


def main() -> None:
    args = parse_args()
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    audio = load_audio(args.audio)
    samples_per_chunk = 16_000 * args.chunk_seconds
    transcripts: list[str] = []
    forced_ids = processor.get_decoder_prompt_ids(
        language=args.language, task="transcribe"
    )
    for start in range(0, len(audio), samples_per_chunk):
        chunk = audio[start : start + samples_per_chunk]
        features = processor(
            chunk.numpy(), sampling_rate=16_000, return_tensors="pt"
        ).input_features.to(device)
        with torch.inference_mode():
            predicted_ids = model.generate(features, forced_decoder_ids=forced_ids)
        transcripts.append(
            processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        )
    print(" ".join(part for part in transcripts if part))


if __name__ == "__main__":
    main()
