import os
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import load_dataset, Dataset
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import pandas as pd
import torchaudio

# Configuration
hug_token = "xxx"
model_name_or_path = "openai/whisper-medium"
language = "japanese"
task = "transcribe"
output_dir = "/content/drive/MyDrive/whisper/"

# Initialize feature extractor, tokenizer, and processor
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# Define a function to prepare the dataset
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

# Load data from CSV
data = pd.read_csv(output_dir + 'openv_ja.csv')

# Create a list of dictionaries with audio paths and transcriptions
dataset = []
for i, row in data.iterrows():
    audio_file = os.path.join(output_dir, row['audio_file'])  # Adjust the audio file path
    transcription = row['transcription']
    waveform, _ = torchaudio.load(audio_file)
    dataset.append({
        'audio': {'array': waveform.squeeze().numpy(), 'sampling_rate': 16000},
        'sentence': transcription
    })

# Convert the list of dictionaries to a Dataset
dataset = Dataset.from_list(dataset)

# Apply the prepare_dataset function to the dataset
processed_dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

# Set both train_dataset and eval_dataset to be the same
train_dataset = processed_dataset
test_dataset = processed_dataset

# Load the Whisper model for conditional generation
model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
model.generation_config.language = language
model.generation_config.task = task
model.generation_config.forced_decoder_ids = None

# Define a data collator for sequence-to-sequence with padding
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# Initialize the data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# Load the Word Error Rate (WER) metric
metric = evaluate.load("wer")

# Define a function to compute metrics
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",  # Use 'steps' for evaluation strategy
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

# Initialize the trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.tokenizer,
)

# Train the model
trainer.train()

# Save the model, tokenizer, and feature extractor
model.save_pretrained(output_dir, safe_serialization=False)
processor.tokenizer.save_pretrained(output_dir)
processor.feature_extractor.save_pretrained(output_dir)

print("finishing the training!")
