import torch
from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperFeatureExtractor, pipeline
import torchaudio
from flask import Flask, request, jsonify
import time 

import torch
import transformers
import torchaudio

app = Flask(__name__)

# Path to the directory containing the fine-tuned model files
model_dir = "./whisper-mid"

# Load the fine-tuned model, tokenizer, and feature extractor
model = WhisperForConditionalGeneration.from_pretrained(model_dir)
tokenizer = WhisperTokenizer.from_pretrained(model_dir)
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_dir)

# Function to load and preprocess the audio file
def load_audio(file_path):
    speech_array, sampling_rate = torchaudio.load(file_path)
    # Convert to single channel (mono) if it is not already
    if speech_array.shape[0] > 1:
        speech_array = torch.mean(speech_array, dim=0, keepdim=True)
    return speech_array.squeeze().numpy(), sampling_rate

# Function to resample audio if necessary
def resample_audio(audio, orig_sr, target_sr):
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        audio = resampler(torch.from_numpy(audio)).numpy()
    return audio

# Function to transcribe audio using the fine-tuned model
def transcribe_audio(model, tokenizer, feature_extractor, audio_file):
    speech_array, sampling_rate = load_audio(audio_file)
    # Ensure the audio is resampled to the target sample rate
    target_sr = feature_extractor.sampling_rate
    speech_array = resample_audio(speech_array, sampling_rate, target_sr)
    
    #tokenizer.set_prefix_tokens("ja")
    # Use the pipeline for automatic speech recognition
    asr_pipeline = pipeline("automatic-speech-recognition", model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)
    
    # The pipeline expects a list of dictionaries with "array" and "sampling_rate"
    inputs = {"array": speech_array, "sampling_rate": target_sr}
    result = asr_pipeline(inputs)
    return result['text']


@app.route('/get_ftune_whisper_text_youtube', methods=['POST'])
def get_ftune_whisper_text_youtube():
    # Start measuring the execution time
    start_time = time.time()

    audio_path = request.form.get('audio_path')
    print(audio_path)
    # Transcribe the audio file
    transcription = transcribe_audio(model, tokenizer, feature_extractor, audio_path)
    print("Whisper Transcription:", transcription)

    total_execution_time = time.time() - start_time

    print(total_execution_time)
    return transcription;



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
