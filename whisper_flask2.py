from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import requests
import time
import mysql.connector
import torch
from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperFeatureExtractor
import torchaudio

app = Flask(__name__)

# Initialize the Whisper model
whisper_model_size = "large-v2"
whisper_model = WhisperModel(whisper_model_size, device="cpu", compute_type="int8")

# Path to the fine-tuned model files
model_dir = "/data/whisper_env/whisper-mid"

# Load the fine-tuned model, tokenizer, and feature extractor
whisper_ft_model = WhisperForConditionalGeneration.from_pretrained(model_dir)
whisper_ft_tokenizer = WhisperTokenizer.from_pretrained(model_dir)
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_dir)

def load_audio(file_path):
    speech_array, sampling_rate = torchaudio.load(file_path)
    if speech_array.shape[0] > 1:
        speech_array = torch.mean(speech_array, dim=0, keepdim=True)
    return speech_array.squeeze().numpy(), sampling_rate

def resample_audio(audio, orig_sr, target_sr):
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        audio = resampler(torch.from_numpy(audio)).numpy()
    return audio

def transcribe_audio(whisper_ft_model, whisper_ft_tokenizer, feature_extractor, audio_file):
    speech_array, sampling_rate = load_audio(audio_file)
    target_sr = feature_extractor.sampling_rate
    speech_array = resample_audio(speech_array, sampling_rate, target_sr)
    
    asr_pipeline = pipeline("automatic-speech-recognition", 
                          model=whisper_ft_model, 
                          tokenizer=whisper_ft_tokenizer, 
                          feature_extractor=feature_extractor)
    
    inputs = {"array": speech_array, "sampling_rate": target_sr}
    result = asr_pipeline(inputs)
    return result['text']

@app.route('/get_whisper_text', methods=['POST'])
def get_whisper_text():
    start_time = time.time()
    request_data = request.get_json()
    
    if 'audio_url' not in request_data:
        return jsonify({"error": "Audio URL is missing in the JSON data"})

    audio_url = request_data['audio_url']
    response = requests.get(audio_url)
    
    if response.status_code != 200:
        return jsonify({"error": "Failed to download the audio file"})

    temp_audio_path = "whisper_audio.webm"
    with open(temp_audio_path, "wb") as audio_file:
        audio_file.write(response.content)

    segments, info = whisper_model.transcribe(temp_audio_path, initial_prompt='、。',language='ja')
    untranslated_result = "\n".join(segment.text for segment in segments)

    total_execution_time = time.time() - start_time
    save_whisper(audio_url, f"(whisper: {total_execution_time:.1f}秒)\n{untranslated_result}", 
                total_execution_time)

    return untranslated_result

@app.route('/get_whisper_text_youtube', methods=['POST'])
def get_whisper_text_youtube():
    start_time = time.time()
    audio_path = request.form.get('audio_path')
    
    segments, info = whisper_model.transcribe(audio_path, initial_prompt='、。？！',language='ja')
    untranslated_result = "\n".join(segment.text for segment in segments)

    total_execution_time = time.time() - start_time
    return untranslated_result

@app.route('/get_ftune_whisper_text_youtube', methods=['POST'])
def get_ftune_whisper_text_youtube():
    start_time = time.time()
    audio_path = request.form.get('audio_path')
    
    transcription = transcribe_audio(whisper_ft_model, whisper_ft_tokenizer, 
                                   feature_extractor, audio_path)
    
    total_execution_time = time.time() - start_time
    return transcription

def save_whisper(file_path, audio_text, total_execution_time):
    try:
        file_path = file_path.replace("", "")
        update_query = "UPDATE bot_mp3_files SET audio_text_2 = %s WHERE file_path = %s"
        data_to_update = (audio_text, file_path)

        mysql_conn = mysql.connector.connect(
            host='xxx',
            user='xxx',
            password='xxx',
            database='xxx'
        )
        
        mysql_cursor = mysql_conn.cursor()
        mysql_cursor.execute(update_query, data_to_update)
        mysql_conn.commit()

    finally:
        if 'mysql_cursor' in locals():
            mysql_cursor.close()
        if 'mysql_conn' in locals():
            mysql_conn.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
