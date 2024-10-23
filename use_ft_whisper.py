import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pydub import AudioSegment

# Load the fine-tuned model and processor
model_path = "./whisper-mid"
processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path)

def mp3_to_array(mp3_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    samples = audio.get_array_of_samples()
    return torch.tensor(samples).float() / (2**15)

def transcribe_audio_chunk(audio_array, start_idx, end_idx):
    chunk = audio_array[start_idx:end_idx]
    inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_length=500,  # Max length per chunk
            num_beams=5,
            early_stopping=True
        )
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]

# Convert MP3 to array
audio_path = "6450.mp3"
audio_array = mp3_to_array(audio_path)

# Tokenize the initial prompt
initial_prompt = "、。"
prompt_ids = processor.tokenizer.encode(initial_prompt, add_special_tokens=False)
forced_decoder_ids = [(i, token_id) for i, token_id in enumerate(prompt_ids)]

# Define chunk size and process in chunks
chunk_size = 16000 * 30  # 30 seconds chunks
num_chunks = len(audio_array) // chunk_size + 1

full_transcription = ""

for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = (i + 1) * chunk_size if (i + 1) * chunk_size < len(audio_array) else len(audio_array)
    chunk_transcription = transcribe_audio_chunk(audio_array, start_idx, end_idx)
    full_transcription += chunk_transcription + " "

print(full_transcription)
