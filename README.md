# Whisper Fine-Tuning & Audio Transcription Service

This repository provides a comprehensive audio transcription service using the Whisper model from Hugging Face. It enables users to convert MP3 audio files into text, manage long recordings by processing them in chunks, and includes scripts for fine-tuning the Whisper model on custom datasets.

## Fine-Tuning the Model

For users interested in customizing the Whisper model, a fine-tuning script is provided for use in [Google Colab](https://colab.research.google.com/). Follow these steps:

1. Open the fine-tuning script in Google Colab.
2. Upload your dataset and configure the script parameters.
3. Run the cells to train the model on your data.

Refer to the script comments for detailed instructions on preparing your dataset and adjusting training parameters.

## Audio Transcription with Whisper

This repository showcases an audio transcription application using the Whisper model from the Hugging Face Transformers library. The application takes an MP3 audio file, processes it, and returns a text transcription.

## Features

- Load and fine-tune Whisper model for audio transcription.
- Convert MP3 audio files to a format suitable for the model.
- Transcribe audio in chunks to handle long recordings efficiently.

## Requirements

- Python 3.7+
- PyTorch
- Hugging Face Transformers
- Pydub
- FFmpeg (for audio processing)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/thisiscatcode/whisper.git
   cd whisper
   ```

2. Install the required packages:
   ```bash
   pip install torch transformers pydub
   ```

3. Install FFmpeg:
   - **Windows:** [Download FFmpeg](https://ffmpeg.org/download.html) and add it to your system PATH.
   - **macOS:** Install via Homebrew:
     ```bash
     brew install ffmpeg
     ```
   - **Linux:** Install via your package manager (e.g., `apt`, `yum`).

## Usage

1. Prepare your MP3 audio file (e.g., `6450.mp3`) and place it in the project directory.
2. Run the transcription script:
   ```bash
   python transcribe.py
   ```

3. The transcribed text will be printed to the console.

## How It Works

The script does the following:

- Loads the fine-tuned Whisper model and processor.
- Converts the MP3 audio file into an array of samples.
- Breaks the audio into chunks for efficient processing.
- Uses the Whisper model to generate text transcriptions from each audio chunk.
- Combines the transcriptions from all chunks into a single output.

## Model Fine-Tuning

The model is fine-tuned to improve transcription accuracy. You can replace the model path (`./whisper-mid`) with your own fine-tuned model.

## Contributing

Feel free to fork this repository and submit pull requests. If you have suggestions or improvements, please open an issue.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [PyDub](https://github.com/jiaaro/pydub)
- [FFmpeg](https://ffmpeg.org/)
