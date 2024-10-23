import subprocess

# Define pip install commands
pip_installs = [
    'pip install datasets',
    'pip install torchaudio',
    'pip install accelerate',
    'pip install git+https://github.com/huggingface/transformers',
    'pip install librosa',
    'pip install evaluate',
    'pip install jiwer',
    'pip install gradio',
    'pip install -q bitsandbytes datasets accelerate',
    'pip install -q git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git@main',
]

# Execute pip install commands
for pip_install in pip_installs:
    subprocess.run(pip_install, shell=True)

# Run the Python script using subprocess.Popen and redirect the output to the specified file
process = subprocess.Popen(['python', '/content/drive/MyDrive/finetunewhisper4.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
output, error = process.communicate()

from google.colab import drive
drive.mount('/content/drive')

# Write the output and error messages to the output file
with open('/content/drive/MyDrive/test_output.txt', 'w') as f:
    f.write(output.decode())
    f.write(error.decode())

# Confirm that the script has finished running
print("Script execution completed. Output written to: /content/drive/test_output.txt")
