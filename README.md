# eng198-gen-ai
ENGIN 198 - Gen AI Project

Problem and Objectives
What is our specific use case?
     - General-purpose transcription (e.g., meetings, lectures).
     - Domain-specific applications (e.g., medical, legal, customer support).
     - Real-time transcription or offline processing.

Success metrics
     - Accuracy (e.g., Word Error Rate, WER).
     - Latency for real-time transcription.
     - User-friendliness of the interface.

Gather and Prepare Data
Collect Audio Data
Use publicly available datasets:
[LibriSpeech](https://www.openslr.org/12) for general speech.
[Common Voice by Mozilla](https://commonvoice.mozilla.org/) for diverse accents and languages.
Domain-specific datasets (e.g., medical transcription datasets).
Record your own dataset if your use case is niche.

Prepare Text Labels
Ensure transcriptions are available and accurately aligned with audio clips.

Preprocess Data
Clean and normalize text (e.g., lowercase, remove punctuations)
Segment long audio files into smaller chunks.

Choose or Build a Model
Pre-Trained Speech-to-Text Models
     - Use an existing model like:
       - OpenAI Whisper (robust across multiple languages and accents).
       - Google Cloud Speech-to-Text API.
       - Microsoft Azure Cognitive Services.
       - Hugging Face models like `wav2vec 2.0` or `Whisper`.

Fine-Tuning
Fine-tune pre-trained models on your domain-specific data for better accuracy.

Custom Models
If building from scratch, explore architectures like:
RNN/GRU/LSTM-based models.
I know we’re exploring RNNs in Data 144 right now… 👀
Transformer-based models for modern STT systems (e.g., Wav2Vec).

Build the Pipeline
Components of the Pipeline:
Input Handling: Accept audio (via files or microphone input).
Preprocessing: Convert audio to the required format (e.g., 16 kHz, mono).
Model Inference: Pass the preprocessed audio through the model to get text output.
Post-processing: Correct minor transcription errors (e.g., grammar or punctuation).
Output: Display or save transcriptions in real-time or as batch files.

Use tools like `ffmpeg` for audio preprocessing and libraries like `SpeechRecognition`, `torchaudio`, or `pydub`.

Implement Additional Features
Real-Time Transcription
Use streaming APIs or frameworks like WebRTC to process audio in real-time.
Language Models for Context
Incorporate language models like GPT to correct and improve transcriptions.
Speaker Identification
Add speaker diarization to identify different speakers in a conversation.
Sentiment Analysis
Post-process text to analyze emotion or sentiment.

Evaluate and Optimize
Evaluate transcription accuracy using:
Word Error Rate (WER).
Sentence Error Rate (SER).
Optimize for:
Faster inference times.
Reduced memory usage.
Experiment with different pre-trained models or fine-tune hyperparameters.


Deploy the System
Create a User Interface
Build a desktop or web app (e.g., using Flask, Django, or React)
Use a command-line tool for simpler projects.
Deploy the Model
Deploy on the cloud using platforms like AWS, Google Cloud, or Azure.


Tools and Libraries
Python Libraries
`speech_recognition`, `pydub`, `librosa` (audio preprocessing).
`transformers`, `torchaudio` (STT models).
APIs
Google Cloud Speech-to-Text, AssemblyAI, Deepgram, or Whisper API.
Framework
PyTorch, TensorFlow, or ONNX.

# Example Code (Using Whisper)
import whisper
# Load the pre-trained model
model = whisper.load_model("base")
# Transcribe audio
result = model.transcribe("path_to_audio_file.wav")
# Print the transcription
print("Transcription:", result["text"])


