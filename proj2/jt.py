# Import necessary libraries
import os
import whisper
from transformers import pipeline
from pydub import AudioSegment

def convert_mp4_to_audio(mp4_file, audio_file="audio.wav"):
    """
    Converts an MP4 video file to audio in WAV format.
    """
    try:
        video = AudioSegment.from_file(mp4_file, format="mp4")
        video.export(audio_file, format="wav")
        return audio_file
    except Exception as e:
        print(f"Error converting MP4 to audio: {e}")
        return None

def transcribe_audio(file_path):
    """
    Transcribes the audio file using Whisper.
    """
    try:
        model = whisper.load_model("base")  # Replace "base" with other model sizes if needed
        result = model.transcribe(file_path)
        return result['text']
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

def summarize_text(text, max_length=150, min_length=30):
    """
    Summarizes the text using a pre-trained summarization model.
    """
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return None

def process_mp4(mp4_file):
    """
    Main function to process an MP4 file: Convert, transcribe, and summarize.
    """
    # Step 1: Convert MP4 to audio
    print("Converting MP4 to audio...")
    audio_file = convert_mp4_to_audio(mp4_file)
    if not audio_file:
        return "Failed to convert MP4 to audio."

    # Step 2: Transcribe audio
    print("Transcribing audio...")
    transcription = transcribe_audio(audio_file)
    if not transcription:
        return "Failed to transcribe audio."

    # Step 3: Summarize transcription
    print("Summarizing transcription...")
    summary = summarize_text(transcription)
    if not summary:
        return "Failed to summarize transcription."

    return summary

if __name__ == "__main__":
    # Input MP4 file
    mp4_file = input("Enter the path to the MP4 file: ")
    if not os.path.exists(mp4_file):
        print("File not found. Please provide a valid MP4 file.")
    else:
        print("Processing file...")
        result = process_mp4(mp4_file)
        print("\nSummary of Key Points:")
        print(result)
