# #######################################################
# Docstring for code-ideas-starting.vocals
# 
# This program either takes in a .wav file and returns the transcript
# of the vocal part to console or saved in a JSON file as a SHA256
# key-value pair or listens to live vocals and returns the transcript
# to console.
# #######################################################


import os
import json
import hashlib
import whisper
import sounddevice as sd
import numpy as np
import queue
import sys
import tempfile
import scipy.io.wavfile as wav

# ------------------------
# CONFIG
# ------------------------
MODEL_SIZE = "base"
SAMPLE_RATE = 16000
CHUNK_DURATION = 5
OUTPUT_FILE = "transcripts.json"


# ------------------------
# SHA256 File Hash
# ------------------------
def get_file_hash(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)
    return sha256.hexdigest()


# ------------------------
# File Transcription
# ------------------------
def transcribe_file(audio_path, model):
    print("Transcribing audio...")
    result = model.transcribe(audio_path)
    return result["text"].strip()


# ------------------------
# Save Transcript (SHA256 Key)
# ------------------------
def save_transcript(audio_path, text):
    file_hash = get_file_hash(audio_path)
    filename = os.path.basename(audio_path)

    # Load existing data
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            try:
                transcripts = json.load(f)
            except json.JSONDecodeError:
                transcripts = {}
    else:
        transcripts = {}

    # Check for duplicate
    if file_hash in transcripts:
        print("This file has already been processed. Skipping save.")
        return False

    # Add new entry
    transcripts[file_hash] = {
        "filename": filename,
        "transcript": text
    }

    # Save JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(transcripts, f, indent=4, ensure_ascii=False)

    print("Transcript saved (SHA256 key, no duplicates).")
    return True


# ------------------------
# Real-Time Microphone
# ------------------------
def transcribe_microphone(model):
    print("Recording from microphone...")
    print("Press Ctrl+C to stop.\n")

    q = queue.Queue()

    def audio_callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=audio_callback
    ):
        try:
            while True:
                print("Listening...")
                frames = []

                for _ in range(int(SAMPLE_RATE / 1024 * CHUNK_DURATION)):
                    frames.append(q.get())

                audio_data = np.concatenate(frames, axis=0)
                audio_int16 = np.int16(audio_data * 32767)

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                    wav.write(tmpfile.name, SAMPLE_RATE, audio_int16)
                    temp_path = tmpfile.name

                result = model.transcribe(temp_path)
                text = result["text"].strip()

                if text:
                    print("You said:", text)

                os.remove(temp_path)

        except KeyboardInterrupt:
            print("\nStopping microphone transcription.")


# ------------------------
# Main Program
# ------------------------
def main():
    print("Loading Whisper model...")
    model = whisper.load_model(MODEL_SIZE)

    print("\nSelect Mode:")
    print("1 - Transcribe audio file")
    print("2 - Real-time microphone transcription")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        input_file = input("Enter path to audio file: ").strip()

        if not os.path.exists(input_file):
            print("File not found.")
            return

        file_hash = get_file_hash(input_file)

        # Check if already processed before running Whisper
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                try:
                    transcripts = json.load(f)
                except json.JSONDecodeError:
                    transcripts = {}
        else:
            transcripts = {}

        if file_hash in transcripts:
            print("\nThis file was already processed.")
            print("Transcript:\n")
            print(transcripts[file_hash]["transcript"])
            return

        text = transcribe_file(input_file, model)

        print("\n--- Transcription ---\n")
        print(text)

        save_transcript(input_file, text)

    elif choice == "2":
        transcribe_microphone(model)

    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
