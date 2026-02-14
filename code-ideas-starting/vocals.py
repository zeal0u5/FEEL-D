import os
import subprocess
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
MODEL_SIZE = "base"   # tiny, base, small, medium, large
SAMPLE_RATE = 16000
CHUNK_DURATION = 5    # seconds per transcription chunk (mic mode)

# ------------------------
# Vocal Separation
# ------------------------
def separate_vocals(input_file):
    print("Separating vocals using Demucs...")
    
    subprocess.run([
        "demucs",
        "--two-stems=vocals",
        input_file
    ], check=True)

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    vocals_path = os.path.join(
        "separated",
        "htdemucs",
        base_name,
        "vocals.wav"
    )

    return vocals_path


# ------------------------
# File Transcription
# ------------------------
def transcribe_file(audio_path, model):
    print("Transcribing audio...")
    result = model.transcribe(audio_path)
    return result["text"]


# ------------------------
# Real-Time Microphone???
# not sure if we want to or need to do this
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

                # Save temp WAV
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                    wav.write(tmpfile.name, SAMPLE_RATE, audio_data)
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
    print("1 - Transcribe music file (with vocal separation)")
    print("2 - Real-time microphone transcription")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        input_file = input("Enter path to audio file: ").strip()

        if not os.path.exists(input_file):
            print("File not found.")
            return

        vocals_file = separate_vocals(input_file)

        if not os.path.exists(vocals_file):
            print("Vocal separation failed.")
            return

        lyrics = transcribe_file(vocals_file, model)

        print("\n--- Detected Lyrics ---\n")
        print(lyrics)

    elif choice == "2":
        transcribe_microphone(model)

    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
