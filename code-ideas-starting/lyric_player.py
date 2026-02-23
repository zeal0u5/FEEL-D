# #######################################################
# Cross-Platform Minimal Lyric Player
# Works on Desktop + Raspberry Pi 3B
# -------------------------------------------------------
# - SQLite caching
# - Word-level timestamps
# - Inline karaoke highlighting
# - Fullscreen option for Pi
# #######################################################

import os
import sys
import hashlib
import sqlite3
import json
import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import threading
import time
import tkinter as tk
from tkinter import filedialog
import argparse

# ------------------------
# CONFIG
# ------------------------
MODEL_SIZE = "base"
DB_FILE = "transcripts.db"
WINDOW_SIZE = 240
FONT_SIZE = 20


# ------------------------
# DATABASE
# ------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transcripts (
            file_hash TEXT PRIMARY KEY,
            filename TEXT,
            transcript TEXT,
            word_data TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_to_db(file_hash, filename, transcript, word_data):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO transcripts VALUES (?, ?, ?, ?)
    """, (file_hash, filename, transcript, json.dumps(word_data)))
    conn.commit()
    conn.close()


def load_from_db(file_hash):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT transcript, word_data FROM transcripts WHERE file_hash=?", (file_hash,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return row[0], json.loads(row[1])
    return None, None


# ------------------------
# HASH
# ------------------------
def get_file_hash(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)
    return sha256.hexdigest()


# ------------------------
# AUDIO
# ------------------------
def play_audio(audio_path):
    samplerate, data = wav.read(audio_path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    sd.play(data, samplerate)
    sd.wait()


# ------------------------
# TRANSCRIBE
# ------------------------
def transcribe(audio_path):
    model = whisper.load_model(MODEL_SIZE)
    result = model.transcribe(audio_path, word_timestamps=True)

    word_data = []
    for segment in result["segments"]:
        for word in segment.get("words", []):
            word_data.append({
                "start": word["start"],
                "end": word["end"],
                "word": word["word"]
            })

    return result["text"], word_data


# ------------------------
# GUI PLAYER
# ------------------------
class LyricPlayer:
    def __init__(self, root, fullscreen=False):
        self.root = root
        self.fullscreen = fullscreen

        self.root.configure(bg="black")

        if fullscreen:
            self.root.attributes("-fullscreen", True)
        else:
            self.root.geometry(f"{WINDOW_SIZE}x{WINDOW_SIZE}")

        self.text = tk.Text(
            root,
            wrap="word",
            font=("Helvetica", FONT_SIZE, "bold"),
            bg="black",
            fg="white",
            bd=0,
            highlightthickness=0
        )
        self.text.pack(expand=True, fill="both")

        self.text.tag_configure("highlight", foreground="yellow")
        self.word_positions = []
        self.word_data = []

    def prepare_text(self, transcript):
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", transcript)

        # Build word index positions
        index = "1.0"
        self.word_positions = []

        for word in transcript.split():
            start_index = self.text.search(word, index, stopindex=tk.END)
            if not start_index:
                continue
            end_index = f"{start_index}+{len(word)}c"
            self.word_positions.append((start_index, end_index))
            index = end_index

    def highlight_word(self, index):
        self.text.tag_remove("highlight", "1.0", tk.END)
        if index < len(self.word_positions):
            start, end = self.word_positions[index]
            self.text.tag_add("highlight", start, end)
            self.text.see(start)

    def play(self, filepath):
        file_hash = get_file_hash(filepath)
        transcript, word_data = load_from_db(file_hash)

        if not transcript:
            transcript, word_data = transcribe(filepath)
            save_to_db(file_hash, os.path.basename(filepath), transcript, word_data)

        self.word_data = word_data
        self.prepare_text(transcript)

        audio_thread = threading.Thread(target=play_audio, args=(filepath,))
        lyric_thread = threading.Thread(target=self.sync_words)

        audio_thread.start()
        lyric_thread.start()

    def sync_words(self):
        start_time = time.perf_counter()

        for i, word in enumerate(self.word_data):
            while time.perf_counter() - start_time < word["start"]:
                time.sleep(0.001)

            self.root.after(0, self.highlight_word, i)


# ------------------------
# MAIN
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args()

    init_db()

    root = tk.Tk()
    player = LyricPlayer(root, fullscreen=args.fullscreen)

    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        player.play(file_path)

    root.mainloop()


if __name__ == "__main__":
    main()