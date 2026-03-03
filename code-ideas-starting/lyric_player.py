# #######################################################
# Cross-Platform Minimal Lyric Player
# - SQLite caching
# - Editable transcripts
# - Re-run Whisper for alignment after edits
# - Word-level highlighting
# #######################################################

import os
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
from tkinter import filedialog, messagebox
import argparse
import difflib

# ------------------------
# CONFIG
# ------------------------
MODEL_SIZE = "base"
DB_FILE = "transcripts.db"
WINDOW_SIZE = 600
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
# TRANSCRIPT EDITOR
# ------------------------
def review_transcript(original_text):
    review_root = tk.Toplevel()
    review_root.title("Edit Transcript")
    review_root.geometry("800x600")

    text_box = tk.Text(review_root, wrap="word", font=("Helvetica", 14))
    text_box.pack(expand=True, fill="both")
    text_box.insert("1.0", original_text)

    result = {"text": None}

    def save():
        result["text"] = text_box.get("1.0", tk.END).strip()
        review_root.destroy()

    def cancel():
        review_root.destroy()

    button_frame = tk.Frame(review_root)
    button_frame.pack(fill="x")

    tk.Button(button_frame, text="Save Changes", command=save).pack(side="right")
    tk.Button(button_frame, text="Cancel", command=cancel).pack(side="right")

    review_root.grab_set()
    review_root.wait_window()

    return result["text"]


# ------------------------
# ALIGN EDITED TEXT USING FRESH WHISPER RUN
# ------------------------
def align_to_edited_text(audio_path, edited_text):
    model = whisper.load_model(MODEL_SIZE)
    result = model.transcribe(audio_path, word_timestamps=True)

    whisper_words = []
    for segment in result["segments"]:
        for word in segment.get("words", []):
            whisper_words.append({
                "start": word["start"],
                "end": word["end"],
                "word": word["word"].strip()
            })

    edited_words = edited_text.split()
    whisper_word_list = [w["word"] for w in whisper_words]

    matcher = difflib.SequenceMatcher(None, whisper_word_list, edited_words)

    aligned_word_data = []
    last_end = 0.0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():

        if tag == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                word_entry = whisper_words[i].copy()
                word_entry["word"] = edited_words[j]
                aligned_word_data.append(word_entry)
                last_end = word_entry["end"]

        elif tag == "replace":
            if i1 < len(whisper_words):
                start = whisper_words[i1]["start"]
                end = whisper_words[i2 - 1]["end"]
            else:
                start = last_end
                end = start + 0.5

            duration = max(end - start, 0.1)
            step = duration / max(j2 - j1, 1)

            for idx, j in enumerate(range(j1, j2)):
                new_start = start + idx * step
                new_end = new_start + step
                aligned_word_data.append({
                    "start": new_start,
                    "end": new_end,
                    "word": edited_words[j]
                })
                last_end = new_end

        elif tag == "insert":
            step = 0.25
            for j in range(j1, j2):
                new_start = last_end
                new_end = new_start + step
                aligned_word_data.append({
                    "start": new_start,
                    "end": new_end,
                    "word": edited_words[j]
                })
                last_end = new_end

        elif tag == "delete":
            continue

    return aligned_word_data


# ------------------------
# TRANSCRIBE
# ------------------------
def transcribe(audio_path):
    model = whisper.load_model(MODEL_SIZE)
    result = model.transcribe(audio_path, word_timestamps=True)

    edited_text = review_transcript(result["text"])

    if edited_text:
        word_data = align_to_edited_text(audio_path, edited_text)
        return edited_text, word_data

    word_data = []
    for segment in result["segments"]:
        for word in segment.get("words", []):
            word_data.append({
                "start": word["start"],
                "end": word["end"],
                "word": word["word"].strip()
            })

    return result["text"], word_data


# ------------------------
# GUI PLAYER
# ------------------------
class LyricPlayer:
    def __init__(self, root, fullscreen=False):
        self.root = root
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
        self.current_file = None
        self.current_hash = None

        tk.Button(root, text="Edit Transcript", command=self.edit_existing).pack(fill="x")

    def prepare_text(self, transcript):
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", transcript)

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
        self.current_file = filepath
        self.current_hash = get_file_hash(filepath)

        transcript, word_data = load_from_db(self.current_hash)

        if not transcript:
            transcript, word_data = transcribe(filepath)
            save_to_db(self.current_hash, os.path.basename(filepath), transcript, word_data)

        self.word_data = word_data
        self.prepare_text(transcript)

        threading.Thread(target=play_audio, args=(filepath,), daemon=True).start()
        threading.Thread(target=self.sync_words, daemon=True).start()

    def sync_words(self):
        start_time = time.perf_counter()
        for i, word in enumerate(self.word_data):
            while time.perf_counter() - start_time < word["start"]:
                time.sleep(0.001)
            self.root.after(0, self.highlight_word, i)

    def edit_existing(self):
        if not self.current_hash:
            messagebox.showinfo("No File", "Load a file first.")
            return

        transcript, _ = load_from_db(self.current_hash)
        edited_text = review_transcript(transcript)

        if edited_text:
            new_word_data = align_to_edited_text(self.current_file, edited_text)

            save_to_db(
                self.current_hash,
                os.path.basename(self.current_file),
                edited_text,
                new_word_data
            )

            self.word_data = new_word_data
            self.prepare_text(edited_text)

            messagebox.showinfo("Updated", "Transcript re-aligned with fresh Whisper pass.")


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