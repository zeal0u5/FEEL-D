# melody.py
import librosa
import numpy as np

def extract_melody(audio_path):
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    f0 = librosa.yin(
        y,
        fmin=80,
        fmax=1000,
        sr=sr
    )

    frame_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr)

    melody = []
    for t, freq in zip(frame_times, f0):
        if not np.isnan(freq):
            melody.append({"time": float(t), "freq": float(freq)})

    return melody
