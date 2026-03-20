# melody_visualizer.py
import tkinter as tk
import time

class MelodyVisualizer:
    def __init__(self, root, melody_data):
        self.root = root
        self.melody_data = melody_data

        self.canvas = tk.Canvas(root, bg="black", height=200)
        self.canvas.pack(fill="x")

        self.max_freq = max([m["freq"] for m in melody_data]) if melody_data else 1000
        self.min_freq = min([m["freq"] for m in melody_data]) if melody_data else 80

        self.start_time = None

    def draw_point(self, t, freq):
        width = self.canvas.winfo_width()
        x = (t * 100) % width

        y_range = self.max_freq - self.min_freq
        y = 200 - ((freq - self.min_freq) / y_range) * 200

        r = 3
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="yellow", outline="")

    def start(self):
        self.start_time = time.perf_counter()
        self.update()

    def update(self):
        now = time.perf_counter() - self.start_time

        for note in self.melody_data:
            if abs(note["time"] - now) < 0.02:
                self.draw_point(note["time"], note["freq"])

        self.root.after(10, self.update)
