import time
import threading
import numpy as np
import librosa
import sounddevice as sd
import RPi.GPIO as GPIO

class HapticMusicPlayer:
    def __init__(
        self,
        audio_file,
        gpio_pin=18,
        pwm_freq=200,
        beat_duty=75,
        melody_min_duty=20,
        melody_max_duty=60,
        beat_pulse_duration=0.08
    ):
        self.audio_file = audio_file
        self.gpio_pin = gpio_pin
        self.pwm_freq = pwm_freq
        self.beat_duty = beat_duty
        self.melody_min_duty = melody_min_duty
        self.melody_max_duty = melody_max_duty
        self.beat_pulse_duration = beat_pulse_duration

        self._setup_gpio()
        self._analyze_audio()

    # -----------------------
    # Hardware setup
    # -----------------------
    def _setup_gpio(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.gpio_pin, GPIO.OUT)
        self.pwm = GPIO.PWM(self.gpio_pin, self.pwm_freq)
        self.pwm.start(0)

    # -----------------------
    # Audio analysis
    # -----------------------
    def _analyze_audio(self):
        print("Loading audio...")
        self.y, self.sr = librosa.load(self.audio_file, sr=None, mono=True)

        print("Detecting beats...")
        _, beat_frames = librosa.beat.beat_track(y=self.y, sr=self.sr)
        self.beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)

        print("Extracting melody...")
        self.frame_length = 2048
        self.hop_length = 256

        pitches = librosa.yin(
            self.y,
            fmin=80,
            fmax=800,
            sr=self.sr,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )

        times = librosa.frames_to_time(
            np.arange(len(pitches)),
            sr=self.sr,
            hop_length=self.hop_length
        )

        # Remove unvoiced frames
        pitches = np.nan_to_num(pitches)

        # Normalize pitch → vibration intensity
        voiced = pitches[pitches > 0]
        p_min, p_max = voiced.min(), voiced.max()

        self.melody_times = times
        self.melody_duty = np.interp(
            pitches,
            [p_min, p_max],
            [self.melody_min_duty, self.melody_max_duty]
        )

    # -----------------------
    # Melody vibration loop
    # -----------------------
    def _melody_loop(self, start_time):
        for t, duty in zip(self.melody_times, self.melody_duty):
            target = start_time + t
            now = time.monotonic()
            if target > now:
                time.sleep(target - now)

            self.pwm.ChangeDutyCycle(duty)

    # -----------------------
    # Beat pulse loop
    # -----------------------
    def _beat_loop(self, start_time):
        for bt in self.beat_times:
            target = start_time + bt
            now = time.monotonic()
            if target > now:
                time.sleep(target - now)

            self.pwm.ChangeDutyCycle(self.beat_duty)
            time.sleep(self.beat_pulse_duration)

    # -----------------------
    # Public API
    # -----------------------
    def play(self):
        try:
            print("Starting haptic playback...")
            start_time = time.monotonic()

            melody_thread = threading.Thread(
                target=self._melody_loop,
                args=(start_time,),
                daemon=True
            )

            beat_thread = threading.Thread(
                target=self._beat_loop,
                args=(start_time,),
                daemon=True
            )

            melody_thread.start()
            beat_thread.start()

            sd.play(self.y, self.sr)
            sd.wait()

        finally:
            self.stop()

    def stop(self):
        self.pwm.ChangeDutyCycle(0)
        self.pwm.stop()
        GPIO.cleanup()
        print("Playback finished.")
