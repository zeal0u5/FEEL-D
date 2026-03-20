# motor.py
import RPi.GPIO as GPIO
import time

MOTOR_PIN = 18

def init_motor():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(MOTOR_PIN, GPIO.OUT)
    pwm = GPIO.PWM(MOTOR_PIN, 100)
    pwm.start(0)
    return pwm

def set_motor_frequency(pwm, freq):
    freq = max(20, min(freq, 2000))
    pwm.ChangeFrequency(freq)
    pwm.ChangeDutyCycle(50)

def play_melody(melody_data):
    pwm = init_motor()
    start = time.perf_counter()

    for note in melody_data:
        while time.perf_counter() - start < note["time"]:
            time.sleep(0.001)
        set_motor_frequency(pwm, note["freq"])
