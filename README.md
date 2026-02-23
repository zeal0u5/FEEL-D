# FEEL-D

This is the repository for the code used in the IPRO - Sports Lab

## Using vocals.py

needed imports:

`pip install openai-whisper sounddevice numpy scipy torch sqlite3 tkinter`

needed installs:

ffmpeg: <https://www.gyan.dev/ffmpeg/builds/>

### How to run

On computer:
`python lyric_player.py`

On Raspberry Pi (probably works):
`python lyric_player.py --fullscreen`

Needed on raspberry Pi:
`sudo apt install python3-tk`
`pip3 install openai-whisper sounddevice scipy numpy`

### What it returns

- The program returns a JSON file with the SHA256 hash and transcript as a key value pair (no duplicates)
- Prints the transcript to console

### General Code Formatting

Audio File
    ↓
SHA256
    ↓
Database Lookup
    ↓
IF FOUND:
    Play audio
    Print stored timestamps
ELSE:
    Run Whisper once
    Store segments

### Notes

- ±20–40ms drift
- After caching → "instant" playback
- fullscreen resolution
- Only display 1–2 lines at a time
- Highlight current word in color

## Using HapticMusicPlayer.py

- have not tested yet due to lack of device.
