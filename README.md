# FEEL-D

This is the repository for the code used in the IPRO - Sports Lab

## Using vocals.py

needed imports:

`pip install openai-whisper sounddevice numpy scipy torch`

needed installs:

ffmpeg: <https://www.gyan.dev/ffmpeg/builds/>

### How to run

- Call the program
- Enter the file location of .wav file (without "")

### What it returns

- The program returns a JSON file with the SHA256 hash and transcript as a key value pair (no duplicates)
- Prints the transcript to console

## Using HapticMusicPlayer.py

- have not tested yet due to lack of device.
