import HapticMusicPlayer

def main():
    # Create a HapticMusicPlayer object
    player = HapticMusicPlayer(file="song.mp3")
    
    # Play the song
    player.play()

if __name__ == "__main__":
    main()