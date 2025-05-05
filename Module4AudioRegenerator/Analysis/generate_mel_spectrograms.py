import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def generate_visualizations(audio_path, output_dir, filename):
    y, sr = librosa.load(audio_path)
    
    plt.figure(figsize=(15, 6))
    
    # Waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    # Mel Spectrogram
    plt.subplot(2, 1, 2)
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
    librosa.display.specshow(mel_spect_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    
    # Adjust layout and save
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_visualization.png")
    plt.savefig(output_path)
    plt.close()

def main():
    output_dir = 'Audio_Visualizations'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    audio_dir = 'Audios'
    for filename in os.listdir(audio_dir):
        if filename.endswith(('.wav', '.mp3', '.ogg', '.flac')):
            audio_path = os.path.join(audio_dir, filename)
            
            print(f"Processing {filename}...")
            try:
                generate_visualizations(audio_path, output_dir, filename)
                print(f"Successfully generated visualizations for {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    main() 