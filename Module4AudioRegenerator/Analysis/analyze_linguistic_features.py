import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

def extract_linguistic_features(audio_path):
    # Load audio file
    y, sr = librosa.load(audio_path)
    
    # Extract pitch (fundamental frequency) using librosa's pitch tracking
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    
    # Get the dominant pitch at each time frame
    pitch = []
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch.append(pitches[index, i])
    pitch = np.array(pitch)
    
    # Calculate speech rate (syllables per second)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    speech_rate = len(beats) / librosa.get_duration(y=y, sr=sr)
    
    # Calculate intensity (RMS energy)
    rms = librosa.feature.rms(y=y)[0]
    intensity = np.mean(rms)
    
    # Calculate spectral centroid (brightness)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    brightness = np.mean(spectral_centroid)
    
    # Calculate zero-crossing rate (roughness)
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    roughness = np.mean(zcr)
    
    # Calculate spectral rolloff (timbre)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    timbre = np.mean(rolloff)
    
    # Calculate MFCCs (timbre characteristics)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    
    return {
        'pitch': pitch,
        'speech_rate': speech_rate,
        'intensity': intensity,
        'brightness': brightness,
        'roughness': roughness,
        'timbre': timbre,
        'mfccs': mfcc_mean
    }

def plot_comparative_features(original_file, generated_file, tts_file, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define colors for each file
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    # Create figure with subplots
    plt.style.use('default')  # Use default matplotlib style
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['grid.color'] = '#d3d3d3'
    plt.rcParams['grid.alpha'] = 0.3
    
    # Create a 3x2 grid for the plots
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(3, 2)
    
    # Extract features for each file
    original_features = extract_linguistic_features(original_file)
    generated_features = extract_linguistic_features(generated_file)
    tts_features = extract_linguistic_features(tts_file)
    
    # Plot Pitch Comparison
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(original_features['pitch'], color=colors[0], linewidth=2, label='Original')
    ax.plot(generated_features['pitch'], color=colors[1], linewidth=2, label='Generated')
    ax.plot(tts_features['pitch'], color=colors[2], linewidth=2, label='TTS')
    ax.set_title('Pitch Comparison', fontsize=14, pad=20)
    ax.set_xlabel('Time Frame', fontsize=12)
    ax.set_ylabel('Pitch (Hz)', fontsize=12)
    ax.grid(True)
    ax.legend(fontsize=10)
    
    # Plot Speech Rate Comparison
    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(3)
    width = 0.6
    ax.bar(x[0], original_features['speech_rate'], width, 
           label='Original', color=colors[0], alpha=0.8)
    ax.bar(x[1], generated_features['speech_rate'], width, 
           label='Generated', color=colors[1], alpha=0.8)
    ax.bar(x[2], tts_features['speech_rate'], width, 
           label='TTS', color=colors[2], alpha=0.8)
    ax.set_title('Speech Rate Comparison', fontsize=14, pad=20)
    ax.set_xlabel('Audio Type', fontsize=12)
    ax.set_ylabel('Speech Rate (syllables/second)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['Original', 'Generated', 'TTS'], fontsize=12)
    ax.grid(True)
    ax.legend(fontsize=10)
    
    # Plot Intensity Comparison
    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(3)
    width = 0.6
    ax.bar(x[0], original_features['intensity'], width, 
           label='Original', color=colors[0], alpha=0.8)
    ax.bar(x[1], generated_features['intensity'], width, 
           label='Generated', color=colors[1], alpha=0.8)
    ax.bar(x[2], tts_features['intensity'], width, 
           label='TTS', color=colors[2], alpha=0.8)
    ax.set_title('Intensity Comparison', fontsize=14, pad=20)
    ax.set_xlabel('Audio Type', fontsize=12)
    ax.set_ylabel('RMS Energy', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['Original', 'Generated', 'TTS'], fontsize=12)
    ax.grid(True)
    ax.legend(fontsize=10)
    
    # Plot Brightness Comparison
    ax = fig.add_subplot(gs[1, 1])
    x = np.arange(3)
    width = 0.6
    ax.bar(x[0], original_features['brightness'], width, 
           label='Original', color=colors[0], alpha=0.8)
    ax.bar(x[1], generated_features['brightness'], width, 
           label='Generated', color=colors[1], alpha=0.8)
    ax.bar(x[2], tts_features['brightness'], width, 
           label='TTS', color=colors[2], alpha=0.8)
    ax.set_title('Spectral Brightness Comparison', fontsize=14, pad=20)
    ax.set_xlabel('Audio Type', fontsize=12)
    ax.set_ylabel('Spectral Centroid (Hz)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['Original', 'Generated', 'TTS'], fontsize=12)
    ax.grid(True)
    ax.legend(fontsize=10)
    
    # Plot Roughness Comparison
    ax = fig.add_subplot(gs[2, 0])
    x = np.arange(3)
    width = 0.6
    ax.bar(x[0], original_features['roughness'], width, 
           label='Original', color=colors[0], alpha=0.8)
    ax.bar(x[1], generated_features['roughness'], width, 
           label='Generated', color=colors[1], alpha=0.8)
    ax.bar(x[2], tts_features['roughness'], width, 
           label='TTS', color=colors[2], alpha=0.8)
    ax.set_title('Roughness Comparison', fontsize=14, pad=20)
    ax.set_xlabel('Audio Type', fontsize=12)
    ax.set_ylabel('Zero-Crossing Rate', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['Original', 'Generated', 'TTS'], fontsize=12)
    ax.grid(True)
    ax.legend(fontsize=10)
    
    # Plot MFCC Comparison (first 5 coefficients)
    ax = fig.add_subplot(gs[2, 1])
    mfcc_coeffs = np.arange(5)
    width = 0.25
    ax.bar(mfcc_coeffs - width, original_features['mfccs'][:5], width, 
           label='Original', color=colors[0], alpha=0.8)
    ax.bar(mfcc_coeffs, generated_features['mfccs'][:5], width, 
           label='Generated', color=colors[1], alpha=0.8)
    ax.bar(mfcc_coeffs + width, tts_features['mfccs'][:5], width, 
           label='TTS', color=colors[2], alpha=0.8)
    ax.set_title('MFCC Comparison (First 5 Coefficients)', fontsize=14, pad=20)
    ax.set_xlabel('MFCC Coefficient', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_xticks(mfcc_coeffs)
    ax.set_xticklabels([f'MFCC {i+1}' for i in range(5)], fontsize=12)
    ax.grid(True)
    ax.legend(fontsize=10)
    
    # Add main title
    fig.suptitle('Linguistic Features Comparison', fontsize=16, y=0.95)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'linguistic_features_comparison.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Define file paths
    original_file = 'Audios/Original.wav'
    generated_file = 'Audios/Generated.wav'
    tts_file = 'Audios/TTS.wav'
    output_dir = 'Linguistic_Features_Visualizations'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Check if all files exist
    for file_path in [original_file, generated_file, tts_file]:
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return
    
    # Generate visualizations
    plot_comparative_features(original_file, generated_file, tts_file, output_dir)
    print("Visualizations have been generated and saved in the Linguistic_Features_Visualizations directory.")

if __name__ == "__main__":
    main() 