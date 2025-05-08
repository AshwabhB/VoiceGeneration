import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import os
from datetime import datetime
from scipy.signal import lfilter
import parselmouth  # For advanced voice analysis features

# Create output directory
output_dir = "Audio_Analysis_Results"
os.makedirs(output_dir, exist_ok=True)

def load_audio(file_path):
    print(f"Loading audio file: {file_path}")
    y, sr = librosa.load(file_path)
    return y, sr

def extract_linguistic_features(y, sr):
    print("Extracting linguistic features...")
    # Extract pitch (fundamental frequency)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitches[magnitudes > np.max(magnitudes)/10])
    
    # Extract duration
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Extract stress (using RMS energy as a proxy)
    rms = librosa.feature.rms(y=y)[0]
    stress_mean = np.mean(rms)
    
    # Extract speech rate (using onset detection)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    speech_rate = len(onset_frames) / duration
    
    return {
        'pitch': pitch_mean,
        'duration': duration,
        'stress': stress_mean,
        'speech_rate': speech_rate
    }

def extract_advanced_voice_features(y, sr):
    print("Extracting advanced voice features...")
    
    # Fundamental Frequency (F0) and Formants
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    f0 = pitches[magnitudes > np.max(magnitudes)/10]
    f0_mean = np.mean(f0)
    f0_std = np.std(f0)
    
    # Formants using LPC with error handling
    n_coeff = 2 + int(sr/1000)  # Rule of thumb for LPC coefficients
    try:
        lpc_coeffs = librosa.lpc(y, order=n_coeff)
        # Check for invalid values
        if np.any(np.isnan(lpc_coeffs)) or np.any(np.isinf(lpc_coeffs)):
            raise ValueError("Invalid LPC coefficients")
        roots = np.roots(lpc_coeffs)
        roots = roots[roots.imag > 0]
        formants = np.sort(np.arctan2(roots.imag, roots.real) * (sr/(2*np.pi)))
        formants = formants[formants > 0]
    except (ValueError, np.linalg.LinAlgError):
        # If LPC fails, use default values
        formants = np.array([500, 1500, 2500])  # Default formant frequencies
    
    # Spectral Features
    S = np.abs(librosa.stft(y))
    spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
    spectral_flatness = librosa.feature.spectral_flatness(S=S)[0]
    
    # Voice Quality Features
    harmonic, percussive = librosa.effects.hpss(y)
    hnr = np.mean(harmonic) / (np.mean(percussive) + 1e-6)
    
    # Jitter and Shimmer (using parselmouth)
    try:
        sound = parselmouth.Sound(y, sr)
        jitter = sound.to_jitter()
        shimmer = sound.to_shimmer()
    except:
        jitter = 0
        shimmer = 0
    
    # Energy Dynamics
    rms = librosa.feature.rms(y=y)[0]
    energy_mean = np.mean(rms)
    energy_std = np.std(rms)
    
    # Duration-based Features
    duration = librosa.get_duration(y=y, sr=sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    syllable_rate = len(onset_frames) / duration
    
    return {
        'f0': {
            'mean': f0_mean,
            'std': f0_std
        },
        'formants': {
            'f1': formants[0] if len(formants) > 0 else 500,
            'f2': formants[1] if len(formants) > 1 else 1500,
            'f3': formants[2] if len(formants) > 2 else 2500
        },
        'spectral': {
            'centroid': np.mean(spectral_centroid),
            'bandwidth': np.mean(spectral_bandwidth),
            'rolloff': np.mean(spectral_rolloff),
            'flatness': np.mean(spectral_flatness)
        },
        'voice_quality': {
            'hnr': hnr,
            'jitter': jitter,
            'shimmer': shimmer
        },
        'energy': {
            'mean': energy_mean,
            'std': energy_std
        },
        'duration': {
            'total': duration,
            'syllable_rate': syllable_rate
        }
    }

def extract_features_for_analysis(y, sr):
    print("Extracting linguistic features for analysis...")
    # Extract pitch (fundamental frequency)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitches[magnitudes > np.max(magnitudes)/10])
    
    # Extract duration
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Extract stress (using RMS energy as a proxy)
    rms = librosa.feature.rms(y=y)[0]
    stress_mean = np.mean(rms)
    
    # Extract speech rate (using onset detection)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    speech_rate = len(onset_frames) / duration
    
    # Extract voice quality features
    harmonic, percussive = librosa.effects.hpss(y)
    harmonic_ratio = np.mean(harmonic) / (np.mean(harmonic) + np.mean(percussive) + 1e-6)
    
    # Extract advanced voice features
    advanced_features = extract_advanced_voice_features(y, sr)
    
    return {
        'linguistic': {
            'pitch': pitch_mean,
            'duration': duration,
            'stress': stress_mean,
            'speech_rate': speech_rate
        },
        'voice_quality': {
            'harmonic_ratio': harmonic_ratio
        },
        'advanced_features': advanced_features
    }

def extract_features_for_visualization(y, sr):
    print("Extracting audio features for visualization...")
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # Extract chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    # Extract spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    # Extract tonnetz
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    
    return {
        'mfccs': mfccs,
        'chroma': chroma,
        'contrast': contrast,
        'tonnetz': tonnetz
    }

def plot_waveform(y, sr, title, filename):
    print(f"Generating waveform plot for {title}...")
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f'Waveform - {title}')
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_spectrogram(y, sr, title, filename):
    print(f"Generating spectrogram for {title}...")
    plt.figure(figsize=(12, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram - {title}')
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_mfcc(mfccs, title, filename):
    print(f"Generating MFCC plot for {title}...")
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title(f'MFCC - {title}')
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_pitch_contour(y, sr, title, filename):
    print(f"Generating pitch contour for {title}...")
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_times = librosa.times_like(pitches)
    
    plt.figure(figsize=(12, 4))
    plt.plot(pitch_times, pitches[0], label='Pitch')
    plt.title(f'Pitch Contour - {title}')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def calculate_similarity(features1, features2):
    print("Calculating similarity between audio files...")
    similarity_scores = {}
    
    # Calculate similarity for linguistic features
    linguistic1 = features1['linguistic']
    linguistic2 = features2['linguistic']
    
    # Weights for linguistic features
    linguistic_weights = {
        'pitch': 0.15,      # Voice pitch
        'duration': 0.05,   # Timing
        'stress': 0.05,     # Emphasis
        'speech_rate': 0.05 # Rhythm
    }
    
    for feature_name in linguistic1.keys():
        if feature_name == 'duration':
            diff = abs(linguistic1[feature_name] - linguistic2[feature_name]) / linguistic1[feature_name]
        else:
            diff = abs(linguistic1[feature_name] - linguistic2[feature_name]) / (linguistic1[feature_name] + 1e-6)
        similarity_scores[f'linguistic_{feature_name}'] = 1 - min(diff, 1.0)  # Ensure diff doesn't exceed 1.0
    
    # Add voice quality similarity
    voice_quality_diff = abs(features1['voice_quality']['harmonic_ratio'] - features2['voice_quality']['harmonic_ratio'])
    similarity_scores['voice_quality'] = 1 - min(voice_quality_diff, 1.0)  # Ensure diff doesn't exceed 1.0
    
    # Add MFCC similarity
    mfcc1 = features1.get('mfccs', np.zeros((13, 1)))
    mfcc2 = features2.get('mfccs', np.zeros((13, 1)))
    mfcc_diff = np.mean(np.abs(mfcc1 - mfcc2))
    similarity_scores['mfccs'] = 1 - min(mfcc_diff, 1.0)  # Ensure diff doesn't exceed 1.0
    
    # Add advanced features similarity
    adv1 = features1.get('advanced_features', {})
    adv2 = features2.get('advanced_features', {})
    
    # F0 similarity
    f0_diff = abs(adv1.get('f0', {}).get('mean', 0) - adv2.get('f0', {}).get('mean', 0))
    similarity_scores['f0'] = 1 - min(f0_diff, 1.0)  # Ensure diff doesn't exceed 1.0
    
    # Formants similarity
    for i in range(1, 4):
        formant_diff = abs(adv1.get('formants', {}).get(f'f{i}', 0) - adv2.get('formants', {}).get(f'f{i}', 0))
        similarity_scores[f'formant_{i}'] = 1 - min(formant_diff, 1.0)  # Ensure diff doesn't exceed 1.0
    
    # Spectral features similarity
    spectral_features = ['centroid', 'bandwidth', 'rolloff', 'flatness']
    for feature in spectral_features:
        diff = abs(adv1.get('spectral', {}).get(feature, 0) - adv2.get('spectral', {}).get(feature, 0))
        similarity_scores[f'spectral_{feature}'] = 1 - min(diff, 1.0)  # Ensure diff doesn't exceed 1.0
    
    # Voice quality features similarity
    quality_features = ['hnr', 'jitter', 'shimmer']
    for feature in quality_features:
        diff = abs(adv1.get('voice_quality', {}).get(feature, 0) - adv2.get('voice_quality', {}).get(feature, 0))
        similarity_scores[f'quality_{feature}'] = 1 - min(diff, 1.0)  # Ensure diff doesn't exceed 1.0
    
    # Energy features similarity
    energy_diff = abs(adv1.get('energy', {}).get('mean', 0) - adv2.get('energy', {}).get('mean', 0))
    similarity_scores['energy'] = 1 - min(energy_diff, 1.0)  # Ensure diff doesn't exceed 1.0
    
    # Duration features similarity
    duration_diff = abs(adv1.get('duration', {}).get('syllable_rate', 0) - adv2.get('duration', {}).get('syllable_rate', 0))
    similarity_scores['syllable_rate'] = 1 - min(duration_diff, 1.0)  # Ensure diff doesn't exceed 1.0
    
    return similarity_scores

def calculate_overall_similarity(similarity_scores):
    # Weights normalized to sum to 1.0
    weights = {
        'mfccs': 0.15,          # Voice characteristics
        'f0': 0.1,              # Fundamental frequency
        'formant_1': 0.05,      # Formants
        'formant_2': 0.05,
        'formant_3': 0.05,
        'spectral_centroid': 0.05,  # Spectral features
        'spectral_bandwidth': 0.05,
        'spectral_rolloff': 0.05,
        'spectral_flatness': 0.05,
        'quality_hnr': 0.05,    # Voice quality
        'quality_jitter': 0.05,
        'quality_shimmer': 0.05,
        'energy': 0.05,         # Energy
        'syllable_rate': 0.05,  # Duration
        'voice_quality': 0.05,  # Basic voice quality
        'linguistic_pitch': 0.05,  # Linguistic features
        'linguistic_duration': 0.05,
        'linguistic_stress': 0.05,
        'linguistic_speech_rate': 0.05
    }
    
    # Calculate weighted average
    overall_score = 0
    for feature, score in similarity_scores.items():
        if feature in weights:
            score_value = float(score) if isinstance(score, np.ndarray) else score
            overall_score += score_value * weights[feature]
    
    return min(float(overall_score), 1.0)  # Ensure final score doesn't exceed 1.0

def generate_detailed_observations(original_features, tts_features, generated_features, tts_similarity, generated_similarity):
    observations = []
    
    # Compare linguistic features
    def compare_linguistic_features(features1, features2, name1, name2):
        obs = []
        for feature, value1 in features1['linguistic'].items():
            value2 = features2['linguistic'][feature]
            diff = abs(value1 - value2) / (value1 + 1e-6)  # Relative difference
            if diff > 0.1:  # 10% difference threshold for more sensitive detection
                obs.append(f"- {feature}: {name2} differs by {diff*100:.1f}% from {name1}")
        return obs
    
    # TTS vs Original observations
    tts_obs = []
    tts_obs.extend(compare_linguistic_features(original_features, tts_features, "Original", "TTS"))
    
    # Add observations about voice quality
    voice_quality_diff = abs(original_features['voice_quality']['harmonic_ratio'] - tts_features['voice_quality']['harmonic_ratio'])
    if voice_quality_diff > 0.1:
        tts_obs.append(f"- voice_quality: TTS differs by {voice_quality_diff*100:.1f}% from Original")
    
    # Generated vs Original observations
    gen_obs = []
    gen_obs.extend(compare_linguistic_features(original_features, generated_features, "Original", "Generated"))
    
    # Add observations about voice quality
    voice_quality_diff = abs(original_features['voice_quality']['harmonic_ratio'] - generated_features['voice_quality']['harmonic_ratio'])
    if voice_quality_diff > 0.1:
        gen_obs.append(f"- voice_quality: Generated differs by {voice_quality_diff*100:.1f}% from Original")
    
    return tts_obs, gen_obs

def main():
    print("Starting audio analysis...")
    
    # Load audio files
    original_y, original_sr = load_audio("Module4AudioRegenerator/Analysis/Audios/Original.wav")
    tts_y, tts_sr = load_audio("Module4AudioRegenerator/Analysis/Audios/TTS.wav")
    generated_y, generated_sr = load_audio("Module4AudioRegenerator/Analysis/Audios/Generated.wav")
    
    # Extract features for analysis
    original_features = extract_features_for_analysis(original_y, original_sr)
    tts_features = extract_features_for_analysis(tts_y, tts_sr)
    generated_features = extract_features_for_analysis(generated_y, generated_sr)
    
    # Generate plots for each audio file
    for audio_name, audio_data, sr in [
        ("Original", original_y, original_sr),
        ("TTS", tts_y, tts_sr),
        ("Generated", generated_y, generated_sr)
    ]:
        plot_waveform(audio_data, sr, audio_name, f"{audio_name}_waveform.png")
        plot_spectrogram(audio_data, sr, audio_name, f"{audio_name}_spectrogram.png")
        vis_features = extract_features_for_visualization(audio_data, sr)
        plot_mfcc(vis_features['mfccs'], audio_name, f"{audio_name}_mfcc.png")
        plot_pitch_contour(audio_data, sr, audio_name, f"{audio_name}_pitch.png")
    
    # Calculate similarity scores
    tts_similarity = calculate_similarity(original_features, tts_features)
    generated_similarity = calculate_similarity(original_features, generated_features)
    
    # Calculate overall similarity scores
    tts_overall = calculate_overall_similarity(tts_similarity)
    generated_overall = calculate_overall_similarity(generated_similarity)
    
    # Generate detailed observations
    tts_observations, generated_observations = generate_detailed_observations(
        original_features, tts_features, generated_features, tts_similarity, generated_similarity
    )
    
    # Generate report
    print("Generating analysis report...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"audio_analysis_report_{timestamp}.md")
    
    def format_score(score):
        if isinstance(score, np.ndarray):
            return float(score)
        return score
    
    with open(report_file, 'w') as f:
        f.write("# Audio Analysis Report\n\n")
        
        # Write linguistic features
        f.write("## Linguistic Features\n\n")
        f.write("### Original Audio\n")
        for feature, value in original_features['linguistic'].items():
            f.write(f"- {feature}: {value:.4f}\n")
        
        f.write("\n### TTS Audio\n")
        for feature, value in tts_features['linguistic'].items():
            f.write(f"- {feature}: {value:.4f}\n")
        
        f.write("\n### Generated Audio\n")
        for feature, value in generated_features['linguistic'].items():
            f.write(f"- {feature}: {value:.4f}\n")
        
        f.write("\n## Similarity Scores\n\n")
        f.write("### TTS vs Original\n")
        for feature, score in tts_similarity.items():
            score_value = format_score(score)
            f.write(f"- {feature}: {score_value:.4f}\n")
        f.write(f"\n**Overall Similarity Score**: {tts_overall:.4f}\n")
        
        f.write("\n### Generated vs Original\n")
        for feature, score in generated_similarity.items():
            score_value = format_score(score)
            f.write(f"- {feature}: {score_value:.4f}\n")
        f.write(f"\n**Overall Similarity Score**: {generated_overall:.4f}\n")
        
        f.write("\n## Overall Comparison\n\n")
        f.write(f"- TTS vs Original Overall Similarity: {tts_overall:.4f}\n")
        f.write(f"- Generated vs Original Overall Similarity: {generated_overall:.4f}\n")
        if tts_overall > generated_overall:
            f.write("\n**Conclusion**: TTS audio is more similar to the original audio.\n")
        else:
            f.write("\n**Conclusion**: Generated audio is more similar to the original audio.\n")
        
        f.write("\n## Detailed Observations\n\n")
        f.write("### TTS vs Original Comparison\n")
        if tts_observations:
            for obs in tts_observations:
                f.write(f"{obs}\n")
        else:
            f.write("- No significant differences detected\n")
        
        f.write("\n### Generated vs Original Comparison\n")
        if generated_observations:
            for obs in generated_observations:
                f.write(f"{obs}\n")
        else:
            f.write("- No significant differences detected\n")
    
    print(f"Analysis complete! Results saved in {output_dir}")
    print(f"Report generated at: {report_file}")
    print(f"\nOverall Similarity Scores:")
    print(f"TTS vs Original: {tts_overall:.4f}")
    print(f"Generated vs Original: {generated_overall:.4f}")

if __name__ == "__main__":
    main() 