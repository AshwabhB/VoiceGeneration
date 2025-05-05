# Audio Analysis Features by Version with Weights

## 1. Basic Version (audio_comparison_analysis.py)
Core Features with Weights:
- Linguistic Features:
  * Pitch (0.15 weight)
  * Duration (0.05 weight)
  * Stress (0.05 weight)
  * Speech rate (0.05 weight)
- Voice Quality:
  * Harmonic ratio (0.1 weight)
- Spectral Features:
  * MFCCs (0.25 weight)
  * Spectral centroid (0.1 weight)
  * Spectral rolloff (0.1 weight)
  * Spectral contrast (0.1 weight)
- Rhythm Features:
  * Tempo (0.05 weight)

## 2. Linguistic-Focused Version (audio_comparison_analysis_linguistic_features.py)
Enhanced Linguistic Features with Weights:
- Linguistic Features:
  * Pitch (0.36 weight)
  * Duration (0.18 weight)
  * Stress (0.18 weight)
  * Speech rate (0.18 weight)
- Voice Quality:
  * Harmonic ratio (0.1 weight)

## 3. Sophisticated Version (audio_comparison_analysis copy_sophisticated.py)
Comprehensive Feature Set with Weights:
- Advanced Voice Features:
  * Fundamental Frequency (F0) analysis (0.1 weight)
  * Formants (F1, F2, F3) (0.15 weight total, 0.05 each)
  * Jitter and Shimmer (0.15 weight total)
  * Harmonic-to-Noise Ratio (HNR) (0.05 weight)
- Spectral Analysis:
  * Spectral centroid (0.05 weight)
  * Spectral bandwidth (0.05 weight)
  * Spectral rolloff (0.05 weight)
  * Spectral flatness (0.05 weight)
- Energy Dynamics:
  * RMS energy (0.05 weight)
  * Energy standard deviation (0.05 weight)
- Duration Features:
  * Total duration (0.05 weight)
  * Syllable rate (0.05 weight)
- Linguistic Features:
  * Pitch (0.05 weight)
  * Duration (0.05 weight)
  * Stress (0.05 weight)
  * Speech rate (0.05 weight)
- Voice Quality:
  * Harmonic ratio (0.05 weight)
  * Jitter (0.05 weight)
  * Shimmer (0.05 weight)

Weights are normalized to sum to 1.0 in each version

Ling file     

weights = {
         'linguistic_pitch': 0.4,
         'linguistic_duration': 0.2,
         'linguistic_stress': 0.2,
         'linguistic_speech_rate': 0.2,
         'voice_quality': 0.1
     }




non ling file

     weights = {
         'mfccs': 0.25,          # Voice characteristics
         'spectral_centroid': 0.1,  # Timbre
         'spectral_rolloff': 0.1,   # Timbre
         'spectral_contrast': 0.1,  # Timbre
         'tempo': 0.05,            # Rhythm
         'harmonic_ratio': 0.1,     # Voice quality
         'linguistic_pitch': 0.15,  # Voice pitch
         'linguistic_duration': 0.05,  # Timing
         'linguistic_stress': 0.05,    # Emphasis
         'linguistic_speech_rate': 0.05  # Rhythm
     }