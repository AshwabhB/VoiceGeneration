# Audio Processor README

## Overview
This file contains the audio processing utilities for the voice conversion system, including feature extraction, audio enhancement, and processing functions.

## Main Components

### Classes
1. `AudioProcessor`

### Functions
1. `blend_statistics`

## Detailed Component Descriptions

### AudioProcessor Class
A comprehensive audio processing utility class with various methods for audio manipulation and feature extraction.
Parameters:
      -   `sample_rate`: Audio sample rate
      -   `n_mels: Number` of mel bands
      -   `hop_length`: Number of samples between frames
      -   `win_length`: Window size
      -   `fmin`: Minimum frequency
      -   `fmax`: Maximum frequency
      -   `griffin_lim_iters`: Number of Griffin-Lim iterations for phase reconstruction (This algorithm is used when converting a mel spectrogram back to audio in the mel_to_audio method.)

#### Methods:

1. Audio Loading and Saving:
   - `load_audio`: Loads and normalizes audio files
   - `save_audio`: Saves audio with proper normalization

2. Spectrogram Processing:
   - `audio_to_mel`: Converts audio to mel spectrogram
   - `mel_to_audio`: Converts mel spectrogram back to audio with improved phase reconstruction

3. Feature Extraction:
   - `extract_voice_stats`: Extracts voice statistics from audio for better matching
   - `extract_pitch`: Extracts pitch (F0) contour. Returns: pitch_contour (Array of pitch values), voiced_frames (Boolean array indicating voiced frames)
   - `extract_duration`: Extracts duration information. 
   - `extract_stress`: Extracts stress patterns. 
   - `extract_linguistic_features`: Extracts all linguistic features

4. Audio Enhancement:
   - `apply_lowpass_filter`: Applies low-pass filter to reduce high-frequency noise
   - `enhance_audio`: Applies multiple enhancements to improve audio quality

5. Chunk Processing:
   - `split_audio_into_chunks`: Splits audio into overlapping chunks for efficient processing
   - `combine_chunks`: Combines processed chunks with crossfading for smooth transitions

#### Linguistic Features Extracted:

1. Pitch Features:
   - Mean pitch
   - Pitch standard deviation
   - Pitch range
   - Voiced ratio

2. Duration Features:
   - Total duration
   - Speech duration
   - Silence duration
   - Speech rate

3. Stress Features:
   - Peak count
   - Mean peak height
   - Standard deviation of peak height
   - Mean peak interval
   - Standard deviation of peak interval

### blend_statistics Function
Combines (blends) statistics from a source voice and a target voice to create more natural voice conversion.

#### Parameters:
- `source_stats`: Statistics from the source voice
- `target_stats`: Statistics from the target voice
- `blend_ratio`: A value between 0.0 and 1.0 that determines the weight of the target statistics
  - 0.0 means use only source statistics
  - 1.0 means use only target statistics
  - 0.7 (default) means 70% target, 30% source


#### How it works:
- For each statistic in the source voice:
  - If the same statistic exists in the target voice:
    - Creates a weighted average using the blend_ratio
    - Formula: `(1 - blend_ratio) * source_value + blend_ratio * target_value`
  - If the statistic doesn't exist in the target voice:
    - Keeps the source value unchanged

#### Example:
```python
source_stats = {'mean_pitch': 200, 'std_pitch': 30}
target_stats = {'mean_pitch': 300, 'std_pitch': 40}
blend_ratio = 0.7

# Result:
blended_stats = {
    'mean_pitch': (0.3 * 200) + (0.7 * 300) = 270,
    'std_pitch': (0.3 * 30) + (0.7 * 40) = 37
}
```

#### Use Cases:
- Voice conversion: Creating a voice that has characteristics of both source and target
- Style transfer: Gradually transitioning from one voice style to another
- Natural voice synthesis: Creating more natural-sounding voices by blending characteristics

### Butterworth Low-Pass Filter
A Butterworth low-pass filter is a type of signal processing filter that allows frequencies below a certain cutoff frequency to pass through while attenuating (reducing) frequencies above that cutoff. It is implemented in the `apply_lowpass_filter` method of the `AudioProcessor` class.

Key characteristics:
1. **Smooth Frequency Response**: Maximally flat frequency response in the passband (no ripples)
2. **Order of the Filter**: Higher order means steeper roll-off and more attenuation of high frequencies
3. **Cutoff Frequency**: Default set to 3500 Hz, allowing frequencies below this to pass through
4. **Zero-Phase Filtering**: Uses `filtfilt` for forward-backward filtering to eliminate phase distortion

The filter is particularly useful for:
- Reducing high-frequency noise
- Smoothing audio signals
- Preventing aliasing
- Creating more natural sound by removing harsh high frequencies

### Griffin-Lim algorithm
- The Griffin-Lim algorithm works by:
   - Starting with random phase information
   - Iteratively improving the phase estimate by:
      - Converting the spectrogram to audio
      - Computing the spectrogram of that audio
      - Replacing the magnitude with the original spectrogram while keeping the new phase
   - Repeating this process for the specified number of iterations

## Key Features

### Audio Processing
- Mel spectrogram extraction and reconstruction
- Audio chunking with overlap
- Crossfading for smooth transitions
- Multiple enhancement techniques:
  - Pre-emphasis
  - Low-pass filtering
  - Silence trimming
  - Amplitude normalization

### Feature Extraction
- Pitch extraction using Parselmouth (Praat) or fallback method
- Duration analysis using energy-based segmentation
- Stress pattern analysis using peak detection
- Comprehensive linguistic feature extraction

### Audio Enhancement
- Low-pass filtering for noise reduction
- Pre-emphasis for clarity
- Silence trimming
- Amplitude normalization

### Chunk Processing
- Overlapping chunk splitting
- Crossfading for smooth transitions
- Efficient processing of long audio files

### Statistics Blending
- Source and target statistics blending
- Configurable blend ratio
- Multiple statistics types supported 

