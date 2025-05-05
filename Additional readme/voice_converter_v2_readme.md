# Voice Converter V2 README

## Overview
This file contains the main voice conversion implementation using a GAN-based approach with improved architectures and processing techniques. The system is designed to convert text-to-speech (TTS) audio to match a target voice while preserving linguistic content.

## Main Components

### Classes
1. `ImprovedVoiceDataset`
2. `EnhancedVoiceConverter`

### Functions
1. `get_available_files`
2. `main`

## Detailed Component Descriptions

#### ImprovedVoiceDataset Class
A PyTorch Dataset class for handling voice data with improved processing and augmentation.

**Parameters:**
- `data_dir`: Directory containing audio files
- `audio_processor`: Instance of AudioProcessor for audio processing
- `segment_length`: Length of audio segments (default: 8192)
- `augment`: Whether to apply data augmentation (default: True)

**Methods:**
- `__init__`: Initializes the dataset with configuration parameters
- `__len__`: Returns the number of audio files in the dataset
- `__getitem__`: Processes and returns a single training sample with augmentation
  - Applies various augmentation techniques (frequency masking, time masking, or both)
  - Returns source spectrogram, target spectrogram, and mask

#### EnhancedVoiceConverter Class
The main voice conversion system implementing a GAN-based approach with improved architectures.

**Parameters:**
- `data_dir`: Directory containing training data
- `output_dir`: Directory for saving outputs (default: None)
- `model_dir`: Directory for saving models (default: None)
- `sample_rate`: Audio sample rate (default: 22050)
- `batch_size`: Training batch size (default: 8)
- `num_epochs`: Number of training epochs (default: 25)
- `learning_rate`: Learning rate for optimizers (default: 0.0002)

**Methods:**

1. **Training Methods:**
- `train(start_epoch=0)`: Main training loop (calls the ImprovedGenerator class)
  - Implements GAN training with improved stability features
  - Uses learning rate scheduling and gradient clipping
  - Saves checkpoints periodically
- `train_from_epoch(start_epoch)`: Resumes training from a specific epoch

2. **Model Management:**
- `save_model(epoch)`: Saves model weights and training state
- `save_training_state(epoch)`: Saves current training state for resuming
- `load_model(model_path=None, resume_training=False)`: Loads model weights and state

3. **Voice Conversion:**
- `convert_voice(tts_audio_path, target_voice_path, output_path=None)`: Main conversion method
  - Converts TTS audio to target voice
  - Implements multiple enhancement techniques
  - Returns converted audio and output path
- `debug_conversion(tts_audio_path, target_voice_path)`: Debug version of conversion
  - Saves detailed diagnostic information
  - Useful for troubleshooting

### Functions

1. **get_available_files**
  - Helper function to display available audio files
```python
def get_available_files(directory, extension='.wav'):
    """
    Lists available audio files in a directory
    
    Parameters:
    - directory: Directory to search
    - extension: File extension to look for (default: '.wav')
    
    Returns:
    - List of matching files
    """
```

1. **main**
```python
def main():
    """
    Main entry point for the voice conversion system
    - Sets up directories and paths
    - Provides interactive menu for:
      1. Training new model
      2. Resuming training
      3. Converting voice
      4. Debug conversion
    """
```

## Detailed Component Descriptions

### Voice Processing Pipeline
1. **Audio Preprocessing:**
   - Mel spectrogram extraction
   - Audio chunking with overlap
   - Crossfading for smooth transitions
   - Multiple enhancement techniques:
     - Pre-emphasis
     - Low-pass filtering
     - Silence trimming
     - Amplitude normalization

2. **Linguistic Feature Extraction:**
   - Pitch extraction and modification
   - Duration analysis
   - Stress pattern analysis
   - Voice characteristics matching

3. **Training Features:**
   - GAN-based training with improved architectures
   - Residual connections
   - Learnable scaling
   - Spectral normalization
   - Gradient clipping
   - Learning rate scheduling
   - Checkpoint saving and resuming

4. **Voice Conversion Process:**
   1. Load and analyze target voice
   2. Process TTS audio
   3. Extract and match linguistic features
   4. Apply voice conversion
   5. Post-process for quality
   6. Save converted audio

### Key Features

#### Voice Processing
- Mel spectrogram extraction and processing
- Audio chunking with overlap for long audio
- Crossfading for smooth transitions
- Multiple enhancement techniques:
  - Pre-emphasis
  - Low-pass filtering
  - Silence trimming
  - Amplitude normalization

#### Linguistic Feature Extraction
- Pitch extraction and modification
- Duration analysis
- Stress pattern analysis
- Voice characteristics matching

#### Training Features
- GAN-based training with improved architectures
- Residual connections for better training
- Learnable scaling for dynamic range
- Spectral normalization for stability
- Gradient clipping
- Learning rate scheduling
- Checkpoint saving and resuming

#### Voice Conversion Process
1. Load and analyze target voice
2. Process TTS audio
3. Extract and match linguistic features
4. Apply voice conversion
5. Post-process for quality
6. Save converted audio

### Helper Functions
- `get_available_files`: Lists available audio files in a directory
- `main`: Main entry point for the voice conversion system 