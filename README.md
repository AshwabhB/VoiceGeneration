# Voice Conversion System README

## Overview
This project implements an advanced voice conversion system using GAN-based deep learning techniques. It can convert text-to-speech (TTS) audio to match the characteristics of a target voice while preserving the linguistic content.

## Project Structure

### Core Components
1. `voice_converter_v2.py`: Main voice conversion implementation
2. `voice_converter_runner.py`: Runner script and user interface
3. `improved_generator.py`: Neural network architectures
4. `audio_processor.py`: Audio processing utilities

### Directory Structure
- `/content/drive/MyDrive/VC`: Training data directory
- `/content/drive/MyDrive/VoiceGenv4/Output`: Output directory
- `/content/drive/MyDrive/VoiceGenv4/Output/model`: Model weights directory
- `/content/drive/MyDrive/VoiceGenv4/NewGenSample`: Target voice samples
- `/content/drive/MyDrive/VoiceGenv4/Output/generatedTTS`: TTS audio directory

### Dataset
- `The VoxCeleb1 Dataset` was used. ```https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html```

## Complete Process Flow

### Note:- The following modules are to be part of the project in the future, but are not being used right now. 
1. Module3TranscriberAndGenerator : Will be used to predict the following words that can be used to generate the continuation speech.
2. Module5TTS : Text generated from Module3TranscriberAndGenerator will be used to create TTS. Right now, hard-coded static text is used to generate TTS

### 1. Setup and Installation
1. Install required dependencies:
   - librosa (0.10.1)
   - soundfile (0.12.1)
   - tqdm (4.66.1)
   - numpy (1.24.3)
   - torch (2.1.0)
   - matplotlib (3.7.2)
   - scipy (1.11.3)
   - parselmouth (0.4.3)
   - simplejson (3.19.2)

2. Create necessary directories
3. Set up module structure

### 2. Training Process
1. Data Preparation:
   - Load training audio files
   - Extract mel spectrograms
   - Normalize using global statistics
   - Apply data augmentation

2. Model Training:
   - Initialize generator and discriminator
   - Train using GAN approach
   - Apply residual connections
   - Use spectral normalization
   - Implement learnable scaling
   - Save checkpoints periodically

3. Training Features:
   - Gradient clipping
   - Learning rate scheduling
   - Batch normalization
   - Residual connections
   - Checkpoint saving and resuming

### 3. Voice Conversion Process
1. Input Processing:
   - Load TTS audio
   - Load target voice
   - Extract mel spectrograms
   - Normalize using global statistics

2. Feature Extraction:
   - Extract pitch contours
   - Analyze duration patterns
   - Detect stress patterns
   - Extract linguistic features

3. Voice Conversion:
   - Process through generator
   - Apply voice characteristics
   - Match linguistic features
   - Blend source and target statistics

4. Post-processing:
   - Apply audio enhancements
   - Crossfade overlapping chunks
   - Normalize output
   - Save converted audio

### 4. Debug and Analysis
1. Debug Conversion:
   - Save intermediate outputs
   - Track statistics
   - Monitor feature matching
   - Analyze conversion quality

2. Performance Analysis:
   - Monitor training metrics
   - Track conversion quality
   - Analyze feature matching
   - Evaluate output quality

## Key Features

### Neural Network Architecture
- Generator with residual connections
- Discriminator with spectral normalization
- Learnable scaling for dynamic range
- Batch normalization for stability

### Audio Processing
- Mel spectrogram extraction
- Pitch extraction and modification
- Duration analysis
- Stress pattern detection
- Audio enhancement techniques

### Voice Conversion
- Linguistic feature preservation
- Voice characteristic matching
- Natural prosody transfer
- High-quality audio output

### Training Improvements
- Stable GAN training
- Efficient memory usage
- Checkpoint saving
- Training resumption
- Progress monitoring

## Usage

### Training
1. Run `voice_converter_runner.py`
2. Select option 1 for new training
3. Monitor training progress
4. Save checkpoints periodically

### Voice Conversion
1. Run `voice_converter_runner.py`
2. Select option 3 for conversion
3. Choose target voice
4. Select TTS audio
5. Process and save output

### Debug Mode
1. Run `voice_converter_runner.py`
2. Select option 4 for debug
3. Analyze intermediate outputs
4. Monitor conversion process

## Requirements
- Python 3.6+
- CUDA-capable GPU (recommended)
- Sufficient disk space for audio files
- Google Colab support (optional)

## Notes
- Training requires significant computational resources
- Voice conversion quality depends on training data
- Debug mode helps analyze conversion process
- Regular checkpoint saving is recommended

## Detailed Process Flow

### 1. Initial Setup and Execution
1. User runs `voice_converter_runner.py`
2. Script executes `main()` function which:
   - Calls `create_module_directory()` to set up directory structure
   - Calls `install_dependencies()` to install required packages
   - Calls `save_module_files()` to initialize module files
   - Creates instance of `EnhancedVoiceConverter` from `voice_converter_v2.py`

### 2. Training Process (Option 1)
1. User selects option 1 (Train new model)
2. `EnhancedVoiceConverter.train()` is called which:
   - Initializes `ImprovedVoiceDataset` with data from `/content/drive/MyDrive/VC`
   - Creates PyTorch DataLoader for batch processing
   - Initializes `ImprovedGenerator` and `ImprovedDiscriminator` from `improved_generator.py`
   - Sets up optimizers and learning rate schedulers
   - For each epoch:
     - Calls `ImprovedVoiceDataset.__getitem__()` for each batch
     - Processes audio through `AudioProcessor.audio_to_mel()`
     - Updates `GlobalStats` using `update_from_batch()`
     - Trains generator and discriminator in alternating fashion
     - Saves checkpoint using `save_model()` and `save_training_state()`

### 3. Resume Training (Option 2)
1. User selects option 2 (Resume training)
2. `EnhancedVoiceConverter.load_model(resume_training=True)` is called which:
   - Checks for latest checkpoint in model directory
   - Loads model weights using `torch.load()`
   - Restores optimizer states
   - Loads `GlobalStats` values
   - Returns to `train_from_epoch()` with saved epoch number
   - Continues training from saved state

### 4. Voice Conversion (Option 3)
1. User selects option 3 (Convert voice)
2. `EnhancedVoiceConverter.load_model()` is called to load trained model
3. User selects target voice and TTS audio files
4. `EnhancedVoiceConverter.convert_voice()` is called which:
   - Loads target voice using `AudioProcessor.load_audio()`
   - Extracts target features using `AudioProcessor.extract_linguistic_features()`
   - Loads TTS audio and extracts features
   - Processes TTS audio through generator:
     - Splits into chunks using `AudioProcessor.split_audio_into_chunks()`
     - Processes each chunk through generator
     - Combines chunks using `AudioProcessor.combine_chunks()`
   - Applies post-processing:
     - Blends statistics using `blend_statistics()`
     - Enhances audio using `AudioProcessor.enhance_audio()`
     - Saves output using `AudioProcessor.save_audio()`

### 5. Debug Mode (Option 4)
1. User selects option 4 (Debug conversion)
2. `EnhancedVoiceConverter.debug_conversion()` is called which:
   - Creates debug directory with timestamp
   - Saves original audio files
   - Extracts and saves mel spectrograms
   - Processes small test chunk through generator
   - Saves intermediate outputs:
     - Input tensor statistics
     - Output tensor statistics
     - Different denormalization approaches
   - Performs full conversion with detailed logging
   - Saves all debug information to debug directory

### 6. Helper Functions and Utilities
- `get_available_files()`: Used throughout to list audio files
- `AudioProcessor` methods: Used for all audio processing
- `GlobalStats` methods: Used for normalization
- `blend_statistics()`: Used for voice characteristic matching

### 7. Error Handling and Recovery
- Checkpoint saving every epoch
- Gradient clipping during training
- Memory management for large audio files
- Fallback methods for feature extraction
- Error logging in debug mode 