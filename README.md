# Voice Generation Project - Module Details

This document provides a detailed explanation of each module in the Voice Generation Project, with special focus on Module 4.

## Module 1: Feature Extraction and Training

### Overview
The FeatureExtractorAndTrainer class is responsible for extracting audio features and training a speaker recognition model.

### Key Components
1. **Feature Extraction**
   - Extracts MFCCs (Mel-frequency cepstral coefficients) with 20 coefficients
   - Computes spectral features (centroids, rolloff)
   - Calculates zero crossing rate and RMS energy
   - Supports audio augmentation (noise, time stretching, pitch shifting)

2. **Model Architecture**
   - Enhanced MLP (Multi-Layer Perceptron) with residual connections
   - Feature extraction layer (512 neurons)
   - Multiple residual blocks for better feature processing
   - Classification head with multiple layers (512 → 256 → 128 → num_classes)

3. **Training Process**
   - Uses Adam optimizer with learning rate scheduling
   - Implements early stopping with patience
   - Handles class imbalance through weighted loss
   - Supports batch processing and GPU acceleration

## Module 2: Speaker Embedding

### Overview
This module focuses on extracting and processing speaker-specific characteristics from audio.

### Key Components
1. **Speaker Feature Extraction**
   - Extracts speaker-specific features
   - Processes voice characteristics
   - Creates speaker embeddings

2. **Integration**
   - Works with Module 1 for feature extraction
   - Provides input to Module 4 for voice generation

## Module 3: Speech Transcription and Text Generation

### Overview
The SpeechTranscriber class handles audio-to-text conversion and text continuation generation.

### Key Components
1. **Audio Transcription**
   - Uses Whisper model for accurate transcription
   - Supports multiple languages
   - Handles various audio qualities

2. **Text Generation**
   - Uses GPT-2 large model for text continuation
   - Maintains context and coherence
   - Implements text cleaning and validation

3. **Context Processing**
   - Extracts key phrases and topics
   - Maintains sentence completeness
   - Handles text cleaning and formatting

## Module 4: Audio Regenerator (Detailed)

### Overview
The AudioRegenerator implements a conditional GAN for generating audio continuations that match the input speaker's voice characteristics.

### Architecture

1. **Generator Network**
   - U-Net style architecture with encoder-decoder
   - Input channels: 80 (mel spectrogram bands)
   - Hidden channels: 512
   - Encoder path:
     * 4 convolutional layers with increasing channels
     * Batch normalization and LeakyReLU activation
     * Downsampling through stride
   - Decoder path:
     * 4 transposed convolutional layers
     * Batch normalization and ReLU activation
     * Upsampling through stride
     * Final tanh activation

2. **Discriminator Network**
   - Sequential architecture
   - Input channels: 80
   - Hidden channels: 512
   - 5 convolutional layers with increasing channels
   - LeakyReLU activation
   - Final sigmoid activation for binary classification

3. **Audio Processing**
   - Sample rate: 22050 Hz
   - Mel spectrogram conversion
   - Normalization and standardization
   - Mask generation for missing segments

### Training Process

1. **Data Preparation**
   - Loads audio files from dataset
   - Converts to mel spectrograms
   - Creates masks for missing segments
   - Batches data for training

2. **Training Loop**
   - Alternates between generator and discriminator training
   - Uses adversarial loss and L1 loss
   - Implements batch processing
   - Saves model checkpoints

3. **Generation Process**
   - Takes context audio as input
   - Processes through mel spectrogram
   - Generates missing segments
   - Converts back to audio

### Integration Points

1. **With Module 3 (SpeechTranscriber)**
   - Uses transcribed text to condition generation
   - Maintains semantic coherence
   - Aligns audio with text

2. **With Module 2 (SpeakerEmbedding)**
   - Incorporates speaker characteristics
   - Maintains voice consistency
   - Enhances speaker similarity

### Output and Storage

1. **Model Weights**
   - Saved as .pth files
   - Includes generator and discriminator states
   - Stores optimizer states
   - Maintains training metadata

2. **Generated Audio**
   - Saved as .wav files
   - Stored in output/generated directory
   - Named with original file base name + "_gen"

### Hyperparameters

1. **Generator Network**
   - Learning rate: 0.0002
   - Batch size: 16
   - Input channels: 80 (mel spectrogram bands)
   - Hidden channels: 512
   - Number of residual blocks: 6
   - Kernel size: 3x3
   - Stride: 2 for downsampling/upsampling
   - Padding: 1
   - Dropout rate: 0.1

2. **Discriminator Network**
   - Learning rate: 0.0001
   - Batch size: 16
   - Input channels: 80
   - Hidden channels: 512
   - Number of layers: 5
   - Kernel size: 4x4
   - Stride: 2
   - Padding: 1
   - LeakyReLU slope: 0.2

3. **Training Parameters**
   - Number of epochs: 50
   - Optimizer: Adam
   - Beta1: 0.5
   - Beta2: 0.999
   - Weight decay: 0.0001
   - Lambda L1: 100 (L1 loss weight)
   - Lambda GAN: 1 (GAN loss weight)

4. **Audio Processing**
   - Sample rate: 22050 Hz
   - Hop length: 256
   - Window size: 1024
   - Number of mel bands: 80
   - Mel scale: 0-8000 Hz
   - Normalization: Min-max scaling to [-1, 1]

### Usage Example

```python
# Initialize regenerator
regenerator = AudioRegenerator(
    data_dir="path/to/dataset",
    output_dir="path/to/output",
    model_dir="path/to/model",
    num_epochs=50
)

# Train model
regenerator.train()

# Generate missing audio
generated_audio = regenerator.generate_missing("path/to/context.wav")
regenerator.save_generated(generated_audio, "path/to/output.wav")
```

### Performance Considerations

1. **Hardware Requirements**
   - GPU acceleration when available
   - Minimum 8GB RAM recommended
   - Storage for dataset and generated audio

2. **Training Time**
   - Depends on dataset size
   - Affected by number of epochs
   - Varies with hardware capabilities

3. **Quality Factors**
   - Audio quality consistency
   - Speaking style similarity
   - Dataset size and diversity 