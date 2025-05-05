# Voice Converter Runner README

## Overview
This file serves as the runner script for the voice conversion system, handling setup, dependencies, and providing a user interface for the voice conversion process.

## Main Components

### Functions
1. `create_module_directory`
2. `install_dependencies`
3. `save_module_files`
4. `get_available_files`
5. `main`

## Detailed Component Descriptions

### create_module_directory
Creates the necessary directory structure for the voice conversion module.

### install_dependencies
Installs all required Python packages for the voice conversion system:
- librosa (0.10.1) - Audio processing
- soundfile (0.12.1) - Audio file handling
- tqdm (4.66.1) - Progress bars
- numpy (1.24.3) - Numerical computations
- torch (2.1.0) - Deep learning framework
- matplotlib (3.7.2) - Visualization
- scipy (1.11.3) - Scientific computing
- parselmouth (0.4.3) - Pitch extraction
- simplejson (3.19.2) - JSON handling

### save_module_files
Saves the module initialization files to the specified directory.

### get_available_files
Helper function to list available audio files in a directory.

### main
The main entry point that:
1. Sets up required directories
2. Creates the voice converter instance
3. Provides a user interface for:
   - Training new models
   - Resuming training
   - Converting voices
   - Running debug conversions

## Directory Structure
The script manages the following directory structure:
- Data directory: `/content/drive/MyDrive/VC`
- Output directory: `/content/drive/MyDrive/VoiceGenv4/Output`
- Model directory: `/content/drive/MyDrive/VoiceGenv4/Output/model`
- Target voice directory: `/content/drive/MyDrive/VoiceGenv4/NewGenSample`
- TTS audio directory: `/content/drive/MyDrive/VoiceGenv4/Output/generatedTTS`

## User Interface Options
The script provides four main options:
1. Train a new model
2. Resume training from the latest checkpoint
3. Convert voice using an existing model
4. Run debug conversion with detailed diagnostics

## Google Colab Support
The script includes special handling for Google Colab environment:
- Detects if running in Colab
- Skips PyTorch installation in Colab (pre-installed)
- Adjusts paths for Colab environment 