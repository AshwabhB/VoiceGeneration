#!/usr/bin/env python3
# voice_converter_runner.py - Enhanced Voice Converter Runner

import os
import sys
import subprocess
import time
from voice_converter_v2 import EnhancedVoiceConverter

# Check if we're running in Google Colab
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

def create_module_directory():
    """Create the module directory structure"""
    target_dir = os.path.join(os.getcwd(), 'Module4AudioRegenerator')
    os.makedirs(target_dir, exist_ok=True)
    print(f"Created module directory: {target_dir}")
    return target_dir

def install_dependencies():
    """Install required dependencies"""
    print("Installing required dependencies...")
    
    packages = [
        'librosa==0.10.1',
        'soundfile==0.12.1',
        'tqdm==4.66.1',
        'numpy==1.24.3',
        'torch==2.1.0',
        'matplotlib==3.7.2',
        'scipy==1.11.3',
        'parselmouth==0.4.3',
        'simplejson==3.19.2'
    ]
    
    for package in packages:
        try:
            if package == 'torch' and IN_COLAB:
                continue
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}. Continuing anyway...")
    
    print("Dependencies installed.")

def save_module_files(module_dir):
    """Save the module files to the specified directory"""
    module_files = {
        '__init__.py': '# Module initialization file\n'
    }
    
    print(f"Saving module files to {module_dir}...")
    for file_name, content in module_files.items():
        file_path = os.path.join(module_dir, file_name)
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"  Saved {file_name}")

def get_available_files(directory, extension='.wav'):
    """Helper function to display available audio files"""
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return []
    
    files = [f for f in os.listdir(directory) if f.endswith(extension)]
    return files

def main():
    """Main function for enhanced voice conversion"""
    # Set paths according to requirements
    data_dir = '/content/drive/MyDrive/VC'
    output_dir = '/content/drive/MyDrive/VoiceGenv4/Output'
    model_dir = '/content/drive/MyDrive/VoiceGenv4/Output/model'
    target_voice_dir = '/content/drive/MyDrive/VoiceGenv4/NewGenSample'
    tts_audio_dir = '/content/drive/MyDrive/VoiceGenv4/Output/generatedTTS'
    
    # Create necessary directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(target_voice_dir, exist_ok=True)
    os.makedirs(tts_audio_dir, exist_ok=True)
    
    # Print directory information
    print(f"Using directories:")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model directory: {model_dir}")
    print(f"Target voice directory: {target_voice_dir}")
    print(f"TTS audio directory: {tts_audio_dir}")
    
    # Print available audio files for reference
    print("\nAvailable target voice files:")
    target_files = get_available_files(target_voice_dir)
    for i, file in enumerate(target_files):
        print(f"  {i+1}. {file}")
    
    print("\nAvailable TTS audio files:")
    tts_files = get_available_files(tts_audio_dir)
    for i, file in enumerate(tts_files):
        print(f"  {i+1}. {file}")
    
    # Create the enhanced voice converter
    converter = EnhancedVoiceConverter(
        data_dir=data_dir,
        output_dir=output_dir,
        model_dir=model_dir,
        num_epochs=25,
        batch_size=8,
        learning_rate=0.0001
    )
    
    # Determine action: train or convert
    print("\nWhat would you like to do?")
    print("1. Train a new model (will take several hours)")
    print("2. Resume training from the latest checkpoint (if available)")
    print("3. Convert voice using existing model")
    print("4. Run debug conversion with detailed diagnostics")
    
    choice = input("Enter your choice (1-4): ")
    
    if choice == '1':
        print("Training new model from scratch...")
        converter.train()
    
    elif choice == '2':
        print("Attempting to resume training from the latest checkpoint...")
        model_loaded, start_epoch = converter.load_model(resume_training=True)
        if model_loaded:
            converter.train_from_epoch(start_epoch)
        else:
            print("No checkpoint found. Training from scratch...")
            converter.train()
    
    elif choice == '3' or choice == '4':
        print("Loading existing model...")
        model_loaded, _ = converter.load_model()
        
        if not model_loaded:
            print("Error: No model found. Please train a model first.")
            return
        
        if not target_files:
            print("Error: No target voice files found.")
            return
        
        target_idx = 0
        if len(target_files) > 1:
            target_idx = int(input(f"Select target voice (1-{len(target_files)}): ")) - 1
            if target_idx < 0 or target_idx >= len(target_files):
                print("Invalid selection. Using the first file.")
                target_idx = 0
        
        target_voice_path = os.path.join(target_voice_dir, target_files[target_idx])
        print(f"Using target voice: {target_files[target_idx]}")
        
        if not tts_files:
            print("Error: No TTS audio files found.")
            return
        
        tts_idx = 0
        if len(tts_files) > 1:
            tts_idx = int(input(f"Select TTS audio (1-{len(tts_files)}): ")) - 1
            if tts_idx < 0 or tts_idx >= len(tts_files):
                print("Invalid selection. Using the first file.")
                tts_idx = 0
        
        tts_audio_path = os.path.join(tts_audio_dir, tts_files[tts_idx])
        print(f"Using TTS audio: {tts_files[tts_idx]}")
        
        if choice == '3':
            print("Converting voice...")
            _, output_path = converter.convert_voice(tts_audio_path, target_voice_path)
            print(f"Conversion complete! The converted audio is saved at {output_path}")
        
        else:  # choice == '4'
            print("Running debug conversion with detailed diagnostics...")
            debug_dir = converter.debug_conversion(tts_audio_path, target_voice_path)
            print(f"Debug conversion complete! Debug files are saved in {debug_dir}")
    
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()