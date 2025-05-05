import os
import torch
import numpy as np
from pathlib import Path
import soundfile as sf
import librosa
import sys
from gtts import gTTS
import io

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Add the path to the SpeechTranscriber module
sys.path.append('/content/drive/MyDrive/VoiceGenv3')

from Module3TranscriberAndGenerator.SpeechTranscriber import SpeechTranscriber


class TextToSpeechGenerator:
    
    def __init__(self, model_type='gtts', device=None):
        self.model_type = model_type
        self.model_initialized = True
        self.transcriber = SpeechTranscriber()
        print(f"Initializing TextToSpeechGenerator with {model_type}")
    
    def generate_speech(self, text, speaker_embedding=None, speaker_id=None, language="en"):
        try:
            # Create gTTS object
            tts = gTTS(text=text, lang=language, slow=False)
            
            # Save to a bytes buffer
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            # Read the audio data using soundfile
            audio_data, sample_rate = sf.read(audio_buffer)
            
            return audio_data
                
        except Exception as e:
            print(f"Error generating speech: {e}")
            return self._generate_placeholder_audio(text)
    
    def _generate_placeholder_audio(self, text):
        # Calculate a duration based on the text length - roughly 5 characters per second
        duration = max(1.0, len(text) / 5)
        sample_rate = 22050
        
        # Generate a simple sine wave as placeholder
        t = np.linspace(0., duration, int(duration * sample_rate))
        
        # Create a placeholder audio with decreasing frequency
        frequency = 220.0  # Starting frequency in Hz
        
        # Create multiple sine waves with different frequencies based on text length
        audio = np.sin(2 * np.pi * frequency * t)
        
        # Add some words-like modulation
        word_count = max(1, len(text.split()))
        for i in range(word_count):
            # Add a small pause between words
            start_idx = int(i * len(audio) / word_count)
            end_idx = int((i + 0.8) * len(audio) / word_count)
            word_audio = np.sin(2 * np.pi * (frequency * (1 + 0.2 * i)) * t[start_idx:end_idx])
            audio[start_idx:end_idx] = word_audio
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        print(f"Generated placeholder audio for text: '{text}'")
        
        return audio
    
    def save_audio(self, audio, output_path, sample_rate=22050):
        try:
            # Ensure the directory exists
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                print(f"Created directory: {output_dir}")
            
            # Save the audio file
            sf.write(output_path, audio, sample_rate)
            print(f"Audio saved to {output_path}")
        except Exception as e:
            print(f"Error saving audio: {str(e)}")
            # Try saving to a different location
            try:
                # Try saving to the current directory
                fallback_path = "tts_test.wav"
                sf.write(fallback_path, audio, sample_rate)
                print(f"Audio saved to fallback location: {fallback_path}")
            except Exception as e2:
                print(f"Failed to save audio to fallback location: {str(e2)}")

if __name__ == "__main__":
    # Create the TTS generator
    tts_generator = TextToSpeechGenerator(model_type='gtts')
    
    # Get predicted text from SpeechTranscriber
    audio_dir = "/content/drive/MyDrive/VoiceGenv3/NewGenSample/"
    
    # Check if directory exists
    if not os.path.exists(audio_dir):
        print(f"TTS: Directory not found: {audio_dir}")
        print("TTS: Creating directory...")
        os.makedirs(audio_dir, exist_ok=True)
    
    # List all files in the directory
    print(f"TTS: Contents of directory {audio_dir}:")
    try:
        all_files = os.listdir(audio_dir)
        for file in all_files:
            print(f"  - {file}")
    except Exception as e:
        print(f"TTS: Error listing directory: {str(e)}")
    
    # Check for both .wav and .WAV files
    wav_files = [f for f in os.listdir(audio_dir) if f.lower().endswith('.wav')]
    
    if not wav_files:
        print("TTS: No .wav files found in the specified directory.")
        text = "Hello, this is a test of the text to speech generator."  # Fallback text
    else:
        # Use the first .wav file found
        audio_file = os.path.join(audio_dir, wav_files[0])
        print(f"TTS: Processing audio file: {wav_files[0]}")
        
        try:
            transcript, continuation = tts_generator.transcriber.process_audio(audio_file)
            # Use only the continuation part
            text = continuation
            print(f"TTS: Generated continuation: {text}")
        except Exception as e:
            print(f"TTS: Error processing audio: {str(e)}")
            text = "Hello, this is a test of the text to speech generator."  # Fallback text
    
    # Generate speech from the text
    audio = tts_generator.generate_speech(text)
    
    output_dir = "/content/drive/MyDrive/VoiceGenv3/Output/generatedTTS"
    os.makedirs(output_dir, exist_ok=True)
    tts_generator.save_audio(audio, os.path.join(output_dir, "tts_test.wav")) 