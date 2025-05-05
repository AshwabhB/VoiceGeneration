import os
import torch
import numpy as np
from pathlib import Path
import soundfile as sf
import librosa
import sys
from gtts import gTTS
import io
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    
    print("Text-to-Speech Generator")
    print("1. Enter custom text for TTS")
    print("2. Generate text from audio file")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        # Option 1: Get text input from user
        text = input("Enter the text you want to convert to speech: ")
    elif choice == "2":
        # Option 2: Generate text from audio file
        audio_dir = "C:/Users/ashwa/OneDrive/Documents/Projects/VoiceGeneration Project/Module3TranscriberAndGenerator/"
        wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        
        if not wav_files:
            print("No .wav files found in the specified directory.")
            text = "With my women, physical perfection, but it doesn't have to be skinny or like tone like I"  # Fallback text that I used from one of the actual audio transcriptions for testing
        else:
            # Use the first .wav file found
            audio_file = os.path.join(audio_dir, wav_files[0])
            print(f"Processing audio file: {wav_files[0]}")
            
            try:
                transcript, continuation = tts_generator.transcriber.process_audio(audio_file)
                # Use only the continuation part
                text = continuation
                print(f"Generated continuation: {text}")
            except Exception as e:
                print(f"Error processing audio: {str(e)}")
                text = "With my women, physical perfection, but it doesn't have to be skinny or like tone like I"  # Fallback text that I used from one of the actual audio transcriptions for testing
    else:
        print("Invalid choice. Please enter either 1 or 2.")
        sys.exit(1)
    
    # Generate speech from the text
    audio = tts_generator.generate_speech(text)
    

    output_dir = r"C:\Users\ashwa\OneDrive\Documents\Projects\VoiceGeneration Project\Module5TTS\TTSoutput"
    os.makedirs(output_dir, exist_ok=True)
    tts_generator.save_audio(audio, os.path.join(output_dir, "tts_test.wav"))