import whisper
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import numpy as np
import os
import re

class SpeechTranscriber:
    def __init__(self):
        # Initialize Whisper model (using the medium model for better accuracy)
        self.whisper_model = whisper.load_model("medium")
        
        # Initialize a more sophisticated language model for better text generation
        self.model_name = "gpt2-large"  # Using large model for better quality
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.text_generator = pipeline(
            'text-generation',
            model=self.model_name,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )

    def transcribe(self, audio_path):
        """
        Transcribe audio file to text using Whisper model.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            str: Transcribed text
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found at {audio_path}")
            
        # Transcribe the audio with improved parameters
        result = self.whisper_model.transcribe(
            audio_path,
            language="en",
            task="transcribe",
            fp16=False,  # Use FP32 for better accuracy on CPU
            condition_on_previous_text=False  # Prevent model from using previous context
        )
        
        # Clean up the transcription
        text = result["text"].strip()
        # Remove any trailing incomplete sentences
        text = re.sub(r'[^.!?]+$', '', text)
        return text.strip()

    def is_sentence_complete(self, text):
        """
        Check if the text ends with a complete sentence.
        """
        # Common sentence endings
        endings = ['.', '!', '?', '...']
        return any(text.strip().endswith(end) for end in endings)

    def clean_generated_text(self, text):
        """
        Clean up generated text to remove repetitions and ensure coherence.
        """
        # Remove any prompt-like text
        text = re.sub(r'^(Context:|Complete this sentence|maintaining the same context).*?:\s*', '', text)
        
        # Split into sentences
        sentences = re.split(r'([.!?])\s+', text)
        cleaned_sentences = []
        seen_phrases = set()
        
        for i in range(0, len(sentences)-1, 2):
            sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '')
            # Check for significant repetition
            words = sentence.lower().split()
            if len(words) >= 4:  # Only check phrases of 4 or more words
                for j in range(len(words)-3):
                    phrase = ' '.join(words[j:j+4])
                    if phrase in seen_phrases:
                        continue
                    seen_phrases.add(phrase)
            cleaned_sentences.append(sentence)
        
        return ' '.join(cleaned_sentences).strip()

    def extract_context(self, transcript):
        """
        Extract key context from the transcript for better continuation.
        """
        # Get the last complete sentence
        sentences = re.split(r'([.!?])\s+', transcript)
        if len(sentences) > 1:
            last_sentence = sentences[-2] + sentences[-1]
        else:
            last_sentence = transcript

        # Extract key phrases and topics
        words = last_sentence.split()
        key_phrases = []
        topics = set()
        
        # Common topic indicators
        topic_indicators = ['about', 'regarding', 'concerning', 'talking', 'discussing']
        
        for i in range(len(words)-1):
            # Get significant word pairs
            if len(words[i]) > 3 and len(words[i+1]) > 3:
                key_phrases.append(f"{words[i]} {words[i+1]}")
            
            # Identify topics
            if words[i].lower() in topic_indicators and i+1 < len(words):
                topics.add(words[i+1].lower())
        
        return last_sentence, key_phrases, list(topics)

    def predict_next_text(self, transcript, min_words=20, max_words=30):
        """
        Predict the next 20-30 words that might follow the transcript.
        
        Args:
            transcript (str): The transcribed text
            min_words (int): Minimum number of words to generate
            max_words (int): Maximum number of words to generate
            
        Returns:
            str: Predicted continuation text
        """
        # Extract context from the transcript
        last_sentence, key_phrases, topics = self.extract_context(transcript)
        
        # Prepare the prompt with enhanced context
        context = f"Previous text: {last_sentence}\n"
        if topics:
            context += f"Topics: {', '.join(topics)}\n"
        if key_phrases:
            context += f"Key phrases: {', '.join(key_phrases[:3])}\n"
        
        prompt = f"{context}Continue naturally:"
        
        # Generate text continuation with improved parameters
        generated_text = self.text_generator(
            prompt,
            max_new_tokens=max_words + 10,  # Add buffer for sentence completion
            min_new_tokens=min_words,
            num_return_sequences=1,
            temperature=0.8,  # Balanced temperature for coherence
            top_k=50,  # More focused vocabulary
            top_p=0.92,  # Balanced sampling
            do_sample=True,  # Enable sampling
            no_repeat_ngram_size=2,  # Allow some natural repetition
            pad_token_id=50256,
            truncation=True
        )[0]['generated_text']
        
        # Extract only the continuation part
        continuation = generated_text[len(prompt):].strip()
        
        # Clean up the generated text
        continuation = self.clean_generated_text(continuation)
        
        # If we don't have a complete sentence, generate more text
        if not self.is_sentence_complete(continuation):
            # Find the last complete sentence
            sentences = re.split(r'([.!?])\s+', continuation)
            if len(sentences) > 1:
                # Keep only complete sentences
                continuation = ''.join(sentences[:-1]).strip()
            else:
                # If no complete sentences found, generate more with different parameters
                additional_text = self.text_generator(
                    prompt + continuation,
                    max_new_tokens=max_words + 15,
                    min_new_tokens=min_words,
                    num_return_sequences=1,
                    temperature=0.85,  # Slightly higher temperature
                    top_k=75,  # More variety
                    top_p=0.95,  # More diverse sampling
                    do_sample=True,
                    no_repeat_ngram_size=2,
                    pad_token_id=50256,
                    truncation=True
                )[0]['generated_text']
                continuation = additional_text[len(prompt):].strip()
                continuation = self.clean_generated_text(continuation)
        
        return continuation

    def process_audio(self, audio_path):
        """
        Process audio file: transcribe and predict continuation.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            tuple: (transcript, predicted_continuation)
        """
        # First transcribe the audio
        transcript = self.transcribe(audio_path)
        
        # Then predict the continuation
        continuation = self.predict_next_text(transcript)
        
        return transcript, continuation

# Example usage
if __name__ == "__main__":
    transcriber = SpeechTranscriber()
    
    # Directory containing the audio file
    audio_dir = "C:/Users/ashwa/OneDrive/Documents/Projects/VoiceGeneration Project/Module3TranscriberAndGenerator/"
    
    # Find the .wav file in the directory
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    
    if not wav_files:
        print("No .wav files found in the specified directory.")
    else:
        # Use the first .wav file found
        audio_file = os.path.join(audio_dir, wav_files[0])
        print(f"Processing audio file: {wav_files[0]}")
        
        try:
            transcript, continuation = transcriber.process_audio(audio_file)
            print("\n=== Transcription ===")
            print(transcript)
            print("\n=== Predicted Continuation ===")
            print(continuation)
            print("\n=== Complete Text ===")
            print(transcript + " " + continuation)
        except Exception as e:
            print(f"Error processing audio: {str(e)}")