import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import librosa
import librosa.display
from tqdm import tqdm
import sys
import random
from pathlib import Path
import pickle

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Module3TranscriberAndGenerator.SpeechTranscriber import SpeechTranscriber

class AudioDataset(Dataset):
    """Dataset for training the audio GAN"""
    def __init__(self, data_dir, sample_rate=22050, segment_length=16384, hop_length=4096):
        """
        Initialize the dataset
        
        Args:
            data_dir: Directory containing speaker folders
            sample_rate: Audio sample rate
            segment_length: Length of audio segments to use for training
            hop_length: Hop length for sliding window
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.hop_length = hop_length
        self.audio_files = []
        
        # Collect all audio files
        for speaker_id in os.listdir(data_dir):
            speaker_dir = os.path.join(data_dir, speaker_id)
            if os.path.isdir(speaker_dir):
                for video_id in os.listdir(speaker_dir):
                    video_dir = os.path.join(speaker_dir, video_id)
                    if os.path.isdir(video_dir):
                        for file in os.listdir(video_dir):
                            if file.endswith('.wav'):
                                self.audio_files.append(os.path.join(video_dir, file))
        
        print(f"Found {len(self.audio_files)} audio files")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        audio = librosa.util.normalize(audio)
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sample_rate,
            n_mels=80,
            hop_length=256,
            win_length=1024,
            fmin=20,
            fmax=8000
        )
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        mel_spec = torch.FloatTensor(mel_spec)

        # Pad or crop to self.segment_length
        if mel_spec.shape[1] < self.segment_length:
            pad_width = self.segment_length - mel_spec.shape[1]
            mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_width))
        elif mel_spec.shape[1] > self.segment_length:
            start = random.randint(0, mel_spec.shape[1] - self.segment_length)
            mel_spec = mel_spec[:, start:start + self.segment_length]

        # Create a mask for the missing part (simulating missing segment)
        mask = torch.ones_like(mel_spec)
        mask_width = mel_spec.shape[1] // 4  # Mask 1/4 of the spectrogram
        mask_start = random.randint(0, mel_spec.shape[1] - mask_width)
        mask[:, mask_start:mask_start + mask_width] = 0

        input_spec = mel_spec * mask
        target_spec = mel_spec

        return input_spec, target_spec, mask

class Generator(nn.Module):
    """Generator network for the conditional GAN"""
    def __init__(self, input_channels=80, output_channels=80, hidden_channels=512):
        super(Generator, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(hidden_channels * 2, hidden_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_channels * 4),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(hidden_channels * 4, hidden_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_channels * 8),
            nn.LeakyReLU(0.2),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_channels * 8, hidden_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_channels * 4),
            nn.ReLU(),
            
            nn.ConvTranspose1d(hidden_channels * 4, hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            
            nn.ConvTranspose1d(hidden_channels * 2, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            
            nn.Conv1d(hidden_channels, output_channels, kernel_size=7, padding=3),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Discriminator(nn.Module):
    """Discriminator network for the conditional GAN"""
    def __init__(self, input_channels=80, hidden_channels=512):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(hidden_channels * 2, hidden_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(hidden_channels * 4, hidden_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(hidden_channels * 8, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

class AudioRegenerator:
    """Class for generating audio continuations using a conditional GAN"""
    def __init__(self, data_dir, output_dir=None, model_dir=None, 
                 sample_rate=22050, batch_size=16, num_epochs=50, learning_rate=0.0002):
        """
        Initialize the AudioRegenerator
        
        Args:
            data_dir: Directory containing the voice dataset
            output_dir: Directory to save generated audio
            model_dir: Directory to save model weights
            sample_rate: Audio sample rate
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizers
        """
        self.data_dir = data_dir
        self.module_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set default output directories if not provided
        if output_dir is None:
            self.output_dir = os.path.join(self.module_dir, "output")
        else:
            self.output_dir = output_dir
            
        if model_dir is None:
            self.model_dir = os.path.join(self.output_dir, "model")
        else:
            self.model_dir = model_dir
            
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        # Create output directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "generated"), exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize models
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        
        # Initialize loss functions
        self.adversarial_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()
        
        # Initialize dataset and dataloader
        self.dataset = AudioDataset(data_dir, sample_rate=sample_rate)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        # Initialize transcriber for text prediction
        self.transcriber = SpeechTranscriber()
        
        # Path for dataset info
        self.dataset_info_file = os.path.join(self.model_dir, 'dataset_info.pkl')
    
    def train(self):
        """Train the GAN model"""
        print(f"Training Audio GAN for {self.num_epochs} epochs on {self.data_dir}...")
        
        for epoch in range(self.num_epochs):
            g_losses = []
            d_losses = []
            
            for i, (input_spec, target_spec, mask) in enumerate(tqdm(self.dataloader)):
                # Move data to device
                input_spec = input_spec.to(self.device)
                target_spec = target_spec.to(self.device)
                mask = mask.to(self.device)
                
                # Get batch size
                batch_size = input_spec.size(0)
                
                # Create labels
                d_real = self.discriminator(target_spec)
                d_fake = self.discriminator(self.generator(input_spec).detach())
                
                real_labels = torch.ones_like(d_real)
                fake_labels = torch.zeros_like(d_fake)
                
                d_real_loss = self.adversarial_loss(d_real, real_labels)
                d_fake_loss = self.adversarial_loss(d_fake, fake_labels)
                d_loss = d_real_loss + d_fake_loss
                
                d_loss.backward()
                self.d_optimizer.step()
                

                #  Train Generator

                self.g_optimizer.zero_grad()
                
                # Calculate generator loss
                g_fake = self.discriminator(self.generator(input_spec))
                g_loss = self.adversarial_loss(g_fake, real_labels)
                
                # Add L1 loss for better reconstruction
                l1_loss = self.l1_loss(self.generator(input_spec), target_spec)
                g_loss = g_loss + 10 * l1_loss
                
                g_loss.backward()
                self.g_optimizer.step()
                
                # Store losses
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
                
                # Print progress
                if i % 10 == 0:
                    print(f"Epoch [{epoch}/{self.num_epochs}] Batch [{i}/{len(self.dataloader)}] "
                          f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")
            
            # Print epoch summary
            avg_g_loss = sum(g_losses) / len(g_losses)
            avg_d_loss = sum(d_losses) / len(d_losses)
            print(f"Epoch [{epoch}/{self.num_epochs}] "
                  f"D_loss: {avg_d_loss:.4f} G_loss: {avg_g_loss:.4f}")
            
            # Save model checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_model(epoch + 1)
    
    def save_model(self, epoch):
        """Save model weights and dataset info"""
        # Save model weights
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'epoch': epoch
        }, os.path.join(self.model_dir, f"audio_gan_epoch_{epoch}.pth"))
        
        # Save dataset info
        dataset_info = {
            'dataset_path': self.data_dir,
            'sample_rate': self.sample_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate
        }
        with open(self.dataset_info_file, 'wb') as f:
            pickle.dump(dataset_info, f)
            
        print(f"Model and dataset info saved at epoch {epoch}")
    
    def load_model(self, model_path=None):
        """Load model weights and check if dataset has changed"""
        # If no model path is provided, use the latest model
        if model_path is None:
            model_files = [f for f in os.listdir(self.model_dir) if f.startswith('audio_gan_epoch_') and f.endswith('.pth')]
            if not model_files:
                print("No saved models found. Training from scratch.")
                return False
            model_path = os.path.join(self.model_dir, sorted(model_files)[-1])
        
        # Check if dataset info exists
        if not os.path.exists(self.dataset_info_file):
            print("No dataset info found. Training from scratch.")
            return False
            
        # Load dataset info
        with open(self.dataset_info_file, 'rb') as f:
            dataset_info = pickle.load(f)
            
        # Check if dataset has changed
        if dataset_info['dataset_path'] != self.data_dir:
            print("Dataset path has changed. Training from scratch.")
            return False
            
        # Load model weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        print(f"Model loaded from {model_path}")
        return True
    
    def generate_missing(self, context_audio, predicted_text=None):
        """
        Generate the missing part of an audio segment
        
        Args:
            context_audio: Path to the context audio file
            predicted_text: Optional text to condition the generation on
            
        Returns:
            Generated audio segment
        """
        # Load and preprocess context audio
        audio, sr = librosa.load(context_audio, sr=self.sample_rate)
        audio = librosa.util.normalize(audio)
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sample_rate,
            n_mels=80,
            hop_length=256,
            win_length=1024,
            fmin=20,
            fmax=8000
        )
        
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize mel spectrogram
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        # Convert to tensor
        mel_spec = torch.FloatTensor(mel_spec).unsqueeze(0).to(self.device)
        
        # Create a mask for the missing part
        mask = torch.ones_like(mel_spec)
        mask_width = mel_spec.shape[2] // 4  # Mask 1/4 of the spectrogram
        mask_start = mel_spec.shape[2] - mask_width
        mask[:, :, mask_start:] = 0
        
        # Apply mask to create input
        input_spec = mel_spec * mask
        
        # Generate missing part
        with torch.no_grad():
            self.generator.eval()
            generated_spec = self.generator(input_spec)
        
        # Convert back to numpy
        generated_spec = generated_spec.squeeze(0).cpu().numpy()
        
        # Convert from log scale back to linear
        generated_spec = librosa.db_to_power(generated_spec, ref=np.max)
        
        # Convert mel spectrogram to audio using Griffin-Lim
        generated_audio = librosa.feature.inverse.mel_to_audio(
            generated_spec,
            sr=self.sample_rate,
            hop_length=256,
            win_length=1024
        )
        
        return generated_audio
    
    def save_generated(self, audio_data, path):
        """
        Save generated audio to a file
        
        Args:
            audio_data: Audio data to save
            path: Path to save the audio file
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the audio file
        librosa.output.write_wav(path, audio_data, self.sample_rate)
        print(f"Generated audio saved to {path}")

def main():
    """Main function to demonstrate usage"""
    # Set paths
    data_dir = "C:/Users/ashwa/OneDrive/Documents/Projects/VoiceGeneration Project/Datasets/VC"
    
    # Get the directory of the current module
    module_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(module_dir, "output")
    model_dir = os.path.join(output_dir, "model")
    new_gen_dir = os.path.join(module_dir, "NewGenSample")
    
 
    dataset_changed = True  #will be false when there isd no changein ds since last training. 
    
    # Create AudioRegenerator
    regenerator = AudioRegenerator(
        data_dir=data_dir,
        output_dir=output_dir,
        model_dir=model_dir,
        num_epochs=50
    )
    
    # Check if we need to train or can use saved model
    if dataset_changed:
        print("Dataset has changed. Training from scratch.")
        # Train the model
        regenerator.train()
    else:
        print("Dataset has not changed. Attempting to load saved model.")
        model_loaded = regenerator.load_model()
        if not model_loaded:
            print("Could not load saved model. Training from scratch.")
            regenerator.train()
    
    # Example of generating missing audio
    if os.path.exists(new_gen_dir):
        for file in os.listdir(new_gen_dir):
            if file.endswith('.wav'):
                context_audio = os.path.join(new_gen_dir, file)
                print(f"Generating missing audio for {context_audio}")
                
                # Generate missing audio
                generated_audio = regenerator.generate_missing(context_audio)
                
                # Save generated audio
                base_name = os.path.splitext(file)[0]
                output_path = os.path.join(output_dir, "generated", f"{base_name}_gen.wav")
                regenerator.save_generated(generated_audio, output_path)
                print(f"Generated 3s audio for missing segment -> {output_path}")
    else:
        print(f"NewGenSample directory not found at {new_gen_dir}")
        print("Please add .wav files to the NewGenSample directory for testing")

if __name__ == "__main__":
    main() 