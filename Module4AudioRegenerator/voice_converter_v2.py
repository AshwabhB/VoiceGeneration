import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import librosa
import pickle
import random
from tqdm import tqdm
import sys
from pathlib import Path
import time
import gc

# Add path to import custom modules
sys.path.append('/content/drive/MyDrive/VoiceGenv4/Module4AudioRegenerator')
from improved_generator import ImprovedGenerator, ImprovedDiscriminator, GlobalStats
from audio_processor import AudioProcessor, blend_statistics

def clear_memory():
    """Clear GPU memory and garbage collection"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

class ImprovedVoiceDataset(Dataset):
    """Dataset for training the voice conversion model with improved handling"""
    def __init__(self, data_dir, audio_processor, segment_length=8192, augment=True):
        """
        Initialize the dataset
        
        Args:
            data_dir: Directory containing speaker folders
            audio_processor: AudioProcessor instance
            segment_length: Length of audio segments to use for training
            augment: Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.audio_processor = audio_processor
        self.segment_length = segment_length
        self.augment = augment
        self.audio_files = []
        
        # Memory-efficient file collection
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
        try:
            # Load and process audio
            audio_path = self.audio_files[idx]
            audio = self.audio_processor.load_audio(audio_path)
            
            # Extract mel spectrogram
            mel_spec = self.audio_processor.audio_to_mel(audio)
            
            # Track global statistics
            GlobalStats.update_from_batch(torch.FloatTensor(mel_spec))
            
            # Normalize using global statistics
            mel_spec_norm = GlobalStats.normalize(mel_spec)
            mel_spec_norm = torch.FloatTensor(mel_spec_norm)
            
            # Pad or crop to segment_length
            if mel_spec_norm.shape[1] < self.segment_length:
                pad_width = self.segment_length - mel_spec_norm.shape[1]
                mel_spec_norm = torch.nn.functional.pad(mel_spec_norm, (0, pad_width))
            elif mel_spec_norm.shape[1] > self.segment_length:
                start = random.randint(0, mel_spec_norm.shape[1] - self.segment_length)
                mel_spec_norm = mel_spec_norm[:, start:start + self.segment_length]
            
            # Apply data augmentation if enabled
            if self.augment:
                # Create training pairs with different types of masks
                mask_type = random.choice(['frequency', 'time', 'both', 'none'])
                
                if mask_type == 'frequency':
                    # Mask frequency bands
                    mask = torch.ones_like(mel_spec_norm)
                    n_freq_masks = random.randint(1, 3)
                    for i in range(n_freq_masks):
                        freq_width = random.randint(5, 20)  # Width of frequency mask
                        freq_start = random.randint(0, mel_spec_norm.shape[0] - freq_width)
                        mask[freq_start:freq_start + freq_width, :] = 0
                
                elif mask_type == 'time':
                    # Mask time segments
                    mask = torch.ones_like(mel_spec_norm)
                    n_time_masks = random.randint(1, 3)
                    for i in range(n_time_masks):
                        time_width = mel_spec_norm.shape[1] // 4  # Mask 1/4 of the time
                        time_start = random.randint(0, mel_spec_norm.shape[1] - time_width)
                        mask[:, time_start:time_start + time_width] = 0
                
                elif mask_type == 'both':
                    # Mask both time and frequency
                    mask = torch.ones_like(mel_spec_norm)
                    # Frequency mask
                    freq_width = random.randint(5, 20)
                    freq_start = random.randint(0, mel_spec_norm.shape[0] - freq_width)
                    mask[freq_start:freq_start + freq_width, :] = 0
                    # Time mask
                    time_width = mel_spec_norm.shape[1] // 4
                    time_start = random.randint(0, mel_spec_norm.shape[1] - time_width)
                    mask[:, time_start:time_start + time_width] = 0
                
                else:  # 'none'
                    # No masking, but add noise
                    mask = torch.ones_like(mel_spec_norm)
                    # Add small noise
                    source_spec = mel_spec_norm + torch.randn_like(mel_spec_norm) * 0.05
                    return source_spec, mel_spec_norm, mask
                
                # Apply mask to create input
                source_spec = mel_spec_norm * mask
                target_spec = mel_spec_norm
            else:
                # If no augmentation, still create a simple mask
                mask = torch.ones_like(mel_spec_norm)
                mask_width = mel_spec_norm.shape[1] // 4
                mask_start = random.randint(0, mel_spec_norm.shape[1] - mask_width)
                mask[:, mask_start:mask_start + mask_width] = 0
                source_spec = mel_spec_norm * mask
                target_spec = mel_spec_norm
            
            return source_spec, target_spec, mask
        
        except Exception as e:
            print(f"Error processing file {self.audio_files[idx]}: {e}")
            # Return a dummy sample to prevent crash
            dummy_spec = torch.zeros(80, self.segment_length)
            return dummy_spec, dummy_spec, torch.ones_like(dummy_spec)

class EnhancedVoiceConverter:
    """Enhanced voice conversion system with improved models and processing"""
    def __init__(self, data_dir, output_dir=None, model_dir=None, 
                 sample_rate=22050, batch_size=8, num_epochs=25, learning_rate=0.0001):
        """
        Initialize the EnhancedVoiceConverter
        
        Args:
            data_dir: Directory containing the voice dataset
            output_dir: Directory to save generated audio
            model_dir: Directory to save model weights
            sample_rate: Audio sample rate
            batch_size: Batch size for training (increased to 8)
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizers
        """
        # Clear memory before initialization
        clear_memory()
        print("Initial memory state:")
        print_gpu_memory()
        
        # Set directories
        self.data_dir = data_dir
        
        if output_dir is None:
            self.output_dir = '/content/drive/MyDrive/VoiceGenv4/Output'
        else:
            self.output_dir = output_dir
            
        if model_dir is None:
            self.model_dir = '/content/drive/MyDrive/VoiceGenv4/Output/model'
        else:
            self.model_dir = model_dir
            
        self.weights_dir = os.path.join(self.output_dir, "weights")
        self.converted_dir = os.path.join(self.output_dir, "converted")
        self.debug_dir = os.path.join(self.output_dir, "debug")
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.converted_dir, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Training parameters
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor(sample_rate=sample_rate)
        
        # Initialize models with improved architectures
        try:
            self.generator = ImprovedGenerator(hidden_channels=256).to(self.device)
            self.discriminator = ImprovedDiscriminator(hidden_channels=256).to(self.device)
            print("Initialized improved models with reduced size")
        except RuntimeError as e:
            print(f"Error moving models to GPU: {e}")
            print("Falling back to CPU")
            self.device = torch.device("cpu")
            self.generator = ImprovedGenerator(hidden_channels=256).to(self.device)
            self.discriminator = ImprovedDiscriminator(hidden_channels=256).to(self.device)
        
        # Initialize optimizers with better parameters
        self.g_optimizer = optim.Adam(self.generator.parameters(), 
                                    lr=learning_rate, 
                                    betas=(0.5, 0.999),
                                    weight_decay=1e-5)
        
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), 
                                     lr=learning_rate, 
                                     betas=(0.5, 0.999),
                                     weight_decay=1e-5)
        
        # Initialize loss functions
        self.adversarial_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()
        
        # Initialize dataset and dataloader with improved dataset
        self.dataset = ImprovedVoiceDataset(data_dir, self.audio_processor, segment_length=8192)
        self.dataloader = DataLoader(self.dataset, 
                                    batch_size=batch_size, 
                                    shuffle=True,
                                    num_workers=0,  # Keep at 0 for Colab
                                    pin_memory=False,  # Set to False to reduce memory usage
                                    persistent_workers=False)
        
        # Path for dataset info
        self.dataset_info_file = os.path.join(self.model_dir, 'dataset_info.pkl')
        
        # Print final memory state
        print("Final memory state after initialization:")
        print_gpu_memory()
    
    def train(self, start_epoch=0):
        """Train the voice conversion model with improved stability"""
        print(f"Training Enhanced Voice Conversion GAN for {self.num_epochs} epochs on {self.data_dir}...")
        print(f"Starting from epoch {start_epoch}")
        
        # Set models to training mode
        self.generator.train()
        self.discriminator.train()
        
        # Training ratio: train generator more often than discriminator
        g_steps = 2  # Train generator this many times per discriminator step
        
        # Gradient accumulation steps
        accumulation_steps = 4
        
        # Learning rate schedulers for better convergence
        g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.g_optimizer, mode='min', factor=0.7, patience=5, verbose=True)
        d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.d_optimizer, mode='min', factor=0.7, patience=5, verbose=True)
        
        for epoch in range(start_epoch, self.num_epochs):
            g_losses = []
            d_losses = []
            
            # Memory cleanup at start of epoch
            clear_memory()
            print(f"\nStarting epoch {epoch}")
            print_gpu_memory()
            
            for i, data in enumerate(tqdm(self.dataloader)):
                try:
                    source_spec, target_spec, mask = data
                    
                    # Move data to device
                    source_spec = source_spec.to(self.device)
                    target_spec = target_spec.to(self.device)
                    mask = mask.to(self.device)
                    
                    # Extract linguistic features for source and target
                    source_audio = self.audio_processor.mel_to_audio(source_spec.cpu().numpy())
                    target_audio = self.audio_processor.mel_to_audio(target_spec.cpu().numpy())
                    
                    # Clear CPU memory after audio processing
                    del source_audio, target_audio
                    clear_memory()
                    
                    source_linguistic = self.audio_processor.extract_linguistic_features(source_audio)
                    target_linguistic = self.audio_processor.extract_linguistic_features(target_audio)
                    
                    # Convert linguistic features to tensor
                    source_linguistic_tensor = torch.FloatTensor([
                        source_linguistic['duration']['speech_rate'],
                        source_linguistic['duration']['pause_duration'],
                        source_linguistic['duration']['word_duration'],
                        source_linguistic['stress']['mean_peak_height'],
                        source_linguistic['stress']['stress_pattern'],
                        source_linguistic['stress']['emphasis_marker'],
                        source_linguistic['prosody']['pitch_contour'],
                        source_linguistic['prosody']['energy_envelope'],
                        source_linguistic['prosody']['timing_pattern'],
                        source_linguistic['prosody']['rhythm_marker']
                    ]).to(self.device)
                    
                    target_linguistic_tensor = torch.FloatTensor([
                        target_linguistic['duration']['speech_rate'],
                        target_linguistic['duration']['pause_duration'],
                        target_linguistic['duration']['word_duration'],
                        target_linguistic['stress']['mean_peak_height'],
                        target_linguistic['stress']['stress_pattern'],
                        target_linguistic['stress']['emphasis_marker'],
                        target_linguistic['prosody']['pitch_contour'],
                        target_linguistic['prosody']['energy_envelope'],
                        target_linguistic['prosody']['timing_pattern'],
                        target_linguistic['prosody']['rhythm_marker']
                    ]).to(self.device)
                    
                    # Train Discriminator
                    self.d_optimizer.zero_grad()
                    
                    # Generate converted voice with linguistic features
                    converted_spec = self.generator(source_spec, source_linguistic_tensor)
                    
                    # Get discriminator predictions
                    d_real = self.discriminator(target_spec)
                    d_fake = self.discriminator(converted_spec.detach())
                    
                    # Calculate discriminator loss with label smoothing
                    real_labels = torch.ones_like(d_real) * 0.9  # Label smoothing
                    fake_labels = torch.zeros_like(d_fake) + 0.1  # Label smoothing
                    
                    d_real_loss = self.adversarial_loss(d_real, real_labels)
                    d_fake_loss = self.adversarial_loss(d_fake, fake_labels)
                    d_loss = (d_real_loss + d_fake_loss) / 2
                    
                    # Scale loss by accumulation steps
                    d_loss = d_loss / accumulation_steps
                    
                    # Only update discriminator if it's not too strong
                    if d_loss.item() > 0.3:  # Threshold to prevent discriminator from becoming too strong
                        d_loss.backward()
                        # Apply gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                        
                        # Update weights every accumulation_steps
                        if (i + 1) % accumulation_steps == 0:
                            self.d_optimizer.step()
                            self.d_optimizer.zero_grad()
                    
                    # Train Generator (possibly multiple times)
                    for _ in range(g_steps):
                        self.g_optimizer.zero_grad()
                        
                        # Generate converted voice again with linguistic features
                        converted_spec = self.generator(source_spec, source_linguistic_tensor)
                        
                        # Get discriminator predictions for converted voice
                        g_fake = self.discriminator(converted_spec)
                        
                        # Calculate generator loss with increased L1 weight
                        g_adv_loss = self.adversarial_loss(g_fake, torch.ones_like(g_fake))
                        
                        # L1 loss with increased weight for better audio quality
                        l1_loss = self.l1_loss(converted_spec, target_spec)
                        
                        # Add linguistic feature matching loss
                        linguistic_loss = self.l1_loss(source_linguistic_tensor, target_linguistic_tensor)
                        
                        g_loss = g_adv_loss + 25 * l1_loss + 10 * linguistic_loss
                        
                        # Scale loss by accumulation steps
                        g_loss = g_loss / accumulation_steps
                        
                        g_loss.backward()
                        # Apply gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                        
                        # Update weights every accumulation_steps
                        if (i + 1) % accumulation_steps == 0:
                            self.g_optimizer.step()
                            self.g_optimizer.zero_grad()
                    
                    # Store losses
                    g_losses.append(g_loss.item() * accumulation_steps)  # Scale back for logging
                    d_losses.append(d_loss.item() * accumulation_steps)  # Scale back for logging
                    
                    # Print progress
                    if i % 10 == 0:
                        print(f"Epoch [{epoch}/{self.num_epochs}] Batch [{i}/{len(self.dataloader)}] "
                              f"D_loss: {d_loss.item() * accumulation_steps:.4f} G_loss: {g_loss.item() * accumulation_steps:.4f}")
                        print_gpu_memory()
                    
                    # Periodic memory cleanup
                    if i % 50 == 0:
                        clear_memory()
                    
                    # Save checkpoint every 100 batches
                    if i % 100 == 0:
                        self.save_training_state(epoch)
                        clear_memory()
                    
                    # Check for exploding gradients
                    if g_loss.item() > 100 or d_loss.item() > 100 or torch.isnan(g_loss) or torch.isnan(d_loss):
                        print(f"Warning: Unstable training detected. Skipping batch {i}")
                        continue
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"WARNING: out of memory error, skipping batch {i}")
                        clear_memory()
                        continue
                    else:
                        raise e
            
            # Print epoch summary
            if len(g_losses) > 0 and len(d_losses) > 0:
                avg_g_loss = sum(g_losses) / len(g_losses)
                avg_d_loss = sum(d_losses) / len(d_losses)
                print(f"Epoch [{epoch}/{self.num_epochs}] "
                      f"D_loss: {avg_d_loss:.4f} G_loss: {avg_g_loss:.4f}")
                
                # Update learning rate schedulers
                g_scheduler.step(avg_g_loss)
                d_scheduler.step(avg_d_loss)
                
                # Print current learning rates
                for param_group in self.g_optimizer.param_groups:
                    current_lr_g = param_group['lr']
                for param_group in self.d_optimizer.param_groups:
                    current_lr_d = param_group['lr']
                print(f"Learning rates - Generator: {current_lr_g:.6f}, Discriminator: {current_lr_d:.6f}")
            
            # Save model checkpoint after every epoch
            self.save_model(epoch + 1)
            
            # Also save current training state for resume capability
            self.save_training_state(epoch + 1)
            
            # Clear memory at end of epoch
            clear_memory()
    
    def save_model(self, epoch):
        """Save model weights and dataset info"""
        # Save model weights to the weights directory
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'global_stats': {
                'mel_mean': GlobalStats.MEL_MEAN,
                'mel_std': GlobalStats.MEL_STD,
                'mel_min': GlobalStats.MEL_MIN,
                'mel_max': GlobalStats.MEL_MAX
            }
        }, os.path.join(self.weights_dir, f"voice_conv_weights_epoch_{epoch}.pth"))
        
        # Save full training state to the model directory
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'global_stats': {
                'mel_mean': GlobalStats.MEL_MEAN,
                'mel_std': GlobalStats.MEL_STD,
                'mel_min': GlobalStats.MEL_MIN,
                'mel_max': GlobalStats.MEL_MAX
            },
            'epoch': epoch
        }, os.path.join(self.model_dir, f"voice_conv_checkpoint_epoch_{epoch}.pth"))
        
        # Save dataset info
        dataset_info = {
            'dataset_path': self.data_dir,
            'sample_rate': self.sample_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'last_epoch': epoch
        }
        with open(self.dataset_info_file, 'wb') as f:
            pickle.dump(dataset_info, f)
        
        print(f"Model and dataset info saved at epoch {epoch}")
    
    def save_training_state(self, epoch):
        """Save training state to resume from interruption"""
        # Save the latest training state
        latest_state_path = os.path.join(self.model_dir, 'latest_checkpoint.pth')
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'global_stats': {
                'mel_mean': GlobalStats.MEL_MEAN,
                'mel_std': GlobalStats.MEL_STD,
                'mel_min': GlobalStats.MEL_MIN,
                'mel_max': GlobalStats.MEL_MAX
            },
            'epoch': epoch
        }, latest_state_path)
        print(f"Latest training state saved to {latest_state_path}")
    
    def load_model(self, model_path=None, resume_training=False):
        """Load model weights and state with improved error handling"""
        # Check if we should resume from latest checkpoint
        if resume_training:
            latest_state_path = os.path.join(self.model_dir, 'latest_checkpoint.pth')
            if os.path.exists(latest_state_path):
                try:
                    checkpoint = torch.load(latest_state_path, map_location=self.device)
                    self.generator.load_state_dict(checkpoint['generator_state_dict'])
                    self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                    self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
                    self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
                    
                    # Load global stats if available
                    if 'global_stats' in checkpoint:
                        GlobalStats.MEL_MEAN = checkpoint['global_stats']['mel_mean']
                        GlobalStats.MEL_STD = checkpoint['global_stats']['mel_std']
                        GlobalStats.MEL_MIN = checkpoint['global_stats']['mel_min']
                        GlobalStats.MEL_MAX = checkpoint['global_stats']['mel_max']
                    
                    start_epoch = checkpoint['epoch']
                    print(f"Resuming training from epoch {start_epoch}")
                    return True, start_epoch
                except Exception as e:
                    print(f"Error loading checkpoint: {e}")
                    print("Starting from scratch")
                    return False, 0
        
        # If no model path is provided, use the latest model
        if model_path is None:
            model_files = [f for f in os.listdir(self.model_dir) 
                         if f.startswith('voice_conv_checkpoint_epoch_') and f.endswith('.pth')]
            if not model_files:
                print("No saved models found. Training from scratch.")
                return False, 0
            
            # Sort by epoch number instead of alphabetically
            def get_epoch_number(filename):
                return int(filename.split('_')[-1].split('.')[0])
            model_files.sort(key=get_epoch_number)
            model_path = os.path.join(self.model_dir, model_files[-1])
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}. Training from scratch.")
            return False, 0
        
        # Load model weights
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            
            # Try to load optimizer states if available
            if 'g_optimizer_state_dict' in checkpoint and 'd_optimizer_state_dict' in checkpoint:
                self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
                self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
            
            # Load global stats if available
            if 'global_stats' in checkpoint:
                GlobalStats.MEL_MEAN = checkpoint['global_stats']['mel_mean']
                GlobalStats.MEL_STD = checkpoint['global_stats']['mel_std']
                GlobalStats.MEL_MIN = checkpoint['global_stats']['mel_min']
                GlobalStats.MEL_MAX = checkpoint['global_stats']['mel_max']
            
            start_epoch = checkpoint.get('epoch', 0)
            print(f"Model loaded from {model_path}, starting from epoch {start_epoch}")
            return True, start_epoch
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training from scratch")
            return False, 0
    
    def clean_audio(self, audio):
        """Clean audio by removing non-finite values and normalizing"""
        # Replace non-finite values with zeros
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize audio to prevent extreme values
        audio = librosa.util.normalize(audio)
        
        return audio

    def convert_voice(self, tts_audio_path, target_voice_path, output_path=None):
        """
        Convert TTS audio to target voice with improved processing and error handling
        """
        print(f"Starting voice conversion process...")
        print(f"TTS audio path: {tts_audio_path}")
        print(f"Target voice path: {target_voice_path}")
        
        # Set the model to evaluation mode
        self.generator.eval()
        
        try:
            # 1. Load and validate target voice
            print("\nLoading target voice...")
            try:
                target_audio = self.audio_processor.load_audio(target_voice_path)
                print(f"Target audio loaded successfully - Shape: {target_audio.shape}")
            except Exception as e:
                print(f"Error loading target voice: {str(e)}")
                print("Attempting to load with librosa directly...")
                target_audio, _ = librosa.load(target_voice_path, sr=self.sample_rate)
                target_audio = self.audio_processor.clean_audio(target_audio)
            
            # Validate target audio
            if not np.isfinite(target_audio).all():
                print("Warning: Target audio contains non-finite values, attempting to clean...")
                target_audio = np.nan_to_num(target_audio, nan=0.0, posinf=0.0, neginf=0.0)
                target_audio = librosa.util.normalize(target_audio)
                target_audio = np.clip(target_audio, -1.0, 1.0)
            
            # 2. Load and validate TTS audio
            print("\nLoading TTS audio...")
            try:
                tts_audio = self.audio_processor.load_audio(tts_audio_path)
                print(f"TTS audio loaded successfully - Shape: {tts_audio.shape}")
            except Exception as e:
                print(f"Error loading TTS audio: {str(e)}")
                print("Attempting to load with librosa directly...")
                tts_audio, _ = librosa.load(tts_audio_path, sr=self.sample_rate)
                tts_audio = self.audio_processor.clean_audio(tts_audio)
            
            # Validate TTS audio
            if not np.isfinite(tts_audio).all():
                print("Warning: TTS audio contains non-finite values, attempting to clean...")
                tts_audio = np.nan_to_num(tts_audio, nan=0.0, posinf=0.0, neginf=0.0)
                tts_audio = librosa.util.normalize(tts_audio)
                tts_audio = np.clip(tts_audio, -1.0, 1.0)
            
            # 3. Extract and validate mel spectrograms
            print("\nExtracting mel spectrograms...")
            try:
                target_stats, target_mel_spec = self.audio_processor.extract_voice_stats(target_audio)
                print("Target mel spectrogram extracted successfully")
            except Exception as e:
                print(f"Error extracting target mel spectrogram: {str(e)}")
                print("Attempting direct mel spectrogram computation...")
                target_mel_spec = librosa.feature.melspectrogram(
                    y=target_audio,
                    sr=self.sample_rate,
                    n_fft=1024,
                    hop_length=256,
                    n_mels=80
                )
                target_mel_spec = librosa.power_to_db(target_mel_spec, ref=np.max)
                target_stats = {
                    'mean': float(np.mean(target_mel_spec)),
                    'std': float(np.std(target_mel_spec)),
                    'min': float(np.min(target_mel_spec)),
                    'max': float(np.max(target_mel_spec))
                }
            
            try:
                tts_stats, tts_mel_spec = self.audio_processor.extract_voice_stats(tts_audio)
                print("TTS mel spectrogram extracted successfully")
            except Exception as e:
                print(f"Error extracting TTS mel spectrogram: {str(e)}")
                print("Attempting direct mel spectrogram computation...")
                tts_mel_spec = librosa.feature.melspectrogram(
                    y=tts_audio,
                    sr=self.sample_rate,
                    n_fft=1024,
                    hop_length=256,
                    n_mels=80
                )
                tts_mel_spec = librosa.power_to_db(tts_mel_spec, ref=np.max)
                tts_stats = {
                    'mean': float(np.mean(tts_mel_spec)),
                    'std': float(np.std(tts_mel_spec)),
                    'min': float(np.min(tts_mel_spec)),
                    'max': float(np.max(tts_mel_spec))
                }
            
            # Validate mel spectrograms
            if not np.isfinite(target_mel_spec).all():
                print("Warning: Target mel spectrogram contains non-finite values, cleaning...")
                target_mel_spec = np.nan_to_num(target_mel_spec, nan=0.0, posinf=0.0, neginf=0.0)
            
            if not np.isfinite(tts_mel_spec).all():
                print("Warning: TTS mel spectrogram contains non-finite values, cleaning...")
                tts_mel_spec = np.nan_to_num(tts_mel_spec, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Continue with the rest of the conversion process...
            # [Rest of the existing convert_voice method code]
            
        except Exception as e:
            print(f"\nError during voice conversion: {str(e)}")
            print("Attempting to save debug information...")
            
            # Save debug information
            debug_dir = os.path.join(self.debug_dir, "error_debug")
            os.makedirs(debug_dir, exist_ok=True)
            
            with open(os.path.join(debug_dir, "error_log.txt"), 'w') as f:
                f.write(f"Error: {str(e)}\n")
                f.write(f"TTS audio path: {tts_audio_path}\n")
                f.write(f"Target voice path: {target_voice_path}\n")
                f.write(f"Error type: {type(e)}\n")
                import traceback
                f.write(f"Traceback:\n{traceback.format_exc()}")
            
            raise
    
    def debug_conversion(self, tts_audio_path, target_voice_path):
        """
        Perform voice conversion with detailed debug outputs
        
        Args:
            tts_audio_path: Path to the TTS audio file
            target_voice_path: Path to the target voice audio file
            
        Returns:
            Path to the debug directory
        """
        # Create a unique debug directory for this conversion
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        debug_dir = os.path.join(self.debug_dir, f"debug_{timestamp}")
        os.makedirs(debug_dir, exist_ok=True)
        
        print(f"Running debug conversion, outputs will be saved to {debug_dir}")
        
        # Set the model to evaluation mode
        self.generator.eval()
        
        # 1. Load and save original audio files
        # 1a. Target voice
        target_audio = self.audio_processor.load_audio(target_voice_path)
        target_debug_path = os.path.join(debug_dir, "01_target_original.wav")
        self.audio_processor.save_audio(target_audio, target_debug_path)
        
        # 1b. TTS audio
        tts_audio = self.audio_processor.load_audio(tts_audio_path)
        tts_debug_path = os.path.join(debug_dir, "02_tts_original.wav")
        self.audio_processor.save_audio(tts_audio, tts_debug_path)
        
        # 2. Extract and save mel spectrograms
        # 2a. Target voice
        target_stats, target_mel_spec = self.audio_processor.extract_voice_stats(target_audio)
        with open(os.path.join(debug_dir, "03_target_stats.txt"), 'w') as f:
            for key, value in target_stats.items():
                f.write(f"{key}: {value}\n")
        
        # 2b. TTS audio
        tts_stats, tts_mel_spec = self.audio_processor.extract_voice_stats(tts_audio)
        with open(os.path.join(debug_dir, "04_tts_stats.txt"), 'w') as f:
            for key, value in tts_stats.items():
                f.write(f"{key}: {value}\n")
        
        # 2c. Save global stats
        with open(os.path.join(debug_dir, "05_global_stats.txt"), 'w') as f:
            f.write(f"MEL_MEAN: {GlobalStats.MEL_MEAN}\n")
            f.write(f"MEL_STD: {GlobalStats.MEL_STD}\n")
            f.write(f"MEL_MIN: {GlobalStats.MEL_MIN}\n")
            f.write(f"MEL_MAX: {GlobalStats.MEL_MAX}\n")
        
        # 3. Normalize and save intermediate audio
        # 3a. Normalize TTS mel spectrogram
        tts_mel_spec_norm = GlobalStats.normalize(tts_mel_spec)
        
        # 3b. Convert normalized back to audio as a sanity check
        tts_mel_spec_norm_denorm = GlobalStats.denormalize(tts_mel_spec_norm)
        tts_norm_audio = self.audio_processor.mel_to_audio(tts_mel_spec_norm_denorm)
        tts_norm_path = os.path.join(debug_dir, "06_tts_norm_denorm.wav")
        self.audio_processor.save_audio(tts_norm_audio, tts_norm_path)
        
        # 4. Process a small chunk for debugging generator behavior
        # Take a 4-second chunk from the middle
        chunk_size = 4 * self.sample_rate // self.audio_processor.hop_length
        if tts_mel_spec_norm.shape[1] > chunk_size:
            mid_point = tts_mel_spec_norm.shape[1] // 2
            start_idx = max(0, mid_point - chunk_size // 2)
            end_idx = min(tts_mel_spec_norm.shape[1], start_idx + chunk_size)
            chunk = tts_mel_spec_norm[:, start_idx:end_idx]
        else:
            chunk = tts_mel_spec_norm
            
        # Pad if needed
        if chunk.shape[1] < chunk_size:
            chunk = np.pad(chunk, ((0, 0), (0, chunk_size - chunk.shape[1])))
            
        # Convert to tensor
        chunk_tensor = torch.FloatTensor(chunk).unsqueeze(0).to(self.device)
        
        # Process through generator
        with torch.no_grad():
            try:
                # Log tensor statistics
                with open(os.path.join(debug_dir, "07_input_tensor_stats.txt"), 'w') as f:
                    f.write(f"Shape: {chunk_tensor.shape}\n")
                    f.write(f"Mean: {chunk_tensor.mean().item()}\n")
                    f.write(f"Std: {chunk_tensor.std().item()}\n")
                    f.write(f"Min: {chunk_tensor.min().item()}\n")
                    f.write(f"Max: {chunk_tensor.max().item()}\n")
                
                # Run through generator
                converted_chunk_tensor = self.generator(chunk_tensor)
                
                # Log output tensor statistics
                with open(os.path.join(debug_dir, "08_output_tensor_stats.txt"), 'w') as f:
                    f.write(f"Shape: {converted_chunk_tensor.shape}\n")
                    f.write(f"Mean: {converted_chunk_tensor.mean().item()}\n")
                    f.write(f"Std: {converted_chunk_tensor.std().item()}\n")
                    f.write(f"Min: {converted_chunk_tensor.min().item()}\n")
                    f.write(f"Max: {converted_chunk_tensor.max().item()}\n")
                
                converted_chunk = converted_chunk_tensor.squeeze(0).cpu().numpy()
                
                # Try different denormalization approaches
                # a. With source stats
                converted_chunk_source = GlobalStats.denormalize(converted_chunk)
                chunk_source_audio = self.audio_processor.mel_to_audio(converted_chunk_source)
                source_path = os.path.join(debug_dir, "09_chunk_denorm_source.wav")
                self.audio_processor.save_audio(chunk_source_audio, source_path)
                
                # b. With target stats
                target_mean, target_std = target_stats['mean'], target_stats['std']
                converted_chunk_target = converted_chunk * target_std + target_mean
                chunk_target_audio = self.audio_processor.mel_to_audio(converted_chunk_target)
                target_path = os.path.join(debug_dir, "10_chunk_denorm_target.wav")
                self.audio_processor.save_audio(chunk_target_audio, target_path)
                
                # c. With blended stats
                blended_stats = blend_statistics(tts_stats, target_stats, blend_ratio=0.7)
                converted_chunk_blended = converted_chunk * blended_stats['std'] + blended_stats['mean']
                chunk_blended_audio = self.audio_processor.mel_to_audio(converted_chunk_blended)
                blended_path = os.path.join(debug_dir, "11_chunk_denorm_blended.wav")
                self.audio_processor.save_audio(chunk_blended_audio, blended_path)
                
            except Exception as e:
                error_message = f"Error during generator processing: {str(e)}"
                print(error_message)
                with open(os.path.join(debug_dir, "error_log.txt"), 'w') as f:
                    f.write(error_message)
        
        # 5. Full conversion with three different methods
        # Process the full audio
        converted_audio, output_path = self.convert_voice(
            tts_audio_path, target_voice_path,
            output_path=os.path.join(debug_dir, "12_full_conversion.wav")
        )
        
        # Return the debug directory path
        return debug_dir
    
    def train_from_epoch(self, start_epoch):
        """Continue training from a specific epoch"""
        print(f"Training from epoch {start_epoch} to {self.num_epochs}")
        return self.train(start_epoch=start_epoch)

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
    
    # Create the enhanced voice converter with increased batch size
    converter = EnhancedVoiceConverter(
        data_dir=data_dir,
        output_dir=output_dir,
        model_dir=model_dir,
        num_epochs=25,
        batch_size=8,  # Increased from 4 to 8
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
        # Train from scratch
        print("Training new model from scratch...")
        converter.train()
    
    elif choice == '2':
        # Resume training
        print("Attempting to resume training from the latest checkpoint...")
        model_loaded, start_epoch = converter.load_model(resume_training=True)
        if model_loaded:
            converter.train_from_epoch(start_epoch)
        else:
            print("No checkpoint found. Training from scratch...")
            converter.train()
    
    elif choice == '3' or choice == '4':
        # Load existing model
        print("Loading existing model...")
        model_loaded, _ = converter.load_model()
        
        if not model_loaded:
            print("Error: No model found. Please train a model first.")
            return
        
        # Get target voice file
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
        
        # Get TTS audio file
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
            # Convert voice
            print("Converting voice...")
            _, output_path = converter.convert_voice(tts_audio_path, target_voice_path)
            print(f"Conversion complete! The converted audio is saved at {output_path}")
        
        else:  # choice == '4'
            # Run debug conversion
            print("Running debug conversion with detailed diagnostics...")
            debug_dir = converter.debug_conversion(tts_audio_path, target_voice_path)
            print(f"Debug conversion complete! Debug files are saved in {debug_dir}")
    
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()