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

sys.path.append('/content/drive/MyDrive/VoiceGenv4/Module4AudioRegenerator')
from improved_generator import ImprovedGenerator, ImprovedDiscriminator, GlobalStats
from audio_processor import AudioProcessor, blend_statistics

class ImprovedVoiceDataset(Dataset):

    def __init__(self, data_dir, audio_processor, segment_length=8192, augment=True):

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

    def __init__(self, data_dir, output_dir=None, model_dir=None, 
                 sample_rate=22050, batch_size=8, num_epochs=25, learning_rate=0.0002):

        # Monitor memory before initialization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"Initial GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        
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
            self.generator = ImprovedGenerator().to(self.device)
            self.discriminator = ImprovedDiscriminator().to(self.device)
            print("Initialized improved models with residual connections and learnable scaling")
        except RuntimeError as e:
            print(f"Error moving models to GPU: {e}")
            print("Falling back to CPU")
            self.device = torch.device("cpu")
            self.generator = ImprovedGenerator().to(self.device)
            self.discriminator = ImprovedDiscriminator().to(self.device)
        
        # Initialize optimizers with better parameters
        self.g_optimizer = optim.Adam(self.generator.parameters(), 
                                    lr=learning_rate, 
                                    betas=(0.5, 0.999),
                                    weight_decay=1e-5)  # Added weight decay
        
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
                                    num_workers=0, 
                                    pin_memory=False)
        
        # Path for dataset info
        self.dataset_info_file = os.path.join(self.model_dir, 'dataset_info.pkl')
        
        # Monitor memory after initialization
        if torch.cuda.is_available():
            print(f"After init GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    def train(self, start_epoch=0):

        print(f"Training Enhanced Voice Conversion GAN for {self.num_epochs} epochs on {self.data_dir}...")
        print(f"Starting from epoch {start_epoch}")
        
        # Set models to training mode
        self.generator.train()
        self.discriminator.train()
        
        # Training ratio: train generator more often than discriminator
        g_steps = 2  # Train generator this many times per discriminator step
        
        # Learning rate schedulers for better convergence
        g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.g_optimizer, mode='min', factor=0.7, patience=5, verbose=True)
        d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.d_optimizer, mode='min', factor=0.7, patience=5, verbose=True)
        
        for epoch in range(start_epoch, self.num_epochs):
            g_losses = []
            d_losses = []
            
            # Memory cleanup at start of epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            for i, data in enumerate(tqdm(self.dataloader)):
                try:
                    source_spec, target_spec, mask = data
                    
                    # Move data to device
                    source_spec = source_spec.to(self.device)
                    target_spec = target_spec.to(self.device)
                    mask = mask.to(self.device)
                    
                    # Train Discriminator
                    self.d_optimizer.zero_grad()
                    
                    # Generate converted voice
                    converted_spec = self.generator(source_spec)
                    
                    # Get discriminator predictions
                    d_real = self.discriminator(target_spec)
                    d_fake = self.discriminator(converted_spec.detach())
                    
                    # Calculate discriminator loss with label smoothing
                    real_labels = torch.ones_like(d_real) * 0.9  # Label smoothing
                    fake_labels = torch.zeros_like(d_fake) + 0.1  # Label smoothing
                    
                    d_real_loss = self.adversarial_loss(d_real, real_labels)
                    d_fake_loss = self.adversarial_loss(d_fake, fake_labels)
                    d_loss = (d_real_loss + d_fake_loss) / 2
                    
                    # Only update discriminator if it's not too strong
                    if d_loss.item() > 0.3:  # Threshold to prevent discriminator from becoming too strong
                        d_loss.backward()
                        # Apply gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                        self.d_optimizer.step()
                    
                    # Train Generator (possibly multiple times)
                    for _ in range(g_steps):
                        self.g_optimizer.zero_grad()
                        
                        # Generate converted voice again
                        converted_spec = self.generator(source_spec)
                        
                        # Get discriminator predictions for converted voice
                        g_fake = self.discriminator(converted_spec)
                        
                        # Calculate generator loss with increased L1 weight
                        g_adv_loss = self.adversarial_loss(g_fake, torch.ones_like(g_fake))
                        
                        # L1 loss with increased weight for better audio quality
                        l1_loss = self.l1_loss(converted_spec, target_spec)
                        g_loss = g_adv_loss + 25 * l1_loss  # Increased L1 weight to 25 (was 10)
                        
                        g_loss.backward()
                        # Apply gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                        self.g_optimizer.step()
                    
                    # Store losses
                    g_losses.append(g_loss.item())
                    d_losses.append(d_loss.item())
                    
                    # Print progress
                    if i % 10 == 0:
                        print(f"Epoch [{epoch}/{self.num_epochs}] Batch [{i}/{len(self.dataloader)}] "
                              f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")
                        
                        # Monitor memory usage
                        if torch.cuda.is_available():
                            print(f"GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
                    
                    # Periodic memory cleanup
                    if i % 50 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Check for exploding gradients
                    if g_loss.item() > 100 or d_loss.item() > 100 or torch.isnan(g_loss) or torch.isnan(d_loss):
                        print(f"Warning: Unstable training detected. Skipping batch {i}")
                        continue
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"WARNING: out of memory error, skipping batch {i}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def save_model(self, epoch):

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
    
    def convert_voice(self, tts_audio_path, target_voice_path, output_path=None):

        print(f"Converting voice from TTS audio: {tts_audio_path}")
        print(f"Using target voice: {target_voice_path}")
        
        # Set the model to evaluation mode
        self.generator.eval()
        
        # 1. Load and analyze the target voice to get voice characteristics
        target_audio = self.audio_processor.load_audio(target_voice_path)
        target_stats, target_mel_spec = self.audio_processor.extract_voice_stats(target_audio)
        
        # Extract detailed linguistic features from target voice
        target_linguistic = self.audio_processor.extract_linguistic_features(target_audio)
        print("\nTarget Voice Linguistic Features:")
        for category, features in target_linguistic.items():
            print(f"{category}:")
            for key, value in features.items():
                print(f"  {key}: {value:.2f}")
        
        # Save original target audio (for reference)
        debug_target_path = os.path.join(self.debug_dir, "target_original.wav")
        self.audio_processor.save_audio(target_audio, debug_target_path)
        
        # 2. Process the TTS audio (source)
        tts_audio = self.audio_processor.load_audio(tts_audio_path)
        
        # Extract detailed linguistic features from TTS audio
        tts_linguistic = self.audio_processor.extract_linguistic_features(tts_audio)
        print("\nTTS Audio Linguistic Features:")
        for category, features in tts_linguistic.items():
            print(f"{category}:")
            for key, value in features.items():
                print(f"  {key}: {value:.2f}")
        
        # Save original TTS audio (for reference)
        debug_tts_path = os.path.join(self.debug_dir, "tts_original.wav")
        self.audio_processor.save_audio(tts_audio, debug_tts_path)
        
        # Extract mel spectrogram for TTS audio
        tts_stats, tts_mel_spec = self.audio_processor.extract_voice_stats(tts_audio)
        
        # Log the mel spectrogram statistics
        print("\nTTS Mel Spectrogram Statistics:")
        for key, value in tts_stats.items():
            print(f"  {key}: {value}")
        
        print("\nTarget Voice Mel Spectrogram Statistics:")
        for key, value in target_stats.items():
            print(f"  {key}: {value}")
        
        # 3. Enhanced normalization with dynamic range adaptation
        tts_mel_spec_norm = GlobalStats.normalize(tts_mel_spec)
        
        # 4. Process in overlapping chunks with improved segmentation
        chunk_size = 8192  # Similar to our training segment length
        overlap = 0.6      # Increased overlap for smoother transitions
        
        # Split into chunks with improved segmentation
        chunks, positions = self.audio_processor.split_audio_into_chunks(
            tts_mel_spec_norm, chunk_size=chunk_size, overlap=overlap)
        
        # Process each chunk with enhanced processing
        converted_chunks = []
        for chunk_type, chunk, start_idx, end_idx in chunks:
            # Convert to tensor with enhanced preprocessing
            chunk_tensor = torch.FloatTensor(chunk).unsqueeze(0).to(self.device)
            
            # Process through generator with enhanced processing
            with torch.no_grad():
                try:
                    converted_chunk = self.generator(chunk_tensor).squeeze(0).cpu().numpy()
                    converted_chunks.append((chunk_type, converted_chunk, start_idx, end_idx))
                except Exception as e:
                    print(f"Error processing chunk: {e}")
                    # In case of error, use the original chunk
                    converted_chunks.append((chunk_type, chunk, start_idx, end_idx))
        
        # Combine chunks with improved crossfading
        if len(converted_chunks) > 0:
            combined_converted = self.audio_processor.combine_chunks(
                converted_chunks, tts_mel_spec_norm.shape[1])
        else:
            print("Warning: No chunks were successfully converted")
            combined_converted = tts_mel_spec_norm
        
        # 5. Enhanced denormalization with improved feature matching
        # Calculate dynamic blend ratios with increased weights
        pitch_blend = 0.95  # Further increased weight for better pitch matching
        duration_blend = 0.85  # Further increased weight for natural timing
        stress_blend = 0.90  # Further increased weight for natural emphasis
        spectral_blend = 0.80  # Further increased weight for better voice quality
        prosody_blend = 0.90  # Further increased weight for prosody matching
        breath_blend = 0.85  # New weight for breath characteristics
        micro_variation_blend = 0.75  # New weight for micro-variations
        
        # Enhanced blend statistics with improved dynamic weights
        blended_stats = {
            'mean': (1 - spectral_blend) * tts_stats['mean'] + spectral_blend * target_stats['mean'],
            'std': (1 - spectral_blend) * tts_stats['std'] + spectral_blend * target_stats['std'],
            'min': min(tts_stats['min'], target_stats['min']),
            'max': max(tts_stats['max'], target_stats['max'])
        }
        
        # Denormalize using enhanced blended statistics
        converted_denorm = combined_converted * blended_stats['std'] + blended_stats['mean']
        
        # 6. Convert mel spectrogram back to audio with improved quality
        # Convert from log scale back to linear with enhanced processing
        converted_power_spec = librosa.db_to_power(converted_denorm)

        # Convert mel spectrogram to audio with more iterations for better quality
        converted_audio = self.audio_processor.mel_to_audio(
            converted_denorm, griffin_lim_iters=1024)  # Further increased iterations
        
        # 7. Enhanced post-processing with improved feature matching
        # Extract pitch from all audio signals with enhanced processing
        converted_pitch, converted_voiced = self.audio_processor.extract_pitch(converted_audio)
        target_pitch, target_voiced = self.audio_processor.extract_pitch(target_audio)
        tts_pitch, tts_voiced = self.audio_processor.extract_pitch(tts_audio)
        
        # Resample pitch contours to the same length with improved alignment
        min_length = min(len(target_pitch), len(tts_pitch), len(converted_pitch))
        target_pitch = target_pitch[:min_length]
        tts_pitch = tts_pitch[:min_length]
        converted_pitch = converted_pitch[:min_length]
        
        # Enhanced pitch matching with improved prosody preservation
        # Normalize pitch contours using enhanced robust statistics
        target_pitch_norm = (target_pitch - np.median(target_pitch[target_pitch > 0])) / np.std(target_pitch[target_pitch > 0])
        tts_pitch_norm = (tts_pitch - np.median(tts_pitch[tts_pitch > 0])) / np.std(tts_pitch[tts_pitch > 0])
        
        # Blend normalized pitch contours with enhanced dynamic weights
        blended_pitch = (1 - pitch_blend) * tts_pitch_norm + pitch_blend * target_pitch_norm
        
        # Rescale to target pitch range with improved dynamic range matching
        target_median = np.median(target_pitch[target_pitch > 0])
        target_std = np.std(target_pitch[target_pitch > 0])
        blended_pitch = blended_pitch * target_std + target_median
        
        # Find common non-zero indices for more accurate pitch shift
        blended_nonzero = blended_pitch > 0
        converted_nonzero = converted_pitch > 0
        common_nonzero = blended_nonzero & converted_nonzero
        
        if np.sum(common_nonzero) > 0:
            # Calculate pitch shift using enhanced robust statistics
            blended_midi = librosa.hz_to_midi(blended_pitch[common_nonzero])
            converted_midi = librosa.hz_to_midi(converted_pitch[common_nonzero])
            avg_pitch_shift = np.median(blended_midi - converted_midi)
            
            # Apply pitch modification with enhanced dynamic range preservation
            converted_audio = librosa.effects.pitch_shift(
                converted_audio, 
                sr=self.sample_rate, 
                n_steps=float(avg_pitch_shift),
                bins_per_octave=192  # Further increased resolution
            )
        
        # 8. Enhanced duration and stress matching with improved prosody
        # Extract duration features with enhanced processing
        target_duration = target_linguistic['duration']
        tts_duration = tts_linguistic['duration']
        
        # Calculate duration scaling factor with enhanced prosody consideration
        duration_scale = (1 - duration_blend) + duration_blend * (target_duration['speech_rate'] / tts_duration['speech_rate'])
        
        # Apply duration modification with enhanced prosody preservation
        converted_audio = librosa.effects.time_stretch(
            converted_audio, 
            rate=duration_scale
        )
        
        # Extract and match stress patterns with enhanced prosody
        target_stress = target_linguistic['stress']
        tts_stress = tts_linguistic['stress']
        
        # Calculate stress enhancement factor with enhanced prosody consideration
        stress_enhance = (1 - stress_blend) + stress_blend * (target_stress['mean_peak_height'] / tts_stress['mean_peak_height'])
        
        # Apply stress enhancement with improved dynamic range
        converted_audio = converted_audio * stress_enhance
        
        # 9. Enhanced natural speech characteristics
        # Extract energy envelope for improved prosody matching
        target_energy = librosa.feature.rms(y=target_audio)[0]
        tts_energy = librosa.feature.rms(y=tts_audio)[0]
        
        # Resample energy envelopes to the same length
        min_length = min(len(target_energy), len(tts_energy))
        target_energy = target_energy[:min_length]
        tts_energy = tts_energy[:min_length]
        
        # Normalize energy envelopes with enhanced processing
        target_energy_norm = (target_energy - np.mean(target_energy)) / np.std(target_energy)
        tts_energy_norm = (tts_energy - np.mean(tts_energy)) / np.std(tts_energy)
        
        # Blend energy envelopes for enhanced natural prosody
        blended_energy = (1 - prosody_blend) * tts_energy_norm + prosody_blend * target_energy_norm
        
        # Rescale to target energy range with improved dynamic range
        target_energy_mean = np.mean(target_energy)
        target_energy_std = np.std(target_energy)
        blended_energy = blended_energy * target_energy_std + target_energy_mean
        
        # Resample blended energy to match converted audio length
        blended_energy = np.interp(
            np.linspace(0, len(blended_energy), len(converted_audio)),
            np.arange(len(blended_energy)),
            blended_energy
        )
        
        # Apply energy envelope to converted audio with enhanced processing
        converted_audio = converted_audio * blended_energy
        
        # 10. Enhanced breath and micro-variations
        # Extract breath characteristics from target with improved processing
        target_breath = librosa.effects.preemphasis(target_audio, coef=0.98)  # Increased coefficient
        breath_envelope = np.abs(librosa.stft(target_breath, n_fft=4096, hop_length=512))  # Increased FFT size
        breath_energy = np.mean(breath_envelope, axis=0)
        
        # Normalize breath energy with enhanced processing
        breath_energy = (breath_energy - np.min(breath_energy)) / (np.max(breath_energy) - np.min(breath_energy))
        
        # Resample breath energy to match converted audio length
        breath_energy = np.interp(
            np.linspace(0, len(breath_energy), len(converted_audio)),
            np.arange(len(breath_energy)),
            breath_energy
        )
        
        # Apply breath characteristics to converted audio with enhanced processing
        breath_scale = 0.15  # Increased breath effect
        converted_audio = converted_audio * (1 + breath_scale * breath_energy)
        
        # Add enhanced micro-variations for improved naturalness
        micro_variations = np.random.normal(0, 0.015, len(converted_audio))  # Increased variation
        converted_audio = converted_audio * (1 + micro_variations)
        
        # 11. Add formant characteristics
        # Extract formant frequencies from target
        target_formants = librosa.effects.preemphasis(target_audio, coef=0.95)
        target_formant_envelope = np.abs(librosa.stft(target_formants, n_fft=4096, hop_length=512))
        
        # Normalize formant envelope
        formant_energy = np.mean(target_formant_envelope, axis=0)
        formant_energy = (formant_energy - np.min(formant_energy)) / (np.max(formant_energy) - np.min(formant_energy))
        
        # Resample formant energy to match converted audio length
        formant_energy = np.interp(
            np.linspace(0, len(formant_energy), len(converted_audio)),
            np.arange(len(formant_energy)),
            formant_energy
        )
        
        # Apply formant characteristics
        formant_scale = 0.1
        converted_audio = converted_audio * (1 + formant_scale * formant_energy)
        
        # 12. Add natural speech jitter and shimmer
        # Calculate jitter (pitch period variation)
        jitter = np.random.normal(0, 0.01, len(converted_audio))
        converted_audio = converted_audio * (1 + jitter)
        
        # Calculate shimmer (amplitude variation)
        shimmer = np.random.normal(0, 0.02, len(converted_audio))
        converted_audio = converted_audio * (1 + shimmer)
        
        # 13. Final audio enhancement with improved naturalness
        converted_audio = self.audio_processor.enhance_audio(
            converted_audio, 
            trim_silence=True,
            apply_lowpass=True,
            apply_preemphasis=True
        )
        
        # 14. Save the converted audio
        if output_path is None:
            # Generate output path based on input filenames
            tts_name = os.path.splitext(os.path.basename(tts_audio_path))[0]
            target_name = os.path.splitext(os.path.basename(target_voice_path))[0]
            output_path = os.path.join(self.converted_dir, f"{tts_name}_as_{target_name}.wav")
        
        # Save final audio
        self.audio_processor.save_audio(converted_audio, output_path)
        print(f"Converted audio saved to {output_path}")
        
        return converted_audio, output_path
    
    def debug_conversion(self, tts_audio_path, target_voice_path):

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

        print(f"Training from epoch {start_epoch} to {self.num_epochs}")
        return self.train(start_epoch=start_epoch)

def get_available_files(directory, extension='.wav'):

    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return []
    
    files = [f for f in os.listdir(directory) if f.endswith(extension)]
    return files

def main():

    data_dir = '/content/drive/MyDrive/VC'
    output_dir = '/content/drive/MyDrive/VoiceGenv4/Output'
    model_dir = '/content/drive/MyDrive/VoiceGenv4/Output/model'
    target_voice_dir = '/content/drive/MyDrive/VoiceGenv4/NewGenSample'
    tts_audio_dir = '/content/drive/MyDrive/VoiceGenv4/Output/generatedTTS'
    
    print(f"Using directories:")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model directory: {model_dir}")
    print(f"Target voice directory: {target_voice_dir}")
    print(f"TTS audio directory: {tts_audio_dir}")

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
        num_epochs=25,  # Increased to 25 for better quality
        batch_size=8,
        learning_rate=0.0001
    )
    
    # Determine action: train or convert (I was having problem where training stopped in between, so thisa allowed me to resume, also I could manually stop in the begnning to see if the model is working o r not)
    print("\nWhat would you like to do?")
    print("1. Train a new model")
    print("2. Resume training from the latest checkpoint")
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