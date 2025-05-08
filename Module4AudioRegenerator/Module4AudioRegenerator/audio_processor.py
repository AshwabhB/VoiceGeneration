import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
from scipy import signal
import torch
from scipy.signal import find_peaks

# Try to import parselmouth, but provide a fallback if it fails
try:
    import parselmouth  # For pitch extraction
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    print("Warning: parselmouth not available, using fallback pitch extraction")
    PARSELMOUTH_AVAILABLE = False

class AudioProcessor:
    
    def __init__(self, sample_rate=22050, n_mels=80, hop_length=256, win_length=1024, 
                 fmin=20, fmax=8000, griffin_lim_iters=128):
 
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
        self.fmin = fmin
        self.fmax = fmax
        self.griffin_lim_iters = griffin_lim_iters
    
    def clean_audio(self, audio):
        """Clean audio by removing non-finite values and normalizing"""
        # Convert to float32 if not already
        audio = audio.astype(np.float32)
        
        # Replace non-finite values with zeros
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Remove DC offset
        audio = audio - np.mean(audio)
        
        # Normalize audio to prevent extreme values
        audio = librosa.util.normalize(audio)
        
        # Clip to prevent any remaining extreme values
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    def load_audio(self, audio_path):
        """Load and clean audio file with enhanced validation"""
        try:
            print(f"Loading audio file: {audio_path}")
            
            # Load audio with librosa
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            print(f"Initial audio stats - Shape: {audio.shape}, Mean: {np.mean(audio):.4f}, Std: {np.std(audio):.4f}")
            
            # Check for non-finite values before cleaning
            non_finite_count = np.sum(~np.isfinite(audio))
            if non_finite_count > 0:
                print(f"Warning: Found {non_finite_count} non-finite values before cleaning")
            
            # Convert to float32 if not already
            audio = audio.astype(np.float32)
            
            # Replace non-finite values with zeros
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Remove DC offset
            audio = audio - np.mean(audio)
            
            # Normalize audio to prevent extreme values
            audio = librosa.util.normalize(audio)
            
            # Clip to prevent any remaining extreme values
            audio = np.clip(audio, -1.0, 1.0)
            
            # Additional cleaning steps
            # 1. Remove any remaining NaN or Inf values
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 2. Remove any remaining extreme values
            audio = np.clip(audio, -1.0, 1.0)
            
            # 3. Apply a gentle low-pass filter to remove any high-frequency artifacts
            nyquist = 0.5 * self.sample_rate
            normal_cutoff = 0.9  # Keep 90% of the frequency range
            b, a = signal.butter(4, normal_cutoff, btype='low')
            audio = signal.filtfilt(b, a, audio)
            
            # 4. Final normalization
            audio = librosa.util.normalize(audio)
            
            # Validate audio after cleaning
            if not np.isfinite(audio).all():
                print("Warning: Audio still contains non-finite values after cleaning")
                print(f"Non-finite count: {np.sum(~np.isfinite(audio))}")
                print(f"NaN count: {np.sum(np.isnan(audio))}")
                print(f"Inf count: {np.sum(np.isinf(audio))}")
                
                # Last resort: replace any remaining non-finite values with zeros
                audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Final validation
            if not np.isfinite(audio).all():
                raise ValueError("Failed to clean audio: non-finite values remain after all cleaning steps")
            
            print(f"Final audio stats - Shape: {audio.shape}, Mean: {np.mean(audio):.4f}, Std: {np.std(audio):.4f}")
            return audio
            
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise
    
    def audio_to_mel(self, audio):
        """Convert audio to mel spectrogram with validation"""
        try:
            # Clean audio first
            audio = self.clean_audio(audio)
            
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.win_length,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax
            )
            
            # Convert to log scale
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Clean mel spectrogram
            mel_spec = np.nan_to_num(mel_spec, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize mel spectrogram
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
            
            return mel_spec
            
        except Exception as e:
            print(f"Error converting audio to mel spectrogram: {str(e)}")
            raise
    
    def mel_to_audio(self, mel_spec, griffin_lim_iters=32):
        """Convert mel spectrogram back to audio with validation"""
        try:
            # Clean mel spectrogram
            mel_spec = np.nan_to_num(mel_spec, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Convert from log scale to power
            mel_power = librosa.db_to_power(mel_spec)
            
            # Clean power spectrogram
            mel_power = np.nan_to_num(mel_power, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Convert to audio using Griffin-Lim
            audio = librosa.feature.inverse.mel_to_audio(
                mel_power,
                sr=self.sample_rate,
                n_fft=self.win_length,
                hop_length=self.hop_length,
                n_iter=griffin_lim_iters
            )
            
            # Clean the reconstructed audio
            audio = self.clean_audio(audio)
            
            return audio
            
        except Exception as e:
            print(f"Error converting mel spectrogram to audio: {str(e)}")
            raise
    
    def extract_voice_stats(self, audio):
        """Extract voice statistics with validation"""
        try:
            # Clean audio first
            audio = self.clean_audio(audio)
            
            # Extract mel spectrogram
            mel_spec = self.audio_to_mel(audio)
            
            # Calculate statistics
            stats = {
                'mean': float(np.mean(mel_spec)),
                'std': float(np.std(mel_spec)),
                'min': float(np.min(mel_spec)),
                'max': float(np.max(mel_spec))
            }
            
            return stats, mel_spec
            
        except Exception as e:
            print(f"Error extracting voice statistics: {str(e)}")
            raise
    
    def apply_lowpass_filter(self, audio, cutoff=3500):
        
        # Design Butterworth low-pass filter
        nyquist = 0.5 * self.sample_rate
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(6, normal_cutoff, btype='low')
        
        # Apply filter
        filtered_audio = signal.filtfilt(b, a, audio)
        
        return filtered_audio
    
    def enhance_audio(self, audio, trim_silence=True, apply_lowpass=True, apply_preemphasis=True):
        """Enhance audio with validation"""
        try:
            # Clean audio first
            audio = self.clean_audio(audio)
            
            if trim_silence:
                audio = librosa.effects.trim(audio, top_db=20)[0]
            
            if apply_lowpass:
                audio = librosa.effects.preemphasis(audio, coef=0.97)
            
            if apply_preemphasis:
                audio = librosa.effects.preemphasis(audio, coef=0.95)
            
            # Final cleaning
            audio = self.clean_audio(audio)
            
            return audio
            
        except Exception as e:
            print(f"Error enhancing audio: {str(e)}")
            raise
    
    def split_audio_into_chunks(self, mel_spec, chunk_size=8192, overlap=0.5):
        
        chunks = []
        positions = []
        
        # Calculate chunk parameters
        overlap_size = int(chunk_size * overlap)
        step = chunk_size - overlap_size
        
        # Ensure we have at least minimum width
        if mel_spec.shape[1] <= chunk_size:
            # Pad if smaller than chunk size
            if mel_spec.shape[1] < chunk_size:
                pad_width = chunk_size - mel_spec.shape[1]
                padded_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)))
                chunks.append(('single', padded_spec, 0, mel_spec.shape[1]))
            else:
                chunks.append(('single', mel_spec, 0, mel_spec.shape[1]))
        else:
            # Process in overlapping chunks for large spectrograms
            for i in range(0, mel_spec.shape[1], step):
                position = i
                
                if i + chunk_size > mel_spec.shape[1]:
                    # Last chunk - handle differently
                    chunk_start = max(0, mel_spec.shape[1] - chunk_size)
                    chunk = mel_spec[:, chunk_start:mel_spec.shape[1]]
                    
                    # Pad if needed
                    if chunk.shape[1] < chunk_size:
                        chunk = np.pad(chunk, ((0, 0), (0, chunk_size - chunk.shape[1])))
                    
                    # Store last chunk
                    offset = mel_spec.shape[1] - chunk_start
                    chunks.append(('end', chunk, chunk_start, chunk_start + offset))
                    positions.append((position, mel_spec.shape[1]))
                    break
                else:
                    # Regular chunk
                    chunk = mel_spec[:, i:i+chunk_size]
                    
                    if i == 0:
                        # First chunk
                        chunks.append(('start', chunk, 0, chunk_size))
                    else:
                        # Middle chunk
                        chunks.append(('middle', chunk, i, i + chunk_size))
                    
                    positions.append((position, min(position + chunk_size, mel_spec.shape[1])))
        
        return chunks, positions
    
    def combine_chunks(self, chunks, original_length):

        # Initialize output array
        combined = np.zeros((self.n_mels, original_length))
        
        # Handle single chunk case
        if len(chunks) == 1 and chunks[0][0] == 'single':
            _, chunk, _, end_idx = chunks[0]
            combined[:, :end_idx] = chunk[:, :end_idx]
            return combined
        
        # Process multiple chunks with crossfading
        for i, (chunk_type, chunk, start_idx, end_idx) in enumerate(chunks):
            if chunk_type == 'start':
                # For first chunk, copy all but overlap region at end
                overlap_end = int((end_idx - start_idx) * 0.5)
                copy_end = end_idx - overlap_end
                combined[:, start_idx:copy_end] = chunk[:, :(copy_end - start_idx)]
                
                # Apply fade out to overlap region
                fade_out = np.linspace(1, 0, overlap_end)
                for j in range(overlap_end):
                    combined[:, copy_end + j] = chunk[:, (copy_end - start_idx) + j] * fade_out[j]
                
            elif chunk_type == 'middle':
                # For middle chunks, apply crossfade
                chunk_width = end_idx - start_idx
                overlap_size = int(chunk_width * 0.5)
                
                # Apply fade in to first part
                fade_in = np.linspace(0, 1, overlap_size)
                for j in range(overlap_size):
                    idx = start_idx + j
                    if idx < combined.shape[1]:
                        combined[:, idx] = combined[:, idx] * (1 - fade_in[j]) + chunk[:, j] * fade_in[j]
                
                # Copy middle part directly
                mid_start = start_idx + overlap_size
                mid_end = end_idx - overlap_size
                copy_length = mid_end - mid_start
                if copy_length > 0:
                    combined[:, mid_start:mid_end] = chunk[:, overlap_size:(overlap_size + copy_length)]
                
                # Apply fade out to last part
                fade_out = np.linspace(1, 0, overlap_size)
                for j in range(overlap_size):
                    idx = mid_end + j
                    if idx < combined.shape[1]:
                        combined[:, idx] = combined[:, idx] * (1 - fade_out[j]) + chunk[:, (overlap_size + copy_length) + j] * fade_out[j]
                
            elif chunk_type == 'end':
                # For last chunk, apply crossfade with previous
                overlap_size = int((end_idx - start_idx) * 0.5)
                
                # Apply fade in to first part
                fade_in = np.linspace(0, 1, overlap_size)
                for j in range(overlap_size):
                    idx = start_idx + j
                    if idx < combined.shape[1]:
                        combined[:, idx] = combined[:, idx] * (1 - fade_in[j]) + chunk[:, j] * fade_in[j]
                
                # Copy remaining part directly
                copy_start = start_idx + overlap_size
                copy_length = min(end_idx, combined.shape[1]) - copy_start
                if copy_length > 0:
                    combined[:, copy_start:copy_start + copy_length] = chunk[:, overlap_size:overlap_size + copy_length]
                
        return combined

    def save_audio(self, audio, path):
        """Save audio with validation"""
        try:
            # Clean audio first
            audio = self.clean_audio(audio)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Normalize and save
            audio = librosa.util.normalize(audio)
            sf.write(path, audio, self.sample_rate)
            
        except Exception as e:
            print(f"Error saving audio: {str(e)}")
            raise

    def extract_pitch(self, audio):
        """Extract pitch with validation"""
        try:
            # Clean audio first
            audio = self.clean_audio(audio)
            
            if PARSELMOUTH_AVAILABLE:
                # Use Parselmouth/Praat if available
                sound = parselmouth.Sound(audio, self.sample_rate)
                pitch = sound.to_pitch()
                pitch_values = pitch.selected_array['frequency']
                voiced_frames = pitch_values > 0
            else:
                # Fallback method using librosa
                pitch_values, voiced_flag = librosa.core.piptrack(
                    y=audio, 
                    sr=self.sample_rate,
                    n_fft=self.win_length,
                    hop_length=self.hop_length,
                    fmin=self.fmin,
                    fmax=self.fmax
                )
                
                # Get the most prominent pitch in each frame
                pitch_values = np.mean(pitch_values, axis=0)
                voiced_frames = pitch_values > 0.5  # Simple thresholding
            
            return pitch_values, voiced_frames
            
        except Exception as e:
            print(f"Error extracting pitch: {str(e)}")
            raise

    def extract_duration(self, audio):

        # Get total duration in seconds
        total_duration = len(audio) / self.sample_rate
        
        # Detect speech segments using energy
        energy = librosa.feature.rms(y=audio)[0]
        energy_threshold = np.mean(energy) * 0.5
        
        # Find speech segments
        speech_segments = energy > energy_threshold
        speech_duration = np.sum(speech_segments) * (self.hop_length / self.sample_rate)
        
        # Calculate speech rate (words per second, approximate)
        # This is a rough estimate based on typical speech patterns
        speech_rate = len(find_peaks(energy)[0]) / total_duration
        
        duration_stats = {
            'total_duration': total_duration,
            'speech_duration': speech_duration,
            'silence_duration': total_duration - speech_duration,
            'speech_rate': speech_rate
        }
        
        return duration_stats

    def extract_stress(self, audio):

        # Extract energy envelope
        energy = librosa.feature.rms(y=audio)[0]
        
        # Find peaks in energy (potential stressed syllables)
        peaks, _ = find_peaks(energy, height=np.mean(energy) * 1.2)
        
        # Calculate peak statistics
        peak_heights = energy[peaks]
        peak_intervals = np.diff(peaks) * (self.hop_length / self.sample_rate)
        
        stress_features = {
            'peak_count': len(peaks),
            'mean_peak_height': float(np.mean(peak_heights)) if len(peak_heights) > 0 else 0.0,
            'std_peak_height': float(np.std(peak_heights)) if len(peak_heights) > 0 else 0.0,
            'mean_peak_interval': float(np.mean(peak_intervals)) if len(peak_intervals) > 0 else 0.0,
            'std_peak_interval': float(np.std(peak_intervals)) if len(peak_intervals) > 0 else 0.0
        }
        
        return stress_features

    def extract_linguistic_features(self, audio):
        """Extract linguistic features with validation"""
        try:
            # Clean audio first
            audio = self.clean_audio(audio)
            
            # Extract pitch
            pitch, voiced = self.extract_pitch(audio)
            
            # Extract duration features
            duration_features = {
                'speech_rate': float(np.mean(voiced)),
                'pause_duration': float(np.mean(~voiced)),
                'word_duration': float(np.mean(np.diff(np.where(voiced)[0])))
            }
            
            # Extract stress features
            stress_features = {
                'mean_peak_height': float(np.mean(pitch[voiced])),
                'stress_pattern': float(np.std(pitch[voiced])),
                'emphasis_marker': float(np.max(pitch[voiced]))
            }
            
            # Extract prosody features
            prosody_features = {
                'pitch_contour': float(np.mean(np.diff(pitch[voiced]))),
                'energy_envelope': float(np.mean(librosa.feature.rms(y=audio)[0])),
                'timing_pattern': float(np.std(np.diff(np.where(voiced)[0]))),
                'rhythm_marker': float(np.mean(np.abs(np.diff(pitch[voiced]))))
            }
            
            return {
                'duration': duration_features,
                'stress': stress_features,
                'prosody': prosody_features
            }
            
        except Exception as e:
            print(f"Error extracting linguistic features: {str(e)}")
            raise

# Utility function for blending source and target statistics
def blend_statistics(source_stats, target_stats, blend_ratio=0.7):
    blended_stats = {}
    for key in source_stats.keys():
        if key in target_stats:
            blended_stats[key] = (1 - blend_ratio) * source_stats[key] + blend_ratio * target_stats[key]
        else:
            blended_stats[key] = source_stats[key]
            
    return blended_stats
