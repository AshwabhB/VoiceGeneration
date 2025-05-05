
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
    
    def load_audio(self, audio_path):
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        audio = librosa.util.normalize(audio)
        return audio
    
    def audio_to_mel(self, audio):
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            win_length=self.win_length,
            fmin=self.fmin,
            fmax=self.fmax,
            power=2.0  # Power 2.0 for energy instead of amplitude
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec)
        
        return mel_spec_db
    
    def mel_to_audio(self, mel_spec_db, griffin_lim_iters=None):

        if griffin_lim_iters is None:
            griffin_lim_iters = self.griffin_lim_iters
            
        # Convert from log scale back to linear
        mel_spec = librosa.db_to_power(mel_spec_db)
        
        # Convert mel spectrogram to audio with specified iterations
        audio = librosa.feature.inverse.mel_to_audio(
            mel_spec,
            sr=self.sample_rate,
            n_fft=self.win_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            fmin=self.fmin,
            fmax=self.fmax,
            power=2.0,
            n_iter=griffin_lim_iters
        )
        
        return audio
    
    def extract_voice_stats(self, audio):

        # Get mel spectrogram
        mel_spec_db = self.audio_to_mel(audio)
        
        # Extract stats
        stats = {
            'mean': float(np.mean(mel_spec_db)),
            'std': float(np.std(mel_spec_db)),
            'min': float(np.min(mel_spec_db)),
            'max': float(np.max(mel_spec_db)),
            'median': float(np.median(mel_spec_db)),
            'q1': float(np.percentile(mel_spec_db, 25)),
            'q3': float(np.percentile(mel_spec_db, 75))
        }
        
        return stats, mel_spec_db
    
    def apply_lowpass_filter(self, audio, cutoff=3500):
        
        # Design Butterworth low-pass filter
        nyquist = 0.5 * self.sample_rate
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(6, normal_cutoff, btype='low')
        
        # Apply filter
        filtered_audio = signal.filtfilt(b, a, audio)
        
        return filtered_audio
    
    def enhance_audio(self, audio, trim_silence=True, apply_lowpass=True, 
                      apply_preemphasis=True):
        
        # Apply pre-emphasis to enhance clarity
        if apply_preemphasis:
            audio = librosa.effects.preemphasis(audio, coef=0.97)
        
        # Apply low-pass filter to reduce high-frequency noise
        if apply_lowpass:
            audio = self.apply_lowpass_filter(audio)
        
        # Trim silence
        if trim_silence:
            audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Normalize amplitude
        audio = librosa.util.normalize(audio)
        
        return audio
    
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
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Normalize and save
        audio = librosa.util.normalize(audio)
        sf.write(path, audio, self.sample_rate)

    def extract_pitch(self, audio):

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

        # Extract pitch features
        pitch_values, voiced_frames = self.extract_pitch(audio)
        
        # Handle the case of no voiced frames
        if np.sum(voiced_frames) > 0:
            pitch_features = {
                'mean_pitch': float(np.mean(pitch_values[voiced_frames])),
                'std_pitch': float(np.std(pitch_values[voiced_frames])),
                'pitch_range': float(np.max(pitch_values[voiced_frames]) - np.min(pitch_values[voiced_frames])),
                'voiced_ratio': float(np.mean(voiced_frames))
            }
        else:
            pitch_features = {
                'mean_pitch': 0.0,
                'std_pitch': 0.0,
                'pitch_range': 0.0,
                'voiced_ratio': 0.0
            }
        
        # Extract duration features
        duration_features = self.extract_duration(audio)
        
        # Extract stress features
        stress_features = self.extract_stress(audio)
        
        # Combine all features
        linguistic_features = {
            'pitch': pitch_features,
            'duration': duration_features,
            'stress': stress_features
        }
        
        return linguistic_features

# Utility function for blending source and target statistics
def blend_statistics(source_stats, target_stats, blend_ratio=0.7):
    blended_stats = {}
    for key in source_stats.keys():
        if key in target_stats:
            blended_stats[key] = (1 - blend_ratio) * source_stats[key] + blend_ratio * target_stats[key]
        else:
            blended_stats[key] = source_stats[key]
            
    return blended_stats
