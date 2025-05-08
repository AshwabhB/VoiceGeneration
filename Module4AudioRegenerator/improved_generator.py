import torch
import torch.nn as nn

class LearnableScaling(nn.Module):
    """Learnable scaling layer that replaces Tanh for better dynamic range"""
    def __init__(self, channels=80):
        super(LearnableScaling, self).__init__()
        # Initialize with reasonable defaults that approximately match mel spectrogram range
        self.scale = nn.Parameter(torch.ones(1, channels, 1) * 3.0)  # Target standard deviation ~3
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))        # Target mean ~0
        
    def forward(self, x):
        return x * self.scale + self.bias

class ResidualBlock(nn.Module):
    """Residual block for improved training stability"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = self.relu(out)
        return out

class ImprovedGenerator(nn.Module):
    """Improved generator with residual connections and learnable scaling"""
    def __init__(self, input_channels=80, output_channels=80, hidden_channels=512, linguistic_dim=10):
        super(ImprovedGenerator, self).__init__()

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
        
        # Linguistic feature processing
        self.linguistic_processor = nn.Sequential(
            nn.Linear(linguistic_dim, hidden_channels * 8),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_channels * 8, hidden_channels * 8),
            nn.LeakyReLU(0.2)
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_channels * 8),
            ResidualBlock(hidden_channels * 8),
            ResidualBlock(hidden_channels * 8)
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
        )
        
        # Learnable output scaling instead of fixed Tanh
        self.output_scaling = LearnableScaling(output_channels)

    def forward(self, x, linguistic_features=None):
        # Process mel spectrogram
        x = self.encoder(x)
        
        # Process linguistic features if provided
        if linguistic_features is not None:
            # Process linguistic features
            linguistic_processed = self.linguistic_processor(linguistic_features)
            # Expand linguistic features to match spatial dimensions
            linguistic_processed = linguistic_processed.unsqueeze(-1).expand(-1, -1, x.size(-1))
            # Combine with mel features
            x = x + linguistic_processed
        
        x = self.res_blocks(x)
        x = self.decoder(x)
        x = self.output_scaling(x)
        return x

class ImprovedDiscriminator(nn.Module):
    """Improved discriminator with spectral normalization for better stability"""
    def __init__(self, input_channels=80, hidden_channels=512):
        super(ImprovedDiscriminator, self).__init__()

        # Apply spectral normalization for stable GAN training
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(input_channels, hidden_channels, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv1d(hidden_channels * 2, hidden_channels * 4, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm1d(hidden_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv1d(hidden_channels * 4, hidden_channels * 8, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm1d(hidden_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv1d(hidden_channels * 8, 1, kernel_size=4, stride=1, padding=0)),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Global normalization statistics for more consistent processing
class GlobalStats:
    """Global statistics for mel spectrogram normalization"""
    # These values are reasonable defaults based on common mel spectrogram ranges
    # Will be updated during training with actual dataset statistics
    MEL_MEAN = -30.0
    MEL_STD = 20.0
    MEL_MIN = -80.0
    MEL_MAX = 0.0
    
    @staticmethod
    def update_from_batch(mel_batch):
        """Update global statistics from a batch of mel spectrograms"""
        # Use moving average to update statistics
        alpha = 0.05  # Small update rate for stability
        
        batch_mean = mel_batch.mean().item()
        batch_std = mel_batch.std().item()
        batch_min = mel_batch.min().item()
        batch_max = mel_batch.max().item()
        
        # Update global stats
        GlobalStats.MEL_MEAN = (1-alpha) * GlobalStats.MEL_MEAN + alpha * batch_mean
        GlobalStats.MEL_STD = (1-alpha) * GlobalStats.MEL_STD + alpha * batch_std
        GlobalStats.MEL_MIN = min(GlobalStats.MEL_MIN, batch_min)
        GlobalStats.MEL_MAX = max(GlobalStats.MEL_MAX, batch_max)
    
    @staticmethod
    def normalize(mel_spec):
        """Normalize mel spectrogram using global statistics"""
        return (mel_spec - GlobalStats.MEL_MEAN) / GlobalStats.MEL_STD
    
    @staticmethod
    def denormalize(norm_spec):
        """Denormalize mel spectrogram using global statistics"""
        return norm_spec * GlobalStats.MEL_STD + GlobalStats.MEL_MEAN