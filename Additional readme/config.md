# Voice Conversion System Configuration

This document details all the parameters and hyperparameters used in the voice conversion system, along with their reasoning and purpose.

## 1. Audio Processing Parameters

These parameters are defined in `audio_processor.py` and control the audio processing pipeline:

| Parameter | Value | Description | Reasoning |
|-----------|-------|-------------|-----------|
| `sample_rate` | 22050 | Audio sample rate | Standard sample rate for speech processing, balancing quality and computational efficiency |
| `n_mels` | 80 | Number of mel bands | Provides good frequency resolution for voice representation |
| `hop_length` | 256 | Frame shift for spectrogram | Provides good temporal resolution for speech analysis |
| `win_length` | 1024 | Window size for spectrogram | Balances time and frequency resolution |
| `fmin` | 20 | Minimum frequency | Covers human speech range |
| `fmax` | 8000 | Maximum frequency | Covers most speech frequencies |
| `griffin_lim_iters` | 128 | Phase reconstruction iterations | Ensures good audio quality in reconstruction |

## 2. Generator Parameters

These parameters are defined in `improved_generator.py` and control the generator architecture:

| Parameter | Value | Description | Reasoning |
|-----------|-------|-------------|-----------|
| `input_channels` | 80 | Input channel count | Matches number of mel bands |
| `output_channels` | 80 | Output channel count | Matches number of mel bands |
| `hidden_channels` | 512 | Hidden layer size | Provides sufficient capacity for voice conversion |
| `scale` | 3.0 | Initial scaling factor | Targets standard deviation of ~3 for normalized output |
| `bias` | 0.0 | Initial bias | Targets mean of ~0 for normalized output |

## 3. Training Parameters

These parameters are defined in `voice_converter_v2.py` and control the training process:

| Parameter | Value | Description | Reasoning |
|-----------|-------|-------------|-----------|
| `batch_size` | 8 | Training batch size | Balances memory usage and training stability |
| `num_epochs` | 25 | Number of training epochs | Provides sufficient training time for convergence |
| `learning_rate` | 0.0001 | Optimizer learning rate | Ensures stable training |
| `betas` | (0.5, 0.999) | Adam optimizer parameters | Provides good momentum and scaling |
| `weight_decay` | 1e-5 | L2 regularization | Prevents overfitting |
| `g_steps` | 2 | Generator steps per discriminator step | Helps generator learn faster |
| `alpha` | 0.05 | Global statistics update rate | Provides stable updates |
| `max_norm` | 1.0 | Gradient clipping threshold | Prevents exploding gradients |

## 4. Loss Function Parameters

These parameters control the loss functions used in training:

| Parameter | Value | Description | Reasoning |
|-----------|-------|-------------|-----------|
| `L1_weight` | 25 | Weight for L1 loss | Emphasizes audio quality in generator |
| `label_smoothing` | 0.1 | Label smoothing factor | Improves training stability |
| `d_loss_threshold` | 0.3 | Discriminator update threshold | Prevents discriminator from becoming too strong |

## 5. Global Statistics

These parameters are used for mel spectrogram normalization:

| Parameter | Value | Description | Reasoning |
|-----------|-------|-------------|-----------|
| `MEL_MEAN` | -30.0 | Initial mean | Provides reasonable default for normalization |
| `MEL_STD` | 20.0 | Initial standard deviation | Provides reasonable default for normalization |
| `MEL_MIN` | -80.0 | Initial minimum value | Covers typical mel spectrogram range |
| `MEL_MAX` | 0.0 | Initial maximum value | Covers typical mel spectrogram range |

## 6. Learning Rate Scheduler Parameters

These parameters control the learning rate scheduling:

| Parameter | Value | Description | Reasoning |
|-----------|-------|-------------|-----------|
| `factor` | 0.7 | Learning rate reduction factor | Provides gradual learning rate reduction |
| `patience` | 5 | Epochs before reduction | Allows sufficient time for improvement |

## 7. Network Architecture

These parameters define the structure of the generator and discriminator networks:

### Generator Architecture
| Component | Details | Description | Reasoning |
|-----------|---------|-------------|-----------|
| Encoder | 4 convolutional layers | Channels: 512 → 1024 → 2048 → 4096 | Progressive feature extraction with increasing capacity |
| Residual Blocks | 3 blocks | Each with skip connections | Improves gradient flow and feature reuse |
| Decoder | 3 transposed convolutional layers | Channels: 4096 → 2048 → 1024 → 512 | Symmetric to encoder for proper reconstruction |
| Output Layer | 1 convolutional layer | Output channels: 80 | Matches mel spectrogram dimensions |

### Discriminator Architecture
| Component | Details | Description | Reasoning |
|-----------|---------|-------------|-----------|
| Input Layer | 1 convolutional layer | Input channels: 80 | Matches generator output dimensions |
| Hidden Layers | 4 convolutional layers | Channels: 512 → 1024 → 2048 → 4096 | Progressive feature analysis with increasing complexity |
| Output Layer | 1 convolutional layer | Output channels: 1 | Binary classification for real/fake |
| Activation | LeakyReLU (slope=0.2) | Used in all layers except output | Prevents dead neurons and provides stable gradients |
| Final Activation | Sigmoid | For binary classification | Provides probability output for real/fake discrimination |
| Normalization | Spectral Normalization | Applied to all convolutional layers | Ensures stable GAN training |

### Layer Parameters
| Parameter | Value | Description | Reasoning |
|-----------|-------|-------------|-----------|
| Kernel Size | 4x4 | Used in discriminator convolutions | Balances receptive field and computational efficiency |
| Stride | 2 | For downsampling/upsampling | Provides efficient spatial reduction/expansion |
| Padding | 1 | Maintains spatial dimensions | Preserves input size through convolutions |
| Batch Size | 8 | Training batch size | Balances memory usage and training stability |
| Learning Rate | 0.0001 | Optimizer learning rate | Ensures stable training with good convergence |
| Label Smoothing | 0.1 | Improves training stability | Prevents discriminator from becoming too confident |
| Loss Threshold | 0.3 | Prevents discriminator from becoming too strong | Maintains balanced GAN training |
| Weight Decay | 1e-5 | L2 regularization | Prevents overfitting while maintaining model capacity |
| Gradient Clipping | 1.0 | Prevents exploding gradients | Ensures numerical stability during training |

These architectural choices provide:
- Sufficient capacity for voice conversion
- Stable training dynamics
- Good feature extraction
- Efficient computation
- Reliable convergence

## Parameter Selection Rationale

The parameters have been carefully chosen based on:

1. **Speech Processing Best Practices**
   - Standard sample rates and window sizes for speech
   - Appropriate frequency ranges for human voice
   - Optimal mel band count for voice representation

2. **GAN Training Stability**
   - Balanced generator and discriminator training
   - Appropriate learning rates and batch sizes
   - Gradient clipping and regularization

3. **Voice Conversion Quality**
   - Sufficient model capacity
   - Appropriate loss function weights
   - Good audio reconstruction parameters

4. **Computational Efficiency**
   - Memory-efficient batch sizes
   - Optimized processing parameters
   - Reasonable training duration

5. **Training Convergence**
   - Stable learning rate scheduling
   - Appropriate regularization
   - Balanced training steps

These parameters work together to provide:
- High-quality voice conversion
- Stable training process
- Efficient resource usage
- Good generalization
- Reliable convergence 