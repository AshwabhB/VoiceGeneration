# Improved Generator README

## Overview
This file contains the improved neural network architectures for the voice conversion system, including the generator, discriminator, and related components.

## Main Components

### Classes
1. `LearnableScaling`
2. `ResidualBlock`
3. `ImprovedGenerator`
4. `ImprovedDiscriminator`
5. `GlobalStats`

## Detailed Component Descriptions

### LearnableScaling Class
A learnable scaling layer that replaces traditional Tanh activation functions for better dynamic range control.

The scale parameter in the LearnableScaling class is initialized to 3.0 because it's designed to match the typical range of mel spectrograms, where a standard deviation of around 3 is common. If we change it to 1.0, here's what would happen:
   - The output range would be reduced by a factor of 3
   - The dynamic range of the generated mel spectrograms would be smaller
   - The model might have less flexibility in representing the full range of audio features

#### Methods:
- `__init__`: Initializes scaling parameters
- `forward`: Applies learnable scaling to input

### ResidualBlock Class
A residual block implementation for improved training stability and gradient flow, prevents vanishing gradient problem.

#### How it Works:
1. **Structure**:
   - Two convolutional layers with batch normalization
   - ReLU activation functions
   - Skip connection that preserves the original input

2. **Key Components**:
   - First convolutional layer with kernel size 3 and padding 1
   - Batch normalization after each convolution
   - ReLU activation for non-linearity
   - Skip connection that adds the original input to the transformed output

3. **Forward Pass Process**:
   - Original input is stored as residual
   - Input passes through first conv + BN + ReLU
   - Then through second conv + BN
   - Original input is added back (skip connection)
   - Final ReLU activation is applied

4. **Benefits**:
   - Improved gradient flow through the network
   - Better preservation of important features
   - More stable training of deep networks
   - Prevention of vanishing gradients
   - Enhanced feature transformation capabilities

#### Methods:
- `__init__`: Initializes convolutional layers and batch normalization
- `forward`: Implements residual connection with skip connection

### ImprovedGenerator Class
An enhanced generator architecture with residual connections and learnable scaling.

#### Methods:
- `__init__`: Initializes encoder, residual blocks, and decoder
- `forward`: Implements the forward pass through the network

#### Architecture:
1. Encoder:
   - Multiple convolutional layers with increasing channels
   - Batch normalization and LeakyReLU activation
2. Residual Blocks:
   - Three residual blocks for better feature extraction
3. Decoder:
   - Transposed convolutions for upsampling
   - Batch normalization and ReLU activation
4. Output Scaling:
   - Learnable scaling layer for dynamic range control

### ImprovedDiscriminator Class
An enhanced discriminator with spectral normalization for better GAN training stability.

#### Methods:
- `__init__`: Initializes convolutional layers with spectral normalization
- `forward`: Implements the forward pass through the network

#### Architecture:
1. Multiple convolutional layers with spectral normalization
2. Batch normalization and LeakyReLU activation
3. Final convolutional layer with sigmoid activation

### GlobalStats Class
Manages global statistics for mel spectrogram normalization for more consistent processing.

#### Methods:
- `update_from_batch`: Updates global statistics from a batch of mel spectrograms
- `normalize`: Normalizes mel spectrogram using global statistics
- `denormalize`: Denormalizes mel spectrogram using global statistics

#### Statistics Tracked:
- MEL_MEAN: Mean of mel spectrograms
- MEL_STD: Standard deviation of mel spectrograms
- MEL_MIN: Minimum value of mel spectrograms
- MEL_MAX: Maximum value of mel spectrograms

## Key Features

### Generator Improvements
- Residual connections for better gradient flow
- Learnable scaling for dynamic range control
- Batch normalization for stable training
- Multiple convolutional layers for better feature extraction

### Discriminator Improvements
- Spectral normalization for stable GAN training
- Multiple convolutional layers for better discrimination
- Batch normalization for stable training
- LeakyReLU activation for better gradient flow

### Normalization Features
- Global statistics tracking
- Moving average updates
- Normalization and denormalization functions
- Statistics blending for voice conversion 