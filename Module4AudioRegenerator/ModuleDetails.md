# Module 4: AudioRegenerator

This module implements a conditional GAN (Generative Adversarial Network) for generating audio continuations that sound like they were spoken by the same speaker as the input audio.

## Overview

The AudioRegenerator class is responsible for:
1. Training a conditional GAN on a dataset of voice recordings
2. Generating missing audio segments that match the style and characteristics of the input audio
3. Saving the generated audio to output files

## Architecture

The implementation uses a conditional GAN with:
- A Generator network with a U-Net style architecture (encoder-decoder)
- A Discriminator network to distinguish between real and generated audio
- Mel spectrogram representation of audio for processing

## Directory Structure

```
Module4AudioRegenerator/
├── AudioRegenerator.py     # Main implementation file
├── requirements.txt        # Dependencies
├── README.md               # This file
├── output/                 # Output directory
│   ├── generated/          # Generated audio files
│   └── model/              # Saved model weights
└── NewGenSample/           # Test audio files for generation
```

## Usage

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Training the Model

```python
from AudioRegenerator import AudioRegenerator

# Initialize the regenerator
regenerator = AudioRegenerator(
    data_dir="path/to/voice/dataset",
    output_dir=None,  # Will use default: Module4AudioRegenerator/output
    model_dir=None,   # Will use default: Module4AudioRegenerator/output/model
    num_epochs=50
)

# Train the model
regenerator.train()
```

### Generating Missing Audio

```python
# Generate missing audio for a context audio file
generated_audio = regenerator.generate_missing("path/to/context_audio.wav")

# Save the generated audio
regenerator.save_generated(generated_audio, "path/to/output/result.wav")
```

### Complete Example

```python
import os
from AudioRegenerator import AudioRegenerator

# Set paths
data_dir = "path/to/voice/dataset"

# Get the directory of the current module
module_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(module_dir, "output")
model_dir = os.path.join(output_dir, "model")
new_gen_dir = os.path.join(module_dir, "NewGenSample")

# Create AudioRegenerator
regenerator = AudioRegenerator(
    data_dir=data_dir,
    output_dir=output_dir,
    model_dir=model_dir,
    num_epochs=50
)

# Train the model
regenerator.train()

# Generate missing audio for test files
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
```

## Integration with Other Modules

This module integrates with:
- Module 3 (SpeechTranscriber): For text prediction to condition the audio generation
- Module 2 (SpeakerEmbedding): For speaker feature extraction (can be extended to use speaker embeddings)

## Output

The module produces:
1. Trained model weights saved to `Module4AudioRegenerator/output/model/audio_gan_epoch_X.pth`
2. Generated audio segments saved to `Module4AudioRegenerator/output/generated/<base_name>_gen.wav`

## Notes

- Training time depends on the size of the dataset and the number of epochs
- GPU acceleration is used if available, otherwise CPU is used
- The model works best with consistent audio quality and similar speaking styles
- Add test audio files to the `Module4AudioRegenerator/NewGenSample` directory for testing 