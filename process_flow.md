```mermaid
graph TD
    A[Start] --> B[Run voice_converter_runner.py]
    B --> C{Select Option}
    
    C -->|Option 1: Train New Model| D[Training Process]
    C -->|Option 2: Resume Training| E[Resume Training]
    C -->|Option 3: Convert Voice| F[Voice Conversion]
    C -->|Option 4: Debug| G[Debug Mode]
    
    D --> D1[Initialize Dataset]
    D1 --> D2[Create DataLoader]
    D2 --> D3[Initialize Models]
    D3 --> D4[Setup Optimizers]
    D4 --> D5[Training Loop]
    D5 --> D6[Save Checkpoint]
    
    E --> E1[Load Latest Checkpoint]
    E1 --> E2[Restore States]
    E2 --> E3[Continue Training]
    
    F --> F1[Load Trained Model]
    F1 --> F2[Select Target Voice]
    F2 --> F3[Select TTS Audio]
    F3 --> F4[Process Audio]
    F4 --> F5[Post-processing]
    F5 --> F6[Save Output]
    
    G --> G1[Create Debug Directory]
    G1 --> G2[Save Original Files]
    G2 --> G3[Extract Features]
    G3 --> G4[Process Test Chunk]
    G4 --> G5[Save Debug Info]
    
    subgraph Training Process
    D1
    D2
    D3
    D4
    D5
    D6
    end
    
    subgraph Voice Conversion
    F1
    F2
    F3
    F4
    F5
    F6
    end
    
    subgraph Debug Mode
    G1
    G2
    G3
    G4
    G5
    end
```

## Process Flow Description

### Main Options
1. **Training Process (Option 1)**
   - Initialize dataset and data loader
   - Set up neural network models
   - Train with batch processing
   - Save checkpoints periodically

2. **Resume Training (Option 2)**
   - Load latest checkpoint
   - Restore model and optimizer states
   - Continue training from saved state

3. **Voice Conversion (Option 3)**
   - Load trained model
   - Process target voice and TTS audio
   - Apply voice conversion
   - Save converted output

4. **Debug Mode (Option 4)**
   - Create debug environment
   - Save intermediate outputs
   - Analyze conversion process
   - Generate detailed logs

### Key Components
- **Audio Processing**: Handles all audio file operations
- **Neural Networks**: Generator and discriminator models
- **Data Management**: Dataset handling and normalization
- **Debug Tools**: Analysis and logging utilities 