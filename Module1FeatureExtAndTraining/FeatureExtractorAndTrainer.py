import os
import glob
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import random
import pickle

class FeatureExtractorAndTrainer:
    def __init__(self, dataset_path, features_dir='Module1FeatureExtAndTraining/output/features', model_path='Module1FeatureExtAndTraining/output/model/speaker_model.pth', sample_rate=24000, n_mfcc=13):
        self.dataset_path = dataset_path
        self.features_dir = features_dir
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        os.makedirs(self.features_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Path for saved features
        self.features_file = os.path.join(self.features_dir, 'features.npy')
        self.labels_file = os.path.join(self.features_dir, 'labels.npy')
        self.dataset_info_file = os.path.join(self.features_dir, 'dataset_info.pkl')

    def extract_features(self, audio_path, augment=False):
        """Extract features from an audio file with optional augmentation."""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Apply augmentation if requested
            if augment:
                # Add random noise
                noise_factor = 0.005
                noise = np.random.randn(len(y))
                y = y + noise_factor * noise
                
                # Time stretching
                rate = random.uniform(0.9, 1.1)
                y = librosa.effects.time_stretch(y, rate=rate)
                
                # Pitch shifting
                n_steps = random.randint(-2, 2)
                y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
            
            # Extract MFCCs with more coefficients
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # Increased from 13 to 20
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            # Extract spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_centroids_mean = np.mean(spectral_centroids)
            spectral_centroids_std = np.std(spectral_centroids)
            
            # Extract spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_rolloff_mean = np.mean(spectral_rolloff)
            
            # Extract zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_mean = np.mean(zcr)
            zcr_std = np.std(zcr)
            
            # Extract RMS energy
            rms = librosa.feature.rms(y=y)[0]
            rms_mean = np.mean(rms)
            
            # Combine all features
            features = np.concatenate([
                mfcc_mean,
                mfcc_std,
                [spectral_centroids_mean, spectral_centroids_std],
                [spectral_rolloff_mean],
                [zcr_mean, zcr_std],
                [rms_mean]
            ])
            
            return features
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None

    def extract_features_batch(self, augment=False, force_extract=False):
        """Extract features from all audio files in the dataset or load from saved files."""
        # Check if saved features exist and dataset hasn't changed
        if not force_extract and os.path.exists(self.features_file) and os.path.exists(self.labels_file) and os.path.exists(self.dataset_info_file):
            try:
                # Load dataset info
                with open(self.dataset_info_file, 'rb') as f:
                    dataset_info = pickle.load(f)
                
                # Check if dataset path matches
                if dataset_info['dataset_path'] == self.dataset_path:
                    print("Loading saved features...")
                    features = np.load(self.features_file)
                    labels = np.load(self.labels_file)
                    self.label_encoder = dataset_info['label_encoder']
                    self.scaler = dataset_info['scaler']
                    print(f"Loaded {len(features)} saved feature vectors.")
                    return features, labels
            except Exception as e:
                print(f"Error loading saved features: {e}")
                print("Will extract features again.")
        
        # Extract features if saved features don't exist or dataset has changed
        print(f"Scanning for wav files in {self.dataset_path}...")
        audio_files = glob.glob(os.path.join(self.dataset_path, '**', '*.wav'), recursive=True)
        print(f"Found {len(audio_files)} audio files.")
        
        features = []
        labels = []
        
        # Add progress reporting
        total_files = len(audio_files)
        processed_files = 0
        last_percent = -1
        
        for wav_path in audio_files:
            # Extract features
            feature_vector = self.extract_features(wav_path, augment)
            if feature_vector is not None:
                features.append(feature_vector)
                
                # Speaker label: use the first folder after dataset_path as speaker id
                rel_path = os.path.relpath(wav_path, self.dataset_path)
                speaker_id = rel_path.split(os.sep)[0]
                labels.append(speaker_id)
            
            # Update progress
            processed_files += 1
            percent = int((processed_files / total_files) * 100)
            if percent != last_percent and percent % 5 == 0:  # Report every 5%
                print(f"Progress: {percent}% ({processed_files}/{total_files} files processed)")
                last_percent = percent
        
        features = np.array(features)
        labels = np.array(labels)
        print(f"Extracted features for {len(features)} audio samples.")
        
        # Save features and dataset info
        np.save(self.features_file, features)
        np.save(self.labels_file, labels)
        
        # Save dataset info for future reference
        dataset_info = {
            'dataset_path': self.dataset_path,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler
        }
        with open(self.dataset_info_file, 'wb') as f:
            pickle.dump(dataset_info, f)
        
        return features, labels

    def train_model(self, features, labels, epochs=5000, batch_size=64, lr=0.01, patience=20, min_loss_threshold=0.5):
        # Encode speaker labels
        y = self.label_encoder.fit_transform(labels)
        num_speakers = len(np.unique(y))
        print(f"Training MLP on {num_speakers} speakers...")
        
        # Filter out speakers with too few samples (less than 2)
        unique_labels, counts = np.unique(y, return_counts=True)
        valid_labels = unique_labels[counts >= 2]
        mask = np.isin(y, valid_labels)
        
        if not np.all(mask):
            print(f"Filtering out {np.sum(~mask)} samples from speakers with too few samples.")
            features = features[mask]
            y = y[mask]
            # Re-encode labels to ensure continuous indices
            y = self.label_encoder.fit_transform(y)
            num_speakers = len(np.unique(y))
            print(f"Now training on {num_speakers} speakers with sufficient samples.")
        
        # Check for class imbalance
        unique_labels, counts = np.unique(y, return_counts=True)
        max_samples = np.max(counts)
        min_samples = np.min(counts)
        if max_samples > 3 * min_samples:
            print(f"Warning: Class imbalance detected. Max samples: {max_samples}, Min samples: {min_samples}")
            print("Consider using class weights or data augmentation.")
        
        # Normalize features
        X = self.scaler.fit_transform(features)
        # Save scaler and label encoder
        joblib.dump(self.scaler, os.path.join(self.features_dir, 'scaler.pkl'))
        joblib.dump(self.label_encoder, os.path.join(self.features_dir, 'label_encoder.pkl'))
        
        # Train/test split - removed stratify parameter
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)
        
        # Define improved MLP with residual connections and more neurons
        class ResidualBlock(nn.Module):
            def __init__(self, in_features, hidden_features):
                super().__init__()
                self.block = nn.Sequential(
                    nn.Linear(in_features, hidden_features),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_features),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_features, in_features),
                    nn.BatchNorm1d(in_features)
                )
                
            def forward(self, x):
                return x + self.block(x)
        
        class EnhancedMLP(nn.Module):
            def __init__(self, input_dim, num_classes):
                super().__init__()
                
                # Initial feature extraction
                self.feature_extractor = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.BatchNorm1d(512),
                    nn.Dropout(0.3)
                )
                
                # Residual blocks
                self.res_blocks = nn.Sequential(
                    ResidualBlock(512, 512),
                    ResidualBlock(512, 512),
                    ResidualBlock(512, 512),
                    ResidualBlock(512, 512)
                )
                
                # Classification head
                self.classifier = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Dropout(0.2),
                    nn.Linear(128, num_classes)
                )
                
            def forward(self, x):
                x = self.feature_extractor(x)
                x = self.res_blocks(x)
                x = self.classifier(x)
                return x
        
        model = EnhancedMLP(X.shape[1], num_speakers)
        
        # Use weighted loss to handle class imbalance
        class_weights = torch.ones(num_speakers)
        for i, count in enumerate(counts):
            class_weights[i] = 1.0 / count
        class_weights = class_weights / class_weights.sum() * num_speakers
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Improved learning rate scheduler with warmup
        def lr_lambda(epoch):
            if epoch < 10:  # Warmup phase
                return 0.1 + 0.9 * epoch / 10
            else:  # Cosine annealing
                return 0.5 * (1 + np.cos(np.pi * (epoch - 10) / (epochs - 10)))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Early stopping variables
        best_val_loss = float('inf')
        best_model_state = None
        increasing_loss_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                val_acc = (val_outputs.argmax(dim=1) == y_val).float().mean().item()
            
            # Print learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Val Loss: {val_loss.item():.4f} - Val Acc: {val_acc:.4f} - LR: {current_lr:.6f}")
            
            # Early stopping check - stop if validation loss increases more than patience times
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                increasing_loss_counter = 0
            else:
                if val_loss > best_val_loss:  # Only count epochs where loss increases
                    increasing_loss_counter += 1
                    print(f"Validation loss increased. Count: {increasing_loss_counter}/{patience}")
                
                # Stop if loss has increased patience times
                if increasing_loss_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs. Validation loss has increased {patience} times.")
                    break
        
        # Load the best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            train_outputs = model(X_train)
            train_loss = criterion(train_outputs, y_train).item()
            train_acc = (train_outputs.argmax(dim=1) == y_train).float().mean().item()
            
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            val_acc = (val_outputs.argmax(dim=1) == y_val).float().mean().item()
        
        print("\nTraining Summary:")
        print(f"Total epochs: {epoch+1}")
        print(f"Final training loss: {train_loss:.4f}")
        print(f"Final training accuracy: {train_acc:.4f}")
        print(f"Final validation loss: {val_loss:.4f}")
        print(f"Final validation accuracy: {val_acc:.4f}")
        
        if increasing_loss_counter >= patience:
            print("Training stopped early due to validation loss increasing too many times.")
        else:
            print("Training completed all epochs.")
        
        self.save_model(model)
        print(f"Training complete. Saved model to {self.model_path}")
        return model

    def save_model(self, model):
        torch.save(model.state_dict(), self.model_path)
        
    def classify_new_samples(self, model, new_samples_dir='Module1FeatureExtAndTraining/NewSample'):
        """Classify new audio samples using the trained model."""
        print("\nLooking for new Samples in NewSample")
        
        # Check if the directory exists
        if not os.path.exists(new_samples_dir):
            print(f"Directory {new_samples_dir} not found.")
            return
            
        # Find all wav files in the directory
        audio_files = glob.glob(os.path.join(new_samples_dir, '*.wav'))
        if not audio_files:
            print("No wav files found in the NewSample directory.")
            return
            
        print(f"Found {len(audio_files)} audio files to classify.")
        
        # Load the model
        model.eval()
        
        # Process each audio file
        for wav_path in audio_files:
            try:
                # Extract features using the same method as training
                feature_vector = self.extract_features(wav_path)
                if feature_vector is None:
                    continue
                
                # Normalize features
                X = self.scaler.transform(feature_vector.reshape(1, -1))
                X = torch.tensor(X, dtype=torch.float32)
                
                # Get prediction
                with torch.no_grad():
                    outputs = model(X)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0, predicted_class].item()
                
                # Convert numeric class back to speaker ID (folder name)
                speaker_id = self.label_encoder.inverse_transform([predicted_class])[0]
                
                print(f"\nThe Audio file {os.path.basename(wav_path)} belongs to Speaker: {speaker_id}, confidence: {confidence:.2f}")
                
            except Exception as e:
                print(f"Error processing {wav_path}: {e}")

if __name__ == '__main__':
    # Use absolute path to the dataset
    dataset_path = r"C:\Users\ashwa\OneDrive\Documents\Projects\VoiceGeneration Project\Datasets\VC"
    
 
    dataset_changed = True  #will be false when there isd no changein ds since last training
    
    fet = FeatureExtractorAndTrainer(dataset_path)
    
    # Extract features or load from saved files
    features, labels = fet.extract_features_batch(augment=True, force_extract=dataset_changed)
    
    # Train the model
    model = fet.train_model(features, labels)
    
    # Classify new samples
    fet.classify_new_samples(model) 