# src/models_gpu.py
import torch
import numpy as np
import pandas as pd
import os
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.utils import load_config

logger = logging.getLogger(__name__)

class EmotionDataset(Dataset):
    """PyTorch dataset for emotion recognition features"""
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class EmotionClassifier(nn.Module):
    """Neural network for emotion classification"""
    def __init__(self, input_size, hidden_size, num_classes):
        super(EmotionClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

class GPUModelTrainer:
    """GPU-accelerated model trainer for emotion recognition"""
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() and 
                                   self.config.get('model', {}).get('gpu', {}).get('use_gpu', False) 
                                   else "cpu")
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = None
        logger.info(f"Using device: {self.device}")
    
    def prepare_data(self, feature_df, target_col='emotion', test_size=0.2):
        """Prepare data for GPU training"""
        # Extract features and labels
        X = np.vstack(feature_df['feature_vector'].values)
        y = feature_df[target_col].values
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Create PyTorch datasets
        train_dataset = EmotionDataset(X_train, y_train)
        test_dataset = EmotionDataset(X_test, y_test)
        
        # Create data loaders
        batch_size = self.config.get('model', {}).get('gpu', {}).get('batch_size', 32)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        self.classes_ = self.label_encoder.classes_
        logger.info(f"Prepared data with {X_train.shape[1]} features and {len(self.classes_)} classes")
        
        return train_loader, test_loader, X_test, y_test
    
    def train(self, train_loader, num_epochs=10):
        """Train model on GPU"""
        input_size = next(iter(train_loader))[0].shape[1]
        hidden_size = 128
        num_classes = len(self.classes_)
        
        # Initialize model
        self.model = EmotionClassifier(input_size, hidden_size, num_classes)
        self.model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total
            logger.info(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.2f}%")
        
        logger.info("Training finished")
        return self.model
    
    def evaluate(self, test_loader):
        """Evaluate model on test data"""
        if self.model is None:
            logger.error("Model not trained yet")
            return {}
        
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = correct / total
        
        # Calculate precision, recall, and F1 score
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def save_model(self, path):
        """Save the trained model"""
        if self.model is None:
            logger.error("No model to save")
            return False
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'input_size': next(self.model.parameters()).size(1),
            'hidden_size': 128,  # This should match the hidden size used in training
            'num_classes': len(self.classes_),
            'classes': self.classes_,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }
        
        torch.save(model_data, path)
        logger.info(f"Model saved to {path}")
        return True
    
    def load_model(self, path):
        """Load a trained model"""
        if not os.path.exists(path):
            logger.error(f"Model file {path} does not exist")
            return False
        
        # Set weights_only=False to handle PyTorch 2.6 change
        try:
            # Try the safe globals approach first
            import torch.serialization
            import numpy
            torch.serialization.add_safe_globals(['numpy._core.multiarray._reconstruct'])
            model_data = torch.load(path, map_location=self.device)
        except Exception as e:
            # Fallback to weights_only=False if that fails
            logger.warning(f"First load attempt failed, trying with weights_only=False: {str(e)}")
            model_data = torch.load(path, map_location=self.device, weights_only=False)
        
        # Initialize model
        input_size = model_data['input_size']
        hidden_size = model_data['hidden_size']
        num_classes = model_data['num_classes']
        
        self.model = EmotionClassifier(input_size, hidden_size, num_classes)
        self.model.load_state_dict(model_data['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load other components
        self.classes_ = model_data['classes']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        
        logger.info(f"Model loaded from {path}")
        return True
    
    def predict(self, features):
        """Predict emotions from features"""
        if self.model is None:
            logger.error("Model not loaded or trained")
            return None
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(features_tensor)
            _, predicted = torch.max(outputs, 1)
            
        # Convert to labels
        predicted_labels = self.label_encoder.inverse_transform(predicted.cpu().numpy())
        
        return predicted_labels