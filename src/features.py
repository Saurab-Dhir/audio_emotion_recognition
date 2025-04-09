import numpy as np
import pandas as pd
import librosa
import logging
import os
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm

from utils import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Feature extraction for audio emotion recognition
    """
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize feature extractor
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config = load_config(config_path)
        self.feature_config = self.config['features']
        
    def extract_mfcc_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract MFCC features
        
        Args:
            y (np.ndarray): Audio time series
            sr (int): Sample rate
            
        Returns:
            np.ndarray: MFCC features
        """
        n_mfcc = self.feature_config['mfcc'].get('n_mfcc', 13)
        include_delta = self.feature_config['mfcc'].get('include_delta', True)
        include_delta_delta = self.feature_config['mfcc'].get('include_delta_delta', True)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        features = [mfccs]
        
        # Extract delta features
        if include_delta:
            mfcc_delta = librosa.feature.delta(mfccs)
            features.append(mfcc_delta)
        
        # Extract delta-delta features
        if include_delta_delta:
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            features.append(mfcc_delta2)
        
        # Concatenate features
        features_concat = np.vstack(features)
        
        return features_concat
    
    def extract_prosodic_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract prosodic features
        
        Args:
            y (np.ndarray): Audio time series
            sr (int): Sample rate
            
        Returns:
            np.ndarray: Prosodic features
        """
        prosodic_features = []
        
        # Extract pitch using harmonic product spectrum
        if self.feature_config['prosodic'].get('extract_pitch', True):
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch = np.zeros_like(magnitudes[0, :])
            
            # Select the pitch with highest magnitude at each frame
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch[t] = pitches[index, t]
            
            # Handle cases where pitch is zero (no pitch detected)
            pitch = np.nan_to_num(pitch)
            
            prosodic_features.append(pitch)
        
        # Extract energy (RMS)
        if self.feature_config['prosodic'].get('extract_energy', True):
            # Compute short-time energy
            energy = librosa.feature.rms(y=y)
            prosodic_features.append(energy[0])
        
        # Extract zero crossing rate
        if self.feature_config['prosodic'].get('extract_zero_crossing_rate', True):
            zcr = librosa.feature.zero_crossing_rate(y=y)
            prosodic_features.append(zcr[0])
        
        # Concatenate features
        if prosodic_features:
            # Ensure all features have the same length
            min_length = min(feature.size for feature in prosodic_features)
            prosodic_features = [feature[:min_length] for feature in prosodic_features]
            
            features_concat = np.vstack(prosodic_features)
        else:
            features_concat = np.array([])
        
        return features_concat
    
    def extract_spectral_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract spectral features
        
        Args:
            y (np.ndarray): Audio time series
            sr (int): Sample rate
            
        Returns:
            np.ndarray: Spectral features
        """
        spectral_features = []
        
        # Extract spectral centroid
        if self.feature_config['spectral'].get('extract_spectral_centroid', True):
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_features.append(centroid[0])
        
        # Extract spectral rolloff
        if self.feature_config['spectral'].get('extract_spectral_rolloff', True):
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_features.append(rolloff[0])
        
        # Extract spectral flux
        if self.feature_config['spectral'].get('extract_spectral_flux', True):
            # Compute STFT
            stft = np.abs(librosa.stft(y))
            
            # Compute flux (difference between consecutive frames)
            flux = np.diff(stft, axis=1)
            flux = np.sum(flux**2, axis=0)
            
            # Append a zero to match original length
            flux = np.concatenate([[0], flux])
            
            spectral_features.append(flux)
        
        # Extract spectral bandwidth
        if self.feature_config['spectral'].get('extract_spectral_bandwidth', True):
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spectral_features.append(bandwidth[0])
        
        # Concatenate features
        if spectral_features:
            # Ensure all features have the same length
            min_length = min(feature.size for feature in spectral_features)
            spectral_features = [feature[:min_length] for feature in spectral_features]