import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from typing import Dict, List, Tuple, Optional, Union
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_directories(config: Dict) -> None:
    """
    Create necessary directories based on configuration
    
    Args:
        config (Dict): Configuration dictionary
    """
    paths = config['paths']
    for path_name, path_value in paths.items():
        os.makedirs(path_value, exist_ok=True)
        logger.info(f"Created directory: {path_value}")

def load_audio(file_path: str, sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample if necessary
    
    Args:
        file_path (str): Path to audio file
        sample_rate (int): Target sample rate
        
    Returns:
        Tuple[np.ndarray, int]: Audio data and sample rate
    """
    try:
        y, sr = librosa.load(file_path, sr=sample_rate)
        return y, sr
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {e}")
        raise

def plot_waveform(y: np.ndarray, sr: int, title: str = "Waveform") -> None:
    """
    Plot audio waveform
    
    Args:
        y (np.ndarray): Audio time series
        sr (int): Sample rate
        title (str): Plot title
    """
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    
def plot_spectrogram(y: np.ndarray, sr: int, title: str = "Spectrogram") -> None:
    """
    Plot spectrogram
    
    Args:
        y (np.ndarray): Audio time series
        sr (int): Sample rate
        title (str): Plot title
    """
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
def plot_mel_spectrogram(y: np.ndarray, sr: int, title: str = "Mel Spectrogram") -> None:
    """
    Plot Mel spectrogram
    
    Args:
        y (np.ndarray): Audio time series
        sr (int): Sample rate
        title (str): Plot title
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
def plot_mfcc(y: np.ndarray, sr: int, n_mfcc: int = 13, title: str = "MFCC") -> None:
    """
    Plot MFCC features
    
    Args:
        y (np.ndarray): Audio time series
        sr (int): Sample rate
        n_mfcc (int): Number of MFCCs to compute
        title (str): Plot title
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()

def extract_ravdess_metadata(filename: str, config: Dict) -> Dict:
    """
    Extract metadata from RAVDESS filename
    Format: modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
    
    Args:
        filename (str): RAVDESS filename
        config (Dict): Configuration dictionary with emotion mappings
        
    Returns:
        Dict: Metadata dictionary
    """
    parts = os.path.basename(filename).split('.')[0].split('-')
    
    emotion_map = config['dataset']['ravdess']['emotions']
    
    metadata = {
        'modality': parts[0],
        'vocal_channel': parts[1],
        'emotion_code': parts[2],
        'emotion': emotion_map.get(parts[2], "unknown"),
        'intensity': parts[3],
        'statement': parts[4],
        'repetition': parts[5],
        'actor': parts[6]
    }
    
    return metadata