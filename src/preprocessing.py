import numpy as np
import librosa
import logging
from typing import Dict, List, Tuple, Optional, Union, Callable
from scipy import signal
from tqdm import tqdm

from .utils import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def normalize_audio(y: np.ndarray) -> np.ndarray:
    """
    Normalize audio to [-1, 1] range
    
    Args:
        y (np.ndarray): Audio time series
        
    Returns:
        np.ndarray: Normalized audio
    """
    # Avoid division by zero
    if np.max(np.abs(y)) > 0:
        return y / np.max(np.abs(y))
    return y

def remove_silence(y: np.ndarray, sr: int, threshold_db: float = -50.0, min_duration_ms: int = 100) -> np.ndarray:
    """
    Remove silence from audio
    
    Args:
        y (np.ndarray): Audio time series
        sr (int): Sample rate
        threshold_db (float): Threshold in dB below reference to consider as silence
        min_duration_ms (int): Minimum duration in milliseconds to keep segment
        
    Returns:
        np.ndarray: Audio with silence removed
    """
    # Convert to dB
    y_db = librosa.amplitude_to_db(np.abs(y), ref=np.max)
    
    # Find non-silent segments
    non_silent = (y_db > threshold_db)
    
    # Minimum number of samples for a segment to be kept
    min_samples = int(sr * min_duration_ms / 1000)
    
    # Find runs of True (non-silent segments)
    edges = np.diff(np.concatenate([[0], non_silent, [0]]))
    starts = np.where(edges > 0)[0]
    ends = np.where(edges < 0)[0]
    
    # Filter segments shorter than min_duration_ms
    valid_segments = (ends - starts >= min_samples)
    
    # Extract valid segments
    y_out = np.zeros_like(y)
    ptr = 0
    
    for start, end, valid in zip(starts, ends, valid_segments):
        if valid:
            segment_length = end - start
            y_out[ptr:ptr+segment_length] = y[start:end]
            ptr += segment_length
    
    # Trim to actual length
    return y_out[:ptr]

def apply_bandpass_filter(y: np.ndarray, sr: int, low_cutoff: float = 80.0, high_cutoff: float = 7500.0, order: int = 5) -> np.ndarray:
    """
    Apply bandpass filter to audio
    
    Args:
        y (np.ndarray): Audio time series
        sr (int): Sample rate
        low_cutoff (float): Low cutoff frequency in Hz
        high_cutoff (float): High cutoff frequency in Hz
        order (int): Filter order
        
    Returns:
        np.ndarray: Filtered audio
    """
    nyquist = 0.5 * sr
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    
    # Design filter
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply filter
    y_filtered = signal.filtfilt(b, a, y)
    
    return y_filtered

def reduce_noise_spectral_gating(y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512, 
                               n_std_thresh: float = 1.5) -> np.ndarray:
    """
    Reduce noise using spectral gating
    
    Args:
        y (np.ndarray): Audio time series
        sr (int): Sample rate
        n_fft (int): FFT window size
        hop_length (int): Hop length for STFT
        n_std_thresh (float): Number of standard deviations for threshold
        
    Returns:
        np.ndarray: Noise-reduced audio
    """
    # Compute STFT (Short-Time Fourier Transform)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    
    # Compute magnitude
    mag = np.abs(D)
    
    # Compute noise threshold (assuming first 500ms is noise)
    noise_frames = int(0.5 * sr / hop_length)
    noise_mag = mag[:, :noise_frames] if noise_frames > 0 else mag[:, :1]
    
    # Compute mean and std of noise
    noise_mean = np.mean(noise_mag, axis=1, keepdims=True)
    noise_std = np.std(noise_mag, axis=1, keepdims=True)
    
    # Compute threshold
    threshold = noise_mean + n_std_thresh * noise_std
    
    # Apply threshold (mask)
    mask = (mag > threshold)
    
    # Apply mask to STFT
    D_denoised = D * mask
    
    # Inverse STFT
    y_denoised = librosa.istft(D_denoised, hop_length=hop_length)
    
    # Make sure output has the same length as input
    y_denoised = librosa.util.fix_length(y_denoised, size=len(y))
    
    return y_denoised

def segment_audio(y: np.ndarray, sr: int, segment_length_ms: int = 3000, hop_length_ms: int = 1000) -> List[np.ndarray]:
    """
    Segment audio into fixed-length segments
    
    Args:
        y (np.ndarray): Audio time series
        sr (int): Sample rate
        segment_length_ms (int): Segment length in milliseconds
        hop_length_ms (int): Hop length in milliseconds
        
    Returns:
        List[np.ndarray]: List of audio segments
    """
    # Convert to samples
    segment_length = int(sr * segment_length_ms / 1000)
    hop_length = int(sr * hop_length_ms / 1000)
    
    # Check if audio is too short
    if len(y) < segment_length:
        # Pad with zeros if audio is shorter than segment length
        y_padded = np.zeros(segment_length)
        y_padded[:len(y)] = y
        return [y_padded]
    
    # Segment audio
    segments = []
    for i in range(0, len(y) - segment_length + 1, hop_length):
        segments.append(y[i:i+segment_length])
    
    # Add last segment if needed
    if len(y) - segment_length + 1 % hop_length != 0:
        segments.append(y[-segment_length:])
    
    return segments

class AudioPreprocessor:
    """
    Audio preprocessing pipeline
    """
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize preprocessor
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config = load_config(config_path)
        self.preprocessing_config = self.config['preprocessing']
        
    def preprocess_audio(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply preprocessing pipeline to audio
        
        Args:
            y (np.ndarray): Audio time series
            sr (int): Sample rate
            
        Returns:
            np.ndarray: Preprocessed audio
        """
        # Apply preprocessing steps according to configuration
        # 1. Normalize
        if self.preprocessing_config['normalize']:
            y = normalize_audio(y)
        
        # 2. Apply noise reduction
        if self.preprocessing_config['noise_reduction']:
            y = reduce_noise_spectral_gating(y, sr)
        
        # 3. Remove silence
        if self.preprocessing_config.get('remove_silence', False):
            y = remove_silence(y, sr)
        
        # 4. Apply bandpass filter
        if self.preprocessing_config.get('apply_bandpass', False):
            low_cutoff = self.preprocessing_config.get('bandpass_low_cutoff', 80.0)
            high_cutoff = self.preprocessing_config.get('bandpass_high_cutoff', 7500.0)
            y = apply_bandpass_filter(y, sr, low_cutoff, high_cutoff)
        
        return y
    
    def preprocess_and_segment(self, y: np.ndarray, sr: int) -> List[np.ndarray]:
        """
        Preprocess audio and segment it
        
        Args:
            y (np.ndarray): Audio time series
            sr (int): Sample rate
            
        Returns:
            List[np.ndarray]: List of preprocessed audio segments
        """
        # Apply preprocessing
        y_preprocessed = self.preprocess_audio(y, sr)
        
        # Segment audio
        segment_length_ms = self.preprocessing_config.get('segment_length_ms', 3000)
        hop_length_ms = self.preprocessing_config.get('hop_length_ms', 1000)
        
        segments = segment_audio(y_preprocessed, sr, segment_length_ms, hop_length_ms)
        
        return segments
    
    def batch_preprocess(self, audio_data: List[Tuple[np.ndarray, int]], segment: bool = True) -> List[Union[np.ndarray, List[np.ndarray]]]:
        """
        Preprocess a batch of audio files
        
        Args:
            audio_data (List[Tuple[np.ndarray, int]]): List of (audio_data, sample_rate) tuples
            segment (bool): Whether to segment the audio
            
        Returns:
            List[Union[np.ndarray, List[np.ndarray]]]: List of preprocessed audio or segments
        """
        results = []
        
        for y, sr in tqdm(audio_data, desc="Preprocessing audio"):
            if segment:
                processed = self.preprocess_and_segment(y, sr)
            else:
                processed = self.preprocess_audio(y, sr)
            
            results.append(processed)
        
        return results

def main():
    """
    Main function for testing the preprocessor
    """
    import matplotlib.pyplot as plt
    from cremad_loader import CREMADDataLoader
    
    # Load a sample audio file
    data_loader = CREMADDataLoader()
    metadata_df, audio_data = data_loader.load_dataset(limit=1)
    
    if not audio_data:
        logger.error("No audio data loaded")
        return
    
    y, sr = audio_data[0]
    
    # Create preprocessor
    preprocessor = AudioPreprocessor()
    
    # Preprocess audio
    y_preprocessed = preprocessor.preprocess_audio(y, sr)
    
    # Segment audio
    segments = preprocessor.preprocess_and_segment(y, sr)
    
    # Plot original and preprocessed audio
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title("Original Audio")
    
    plt.subplot(3, 1, 2)
    librosa.display.waveshow(y_preprocessed, sr=sr)
    plt.title("Preprocessed Audio")
    
    plt.subplot(3, 1, 3)
    if segments:
        librosa.display.waveshow(segments[0], sr=sr)
        plt.title(f"First Segment (out of {len(segments)} segments)")
    
    plt.tight_layout()
    plt.savefig("preprocessing_example.png")
    plt.close()
    
    logger.info(f"Original audio shape: {y.shape}")
    logger.info(f"Preprocessed audio shape: {y_preprocessed.shape}")
    logger.info(f"Number of segments: {len(segments)}")
    if segments:
        logger.info(f"First segment shape: {segments[0].shape}")
    
    logger.info("Preprocessing test completed")

if __name__ == "__main__":
    main()