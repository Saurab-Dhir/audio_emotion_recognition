import numpy as np
import pandas as pd
import librosa
import logging
import os
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm

from .utils import load_config

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
            
            features_concat = np.vstack(spectral_features)
        else:
            features_concat = np.array([])
        
        return features_concat
    
    def extract_all_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Extract all features
        
        Args:
            y (np.ndarray): Audio time series
            sr (int): Sample rate
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of features
        """
        features = {}
        
        # Extract MFCC features
        features['mfcc'] = self.extract_mfcc_features(y, sr)
        
        # Extract prosodic features
        features['prosodic'] = self.extract_prosodic_features(y, sr)
        
        # Extract spectral features
        features['spectral'] = self.extract_spectral_features(y, sr)
        
        return features
    
    def compute_feature_statistics(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute statistics of features
        
        Args:
            features (Dict[str, np.ndarray]): Dictionary of features
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of feature statistics
        """
        statistics = {}
        
        for feature_name, feature_data in features.items():
            if feature_data.size == 0:
                continue
                
            # Compute statistics
            feature_mean = np.mean(feature_data, axis=1)
            feature_std = np.std(feature_data, axis=1)
            feature_max = np.max(feature_data, axis=1)
            feature_min = np.min(feature_data, axis=1)
            feature_median = np.median(feature_data, axis=1)
            
            # Create feature statistics array
            feature_stats = np.concatenate([
                feature_mean, 
                feature_std, 
                feature_max, 
                feature_min, 
                feature_median
            ])
            
            statistics[feature_name] = feature_stats
        
        return statistics
    
    def extract_features_from_segment(self, y: np.ndarray, sr: int, compute_statistics: bool = True) -> np.ndarray:
        """
        Extract features from a single audio segment
        
        Args:
            y (np.ndarray): Audio segment
            sr (int): Sample rate
            compute_statistics (bool): Whether to compute feature statistics
            
        Returns:
            np.ndarray: Feature vector
        """
        # Extract all features
        features = self.extract_all_features(y, sr)
        
        if compute_statistics:
            # Compute statistics
            statistics = self.compute_feature_statistics(features)
            
            # Concatenate all statistics into a single feature vector
            feature_vectors = []
            for feature_name in sorted(statistics.keys()):
                feature_vectors.append(statistics[feature_name])
            
            if feature_vectors:
                return np.concatenate(feature_vectors)
            else:
                return np.array([])
        else:
            return features
    
    def extract_features_from_segments(self, segments: List[np.ndarray], sr: int, compute_statistics: bool = True) -> List[np.ndarray]:
        """
        Extract features from multiple audio segments
        
        Args:
            segments (List[np.ndarray]): List of audio segments
            sr (int): Sample rate
            compute_statistics (bool): Whether to compute feature statistics
            
        Returns:
            List[np.ndarray]: List of feature vectors
        """
        features = []
        
        for segment in segments:
            segment_features = self.extract_features_from_segment(segment, sr, compute_statistics)
            features.append(segment_features)
        
        return features
    
    def extract_features_batch(self, audio_segments_list: List[List[np.ndarray]], sr: int, 
                              compute_statistics: bool = True) -> List[List[np.ndarray]]:
        """
        Extract features from multiple audio files with segments
        
        Args:
            audio_segments_list (List[List[np.ndarray]]): List of lists of audio segments
            sr (int): Sample rate
            compute_statistics (bool): Whether to compute feature statistics
            
        Returns:
            List[List[np.ndarray]]: List of lists of feature vectors
        """
        all_features = []
        
        for segments in tqdm(audio_segments_list, desc="Extracting features"):
            features = self.extract_features_from_segments(segments, sr, compute_statistics)
            all_features.append(features)
        
        return all_features
    
    def create_feature_dataset(self, metadata_df: pd.DataFrame, all_features: List[List[np.ndarray]]) -> pd.DataFrame:
        """
        Create a dataset of features and metadata
        
        Args:
            metadata_df (pd.DataFrame): Metadata DataFrame
            all_features (List[List[np.ndarray]]): List of lists of feature vectors
            
        Returns:
            pd.DataFrame: Dataset with features and metadata
        """
        # Create list of dictionaries for each segment
        data = []
        
        for i, (_, metadata) in enumerate(metadata_df.iterrows()):
            if i >= len(all_features):
                logger.warning(f"Metadata index {i} out of range for all_features")
                continue
                
            features_list = all_features[i]
            
            for j, features in enumerate(features_list):
                # Create dictionary of metadata and features
                item = metadata.to_dict()
                item['segment_id'] = j
                
                # Add feature vector as a separate column
                item['feature_vector'] = features
                
                data.append(item)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        return df
    
    def save_features(self, feature_df: pd.DataFrame, output_path: str) -> None:
        """
        Save features to disk
        
        Args:
            feature_df (pd.DataFrame): Feature DataFrame
            output_path (str): Output file path
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save DataFrame using pickle
        with open(output_path, 'wb') as f:
            pickle.dump(feature_df, f)
        
        logger.info(f"Saved features to {output_path}")
    
    def load_features(self, input_path: str) -> pd.DataFrame:
        """
        Load features from disk
        
        Args:
            input_path (str): Input file path
            
        Returns:
            pd.DataFrame: Feature DataFrame
        """
        # Load DataFrame using pickle
        with open(input_path, 'rb') as f:
            feature_df = pickle.load(f)
        
        logger.info(f"Loaded features from {input_path}")
        
        return feature_df

def main():
    """
    Main function for testing the feature extractor
    """
    import matplotlib.pyplot as plt
    from cremad_loader import CREMADDataLoader
    from preprocessing import AudioPreprocessor
    
    # Load a sample audio file
    data_loader = CREMADDataLoader()
    metadata_df, audio_data = data_loader.load_dataset(limit=2)
    
    if not audio_data:
        logger.error("No audio data loaded")
        return
    
    # Preprocess audio
    preprocessor = AudioPreprocessor()
    
    # Process a batch of audio files
    audio_segments_list = []
    
    for y, sr in audio_data:
        segments = preprocessor.preprocess_and_segment(y, sr)
        audio_segments_list.append(segments)
    
    # Create feature extractor
    feature_extractor = FeatureExtractor()
    
    # Extract features from segments
    all_features = feature_extractor.extract_features_batch(audio_segments_list, 
                                                          data_loader.sample_rate)
    
    # Create feature dataset
    feature_df = feature_extractor.create_feature_dataset(metadata_df, all_features)
    
    # Print feature dataset information
    logger.info(f"Feature dataset shape: {feature_df.shape}")
    
    # Print feature vector dimensions
    feature_vector = feature_df['feature_vector'].iloc[0]
    logger.info(f"Feature vector shape: {feature_vector.shape}")
    
    # Save features
    features_path = os.path.join('data', 'features', 'test_features.pkl')
    feature_extractor.save_features(feature_df, features_path)
    
    # Load features
    loaded_df = feature_extractor.load_features(features_path)
    
    # Verify loading
    logger.info(f"Loaded feature dataset shape: {loaded_df.shape}")
    
    # Plot sample features
    plt.figure(figsize=(12, 6))
    plt.plot(feature_vector)
    plt.title("Sample Feature Vector")
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Value")
    plt.tight_layout()
    plt.savefig("feature_vector.png")
    plt.close()
    
    logger.info("Feature extraction test completed")

if __name__ == "__main__":
    main()