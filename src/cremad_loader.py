import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from glob import glob
from tqdm import tqdm

from utils import load_config, load_audio

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_cremad_metadata(filename: str, config: Dict) -> Dict:
    """
    Extract metadata from CREMA-D filename
    Format: ActorID_SentenceID_Emotion_Intensity.wav
    
    Args:
        filename (str): CREMA-D filename
        config (Dict): Configuration dictionary with emotion mappings
        
    Returns:
        Dict: Metadata dictionary
    """
    # Extract the base filename without path and extension
    base_filename = os.path.basename(filename)
    base_filename = os.path.splitext(base_filename)[0]
    
    # Split the filename parts
    parts = base_filename.split('_')
    
    # CREMA-D format: ActorID_SentenceID_Emotion_Intensity.wav
    if len(parts) == 4:
        actor_id, sentence_id, emotion_code, intensity_code = parts
    else:
        logger.warning(f"Unexpected filename format: {filename}")
        return {}
    
    # Map emotion code to emotion name
    emotion_map = config['dataset']['cremad']['emotions']
    intensity_map = config['dataset']['cremad']['intensity']
    
    # Determine gender based on actor ID (from CREMA-D documentation)
    # Female: 1001-1043, Male: 1044-1091
    actor_id_num = int(actor_id)
    if 1001 <= actor_id_num <= 1043:
        gender = "female"
    elif 1044 <= actor_id_num <= 1091:
        gender = "male"
    else:
        gender = "unknown"
    
    metadata = {
        'file_path': filename,
        'actor_id': actor_id,
        'sentence_id': sentence_id,
        'emotion_code': emotion_code,
        'emotion': emotion_map.get(emotion_code, "unknown"),
        'intensity_code': intensity_code,
        'intensity': intensity_map.get(intensity_code, "unknown"),
        'gender': gender
    }
    
    return metadata

class CREMADDataLoader:
    """
    Data loader for CREMA-D dataset
    """
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize data loader
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config = load_config(config_path)
        self.dataset_path = self.config['dataset']['cremad']['path']
        self.sample_rate = self.config['dataset']['cremad']['sample_rate']
        self.emotion_map = self.config['dataset']['cremad']['emotions']
        
    def get_file_paths(self) -> List[str]:
        """
        Get all audio file paths in the dataset
        
        Returns:
            List[str]: List of audio file paths
        """
        file_pattern = os.path.join(self.dataset_path, "*.wav")
        files = glob(file_pattern)
        logger.info(f"Found {len(files)} audio files")
        return files
    
    def create_metadata_dataframe(self) -> pd.DataFrame:
        """
        Create a DataFrame with metadata for all audio files
        
        Returns:
            pd.DataFrame: DataFrame with metadata
        """
        files = self.get_file_paths()
        
        metadata_list = []
        
        for file_path in tqdm(files, desc="Processing metadata"):
            try:
                metadata = extract_cremad_metadata(file_path, self.config)
                if metadata:  # Only add if metadata was successfully extracted
                    metadata_list.append(metadata)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        df = pd.DataFrame(metadata_list)
        
        # Count samples per emotion
        if not df.empty:
            emotion_counts = df['emotion'].value_counts()
            logger.info(f"Samples per emotion:\n{emotion_counts}")
            
            # Count samples per intensity
            intensity_counts = df['intensity'].value_counts()
            logger.info(f"Samples per intensity:\n{intensity_counts}")
            
            # Count samples per gender
            gender_counts = df['gender'].value_counts()
            logger.info(f"Samples per gender:\n{gender_counts}")
        
        return df
    
    def load_audio_data(self, file_paths: List[str], progress_bar: bool = True) -> List[Tuple[np.ndarray, int]]:
        """
        Load audio data for a list of file paths
        
        Args:
            file_paths (List[str]): List of audio file paths
            progress_bar (bool): Whether to show progress bar
            
        Returns:
            List[Tuple[np.ndarray, int]]: List of (audio_data, sample_rate) tuples
        """
        audio_data = []
        
        iterator = tqdm(file_paths, desc="Loading audio") if progress_bar else file_paths
        
        for file_path in iterator:
            try:
                y, sr = load_audio(file_path, self.sample_rate)
                audio_data.append((y, sr))
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        return audio_data
    
    def load_dataset(self, limit: Optional[int] = None, stratify_by: Optional[str] = 'emotion') -> Tuple[pd.DataFrame, List[Tuple[np.ndarray, int]]]:
        """
        Load the dataset with metadata and audio data
        
        Args:
            limit (Optional[int]): Limit the number of files to load
            stratify_by (Optional[str]): Column to stratify sampling by (if limit is not None)
            
        Returns:
            Tuple[pd.DataFrame, List[Tuple[np.ndarray, int]]]: Metadata DataFrame and audio data
        """
        # Create metadata DataFrame
        metadata_df = self.create_metadata_dataframe()
        
        if metadata_df.empty:
            logger.warning("No metadata was extracted, check your dataset path and file format")
            return metadata_df, []
        
        # Optionally limit the number of files with stratified sampling
        if limit is not None and limit < len(metadata_df):
            if stratify_by and stratify_by in metadata_df.columns:
                # Stratified sampling
                grouped = metadata_df.groupby(stratify_by)
                n_per_group = int(np.ceil(limit / len(grouped)))
                
                # Sample from each group
                sampled_dfs = []
                for _, group in grouped:
                    sampled_dfs.append(group.sample(
                        n=min(n_per_group, len(group)), 
                        random_state=42
                    ))
                
                # Combine and take only up to limit
                metadata_df = pd.concat(sampled_dfs).sample(
                    n=min(limit, len(pd.concat(sampled_dfs))), 
                    random_state=42
                )
            else:
                # Random sampling
                metadata_df = metadata_df.sample(n=limit, random_state=42)
        
        # Load audio data
        audio_data = self.load_audio_data(metadata_df['file_path'].tolist())
        
        return metadata_df, audio_data
    
def main():
    """
    Main function to test the data loader
    """
    data_loader = CREMADDataLoader()
    
    # Load a small subset of the data for testing
    metadata_df, audio_data = data_loader.load_dataset(limit=10)
    
    if not metadata_df.empty:
        print(f"Loaded {len(metadata_df)} samples")
        print("\nMetadata sample:")
        print(metadata_df.head())
        
        # Verify audio data
        print(f"\nAudio data sample shapes:")
        for i in range(min(3, len(audio_data))):
            y, sr = audio_data[i]
            print(f"Sample {i+1}: Shape={y.shape}, SR={sr}, Duration={len(y)/sr:.2f}s")
    else:
        print("No data loaded. Check your dataset path and file format.")

if __name__ == "__main__":
    main()