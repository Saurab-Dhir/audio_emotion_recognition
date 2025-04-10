#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for Audio-Based Emotion Recognition System
"""

import os
import argparse
import logging
import pickle
from tqdm import tqdm

from src.utils import load_config, create_directories
from src.cremad_loader import CREMADDataLoader
from src.preprocessing import AudioPreprocessor
from src.features import FeatureExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("emotion_recognition.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_environment():
    """
    Check if the environment is correctly set up
    """
    import sys
    import numpy
    import pandas
    import matplotlib
    import librosa
    import sklearn
    
    logger.info(f"Python version: {sys.version}")
    logger.info(f"NumPy version: {numpy.__version__}")
    logger.info(f"Pandas version: {pandas.__version__}")
    logger.info(f"Matplotlib version: {matplotlib.__version__}")
    logger.info(f"Librosa version: {librosa.__version__}")
    logger.info(f"Scikit-learn version: {sklearn.__version__}")

def test_cremad_loader(config, limit=5):
    """
    Test the CREMA-D data loader with a small subset
    
    Args:
        config (dict): Configuration dictionary
        limit (int): Number of samples to load
    """
    logger.info("Testing CREMA-D data loader...")
    
    # Initialize data loader
    data_loader = CREMADDataLoader(config_path="config.yaml")
    
    # Load a small subset of the data
    metadata_df, audio_data = data_loader.load_dataset(limit=limit)
    
    if not metadata_df.empty and len(audio_data) > 0:
        logger.info(f"Successfully loaded {len(metadata_df)} samples")
        logger.info(f"Metadata sample:\n{metadata_df.head()}")
        
        y, sr = audio_data[0]
        logger.info(f"Audio sample: Shape={y.shape}, SR={sr}, Duration={len(y)/sr:.2f}s")
        return True
    else:
        logger.warning("No data loaded. Check your dataset path and file format.")
        return False

def verify_project_structure():
    """
    Verify that the project structure is correctly set up
    """
    expected_dirs = [
        "data/raw/cremad",
        "data/processed",
        "data/features",
        "notebooks",
        "src",
        "models",
        "results"
    ]
    
    for dir_path in expected_dirs:
        if not os.path.exists(dir_path):
            logger.warning(f"Directory {dir_path} does not exist")
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory {dir_path}")
        else:
            logger.info(f"Directory {dir_path} exists")
    
    expected_files = [
        "config.yaml",
        "requirements.txt",
        "src/utils.py",
        "src/cremad_loader.py",
        "src/preprocessing.py",
        "src/features.py"
    ]
    
    for file_path in expected_files:
        if not os.path.exists(file_path):
            logger.warning(f"File {file_path} does not exist")
        else:
            logger.info(f"File {file_path} exists")

def process_dataset(limit=None, output_path=None):
    """
    Process the dataset with the complete pipeline
    
    Args:
        limit (int, optional): Number of samples to process
        output_path (str, optional): Path to save processed features
    """
    logger.info("Starting dataset processing pipeline...")
    
    # Load configuration
    config = load_config()
    
    # Create necessary directories
    create_directories(config)
    
    # Initialize components
    data_loader = CREMADDataLoader()
    preprocessor = AudioPreprocessor()
    feature_extractor = FeatureExtractor()
    
    # Load dataset
    logger.info(f"Loading dataset{' (limited to ' + str(limit) + ' samples)' if limit else ''}...")
    metadata_df, audio_data = data_loader.load_dataset(limit=limit, stratify_by='emotion')
    
    if metadata_df.empty or not audio_data:
        logger.error("No data loaded. Aborting processing.")
        return False
    
    logger.info(f"Loaded {len(metadata_df)} samples")
    
    # Preprocess audio
    logger.info("Preprocessing audio...")
    preprocessed_segments = []
    
    for y, sr in tqdm(audio_data, desc="Preprocessing"):
        segments = preprocessor.preprocess_and_segment(y, sr)
        preprocessed_segments.append(segments)
    
    # Extract features
    logger.info("Extracting features...")
    all_features = feature_extractor.extract_features_batch(
        preprocessed_segments, data_loader.sample_rate, compute_statistics=True
    )
    
    # Create feature dataset
    logger.info("Creating feature dataset...")
    feature_df = feature_extractor.create_feature_dataset(metadata_df, all_features)
    
    # Save features
    if output_path is None:
        output_path = os.path.join(config['paths']['features'], 'cremad_features.pkl')
    
    logger.info(f"Saving features to {output_path}...")
    feature_extractor.save_features(feature_df, output_path)
    
    logger.info(f"Dataset processing complete. {len(feature_df)} feature vectors saved.")
    return True

def main():
    """
    Main function to run the emotion recognition system
    """
    parser = argparse.ArgumentParser(description="Audio-Based Emotion Recognition System")
    parser.add_argument("--check-env", action="store_true", help="Check environment setup")
    parser.add_argument("--test-loader", action="store_true", help="Test CREMA-D data loader")
    parser.add_argument("--verify-structure", action="store_true", help="Verify project structure")
    parser.add_argument("--process", action="store_true", help="Process dataset through entire pipeline")
    parser.add_argument("--limit", type=int, help="Limit number of samples to process")
    parser.add_argument("--output", type=str, help="Output path for processed features")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Create necessary directories
    create_directories(config)
    
    # Process arguments
    if args.check_env:
        check_environment()
    
    if args.test_loader:
        test_cremad_loader(config, limit=args.limit or 5)
    
    if args.verify_structure:
        verify_project_structure()
    
    if args.process:
        process_dataset(limit=args.limit, output_path=args.output)
    
    # If no specific arguments, run basic verification
    if not any([args.check_env, args.test_loader, args.verify_structure, args.process]):
        check_environment()
        test_cremad_loader(config, limit=5)
        verify_project_structure()
    
    logger.info("Execution completed successfully")

if __name__ == "__main__":
    main()