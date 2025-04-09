#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for Audio-Based Emotion Recognition System
"""

import os
import argparse
import logging
from src.utils import load_config, create_directories
from src.data_loader import RAVDESSDataLoader

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

def test_ravdess_loader(config, limit=5):
    """
    Test the RAVDESS data loader with a small subset
    
    Args:
        config (dict): Configuration dictionary
        limit (int): Number of samples to load
    """
    logger.info("Testing RAVDESS data loader...")
    
    # Initialize data loader
    data_loader = RAVDESSDataLoader(config_path="config.yaml")
    
    # Load a small subset of the data
    metadata_df, audio_data = data_loader.load_dataset(limit=limit)
    
    logger.info(f"Successfully loaded {len(metadata_df)} samples")
    logger.info(f"Metadata sample:\n{metadata_df.head()}")
    
    if len(audio_data) > 0:
        y, sr = audio_data[0]
        logger.info(f"Audio sample: Shape={y.shape}, SR={sr}, Duration={len(y)/sr:.2f}s")
    
    return True

def verify_project_structure():
    """
    Verify that the project structure is correctly set up
    """
    expected_dirs = [
        "data/raw",
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
        "src/data_loader.py"
    ]
    
    for file_path in expected_files:
        if not os.path.exists(file_path):
            logger.warning(f"File {file_path} does not exist")
        else:
            logger.info(f"File {file_path} exists")

def main():
    """
    Main function to run the setup verification
    """
    parser = argparse.ArgumentParser(description="Audio-Based Emotion Recognition System")
    parser.add_argument("--check-env", action="store_true", help="Check environment setup")
    parser.add_argument("--test-loader", action="store_true", help="Test RAVDESS data loader")
    parser.add_argument("--verify-structure", action="store_true", help="Verify project structure")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Create necessary directories
    create_directories(config)
    
    # Process arguments
    if args.check_env or not (args.test_loader or args.verify_structure):
        check_environment()
    
    if args.test_loader or not (args.check_env or args.verify_structure):
        test_ravdess_loader(config)
    
    if args.verify_structure or not (args.check_env or args.test_loader):
        verify_project_structure()
    
    logger.info("Setup verification completed successfully")

if __name__ == "__main__":
    main()