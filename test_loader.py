#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to test the CREMA-D data loader
"""

import os
import sys
import pandas as pd

# Add the src directory to the path
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.cremad_loader import CREMADDataLoader

def main():
    """Test the CREMA-D data loader"""
    print("Testing the CREMA-D data loader...")
    
    # Create data loader
    data_loader = CREMADDataLoader(config_path="config.yaml")
    
    # Load a small subset of the data
    metadata_df, audio_data = data_loader.load_dataset(limit=10)
    
    print(f"Loaded {len(metadata_df)} audio samples")
    
    # Print metadata columns
    print(f"\nColumns in metadata_df: {metadata_df.columns.tolist()}")
    
    # Print first few rows of metadata
    print(f"\nFirst 5 rows of metadata_df:")
    print(metadata_df.head())
    
    # Check if 'emotion' column exists
    if 'emotion' in metadata_df.columns:
        print("\nEmotion distribution:")
        emotion_counts = metadata_df['emotion'].value_counts()
        print(emotion_counts)
    else:
        print("\nWARNING: 'emotion' column not found in the DataFrame!")
        
    # Print sample of audio data
    if audio_data:
        print(f"\nAudio data sample shapes:")
        for i in range(min(3, len(audio_data))):
            y, sr = audio_data[i]
            print(f"Sample {i+1}: Shape={y.shape}, SR={sr}, Duration={len(y)/sr:.2f}s")

if __name__ == "__main__":
    main() 