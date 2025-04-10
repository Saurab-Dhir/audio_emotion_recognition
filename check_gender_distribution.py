#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import pandas as pd

# Load features
feature_path = os.path.join('data', 'features', 'cremad_features.pkl')

print(f"Loading features from {feature_path}...")
with open(feature_path, 'rb') as f:
    feature_df = pickle.load(f)

print(f"Loaded feature dataset with shape {feature_df.shape}")

# Check columns
print(f"\nColumns in dataset: {feature_df.columns.tolist()}")

# Check gender distribution if column exists
if 'gender' in feature_df.columns:
    gender_counts = feature_df['gender'].value_counts()
    print(f"\nGender distribution:")
    print(gender_counts)
    print(f"\nPercentage male: {100 * gender_counts.get('male', 0) / len(feature_df):.1f}%")
    print(f"Percentage female: {100 * gender_counts.get('female', 0) / len(feature_df):.1f}%")
else:
    print("\nNo 'gender' column found in dataset!")

# Check unique actor IDs
if 'actor_id' in feature_df.columns:
    actor_counts = feature_df['actor_id'].value_counts()
    print(f"\nNumber of unique actors: {len(actor_counts)}")
    print(f"First few actor IDs: {sorted(actor_counts.index)[:10]}")
else:
    print("\nNo 'actor_id' column found in dataset!") 