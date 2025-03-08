#!/usr/bin/env python3
"""
Feature Vector Analysis Script

This script loads and analyzes feature vectors from a NumPy file,
providing basic statistics and visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys

def analyze_feature_vectors(file_path, feature_names_path=None):
    """Analyze feature vectors from a NumPy file."""
    # Load feature vectors
    print(f"Loading feature vectors from {file_path}")
    try:
        feature_vectors = np.load(file_path)
    except Exception as e:
        print(f"Error loading feature vectors: {e}")
        return
    
    # Load feature names if available
    feature_names = None
    if feature_names_path and Path(feature_names_path).exists():
        try:
            with open(feature_names_path, 'r') as f:
                feature_names = json.load(f)
            print(f"Loaded {len(feature_names)} feature names")
        except Exception as e:
            print(f"Error loading feature names: {e}")
    
    # Basic information
    print("\n=== Basic Information ===")
    print(f"Shape: {feature_vectors.shape}")
    print(f"Number of vectors: {feature_vectors.shape[0]}")
    print(f"Dimensions per vector: {feature_vectors.shape[1]}")
    print(f"Data type: {feature_vectors.dtype}")
    print(f"Memory usage: {feature_vectors.nbytes / 1024 / 1024:.2f} MB")
    
    # Statistics
    print("\n=== Statistics ===")
    print(f"Mean: {feature_vectors.mean(axis=0)[:5]}...")
    print(f"Min: {feature_vectors.min(axis=0)[:5]}...")
    print(f"Max: {feature_vectors.max(axis=0)[:5]}...")
    print(f"Standard deviation: {feature_vectors.std(axis=0)[:5]}...")
    
    # Sample vectors
    print("\n=== Sample Vectors ===")
    num_samples = min(5, feature_vectors.shape[0])
    for i in range(num_samples):
        if feature_names:
            sample_dict = {name: value for name, value in zip(feature_names, feature_vectors[i])}
            print(f"Vector {i}: {sample_dict}")
        else:
            print(f"Vector {i}: {feature_vectors[i][:5]}...")
    
    # Value distribution
    print("\n=== Value Distribution ===")
    df = pd.DataFrame(feature_vectors)
    print(df.describe().T.head())
    
    # Load labels if available
    try:
        labels_path = Path(file_path).parent / "labels.npy"
        if labels_path.exists():
            labels = np.load(labels_path)
            print(f"\nLoaded {len(labels)} labels")
            print(f"Label distribution: {np.bincount(labels)}")
            
            # Calculate class balance
            if len(np.unique(labels)) == 2:
                pos_count = np.sum(labels == 1)
                neg_count = np.sum(labels == 0)
                pos_pct = pos_count / len(labels) * 100
                print(f"Positive class: {pos_count} ({pos_pct:.2f}%)")
                print(f"Negative class: {neg_count} ({100-pos_pct:.2f}%)")
    except Exception as e:
        print(f"Error loading or analyzing labels: {e}")
    
    # Save visualization
    plt.figure(figsize=(10, 6))
    plt.boxplot(feature_vectors[:, :min(10, feature_vectors.shape[1])])
    if feature_names and len(feature_names) >= min(10, feature_vectors.shape[1]):
        plt.xticks(range(1, min(11, feature_vectors.shape[1])+1), 
                   feature_names[:min(10, feature_vectors.shape[1])], rotation=45)
    plt.title('Feature Distribution')
    plt.tight_layout()
    plt.savefig('feature_distribution.png')
    print("\nSaved feature distribution visualization to 'feature_distribution.png'")

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "output/feature_vectors.npy"
    feature_names_path = sys.argv[2] if len(sys.argv) > 2 else "output/feature_names.json"
    analyze_feature_vectors(file_path, feature_names_path)