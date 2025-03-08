#!/usr/bin/env python3
"""
NPY Data Diagnostics - A tool to explore and understand numpy data files.
"""

import os
import sys
import glob
import numpy as np
import json
from pathlib import Path

def inspect_npy_file(file_path):
    """Show detailed information about a .npy file"""
    try:
        # Load the data
        data = np.load(file_path)
        
        # Get basic info
        print(f"\n==== {os.path.basename(file_path)} ====")
        print(f"Shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        print(f"Size: {data.size} elements")
        print(f"Memory usage: {data.nbytes / (1024*1024):.2f} MB")
        
        # Show data distribution
        if data.size > 0:
            if data.dtype.kind in ['i', 'f', 'u']:  # numeric data
                print(f"Min value: {data.min()}")
                print(f"Max value: {data.max()}")
                print(f"Mean value: {data.mean()}")
                print(f"Unique values: {len(np.unique(data))}")
                
                # Count values (useful for binary classifications)
                if len(np.unique(data)) < 10:
                    values, counts = np.unique(data, return_counts=True)
                    for value, count in zip(values, counts):
                        print(f"  Value {value}: {count} occurrences ({count/data.size:.2%})")
            
            # Show sample of data
            flat_data = data.flatten()
            print("\nSample data (first 10 elements):")
            print(flat_data[:10])
            
            # For 2D data
            if len(data.shape) == 2:
                print(f"\nThis is a 2D array with {data.shape[0]} rows and {data.shape[1]} columns")
                if data.shape[0] > 0 and data.shape[1] > 0:
                    print("First row:")
                    print(data[0])
                    
                    if data.shape[0] > 1:
                        print("\nSecond row:")
                        print(data[1])
            
            # For 1D data with typical sizes
            if len(data.shape) == 1:
                print(f"\nThis is a 1D array with {data.shape[0]} elements")
                if data.size in [23260, 54270, 77530]:  # These appear in your errors
                    print(f"NOTE: Size {data.size} matches a common size in your errors")
        
        return data
    except Exception as e:
        print(f"Error inspecting {file_path}: {e}")
        return None

def find_npy_files(directory):
    """Find all .npy files in a directory"""
    return sorted(glob.glob(os.path.join(directory, "*.npy")))

def inspect_all_files(directory):
    """Inspect all .npy files in a directory"""
    npy_files = find_npy_files(directory)
    print(f"Found {len(npy_files)} .npy files in {directory}")
    
    # Load and store all data for comparison
    loaded_data = {}
    
    for file_path in npy_files:
        data = inspect_npy_file(file_path)
        if data is not None:
            loaded_data[os.path.basename(file_path)] = data
    
    return loaded_data

def analyze_relationships(loaded_data):
    """Analyze relationships between data files"""
    print("\n==== Potential Relationships ====")
    
    # Check for matching dimensions
    files_by_length = {}
    for name, data in loaded_data.items():
        if len(data.shape) == 1:  # 1D array
            length = data.shape[0]
            if length not in files_by_length:
                files_by_length[length] = []
            files_by_length[length].append(name)
        elif len(data.shape) == 2:  # 2D array
            length = data.shape[0]  # First dimension
            if length not in files_by_length:
                files_by_length[length] = []
            files_by_length[length].append(f"{name} (with {data.shape[1]} columns)")
    
    print("Files with matching first dimensions:")
    for length, files in sorted(files_by_length.items()):
        if len(files) > 1:  # If multiple files share this length
            print(f"  Length {length}: {', '.join(files)}")
    
    # Check for common split points (train/test splits)
    for name, data in loaded_data.items():
        if len(data.shape) == 1 and data.dtype.kind in ['i', 'f', 'u']:
            # Check common split points (e.g., 70/30 split)
            for split_ratio in [0.7, 0.8]:
                split_point = int(data.shape[0] * split_ratio)
                print(f"Potential {split_ratio*100:.0f}/{(1-split_ratio)*100:.0f} split point for {name}: {split_point}")

def check_json_files(directory):
    """Check JSON files that might provide context"""
    json_files = glob.glob(os.path.join(directory, "*.json"))
    print(f"\nFound {len(json_files)} JSON files in {directory}")
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            print(f"\n==== {os.path.basename(file_path)} ====")
            if isinstance(data, list):
                print(f"Contains a list with {len(data)} elements")
                if len(data) > 0:
                    print(f"First element type: {type(data[0])}")
                    print(f"Sample: {data[0]}")
            elif isinstance(data, dict):
                print(f"Contains a dictionary with {len(data)} keys")
                print(f"Keys: {', '.join(list(data.keys())[:10])}{'...' if len(data) > 10 else ''}")
                
                # Special handling for feature names
                if 'feature_names.json' in file_path or any(key.startswith('feature') for key in data.keys()):
                    print("This appears to be feature-related data")
                    if isinstance(data, list):
                        print(f"Feature names: {', '.join(data[:10])}{'...' if len(data) > 10 else ''}")
                
                # Special handling for metrics
                if 'metrics' in file_path or any(key in ['precision', 'recall', 'accuracy'] for key in data.keys()):
                    print("This appears to be classification metrics data:")
                    for key, value in data.items():
                        print(f"  {key}: {value}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

def interactive_exploration(directory):
    """Interactive exploration of .npy files"""
    loaded_data = inspect_all_files(directory)
    
    # Check JSON files that might provide context
    check_json_files(directory)
    
    # Analyze potential relationships
    analyze_relationships(loaded_data)
    
    print("\n==== Completed data exploration ====")

if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else "./output"
    interactive_exploration(directory)