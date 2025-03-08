"""
Data management module for entity resolution pipeline.

This module provides a centralized DataManager class for handling serialization,
deserialization, and storage of pipeline data with versioning and validation.
"""

import os
import logging
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any, Optional
import hashlib
import time

logger = logging.getLogger(__name__)

class DataManager:
    """
    Centralized data manager for entity resolution pipeline.
    
    Features:
    - Consistent serialization/deserialization interface
    - Automatic format selection based on data type
    - Metadata tracking for datasets
    - Validation to ensure data consistency
    - Support for both in-memory and memory-mapped storage
    """
    
    def __init__(self, config):
        """
        Initialize the data manager with configuration parameters.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        
        # Set up storage paths
        self.output_dir = Path(config['system']['output_dir'])
        self.checkpoint_dir = Path(config['system']['checkpoint_dir'])
        self.temp_dir = Path(config['system']['temp_dir'])
        
        # Create directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for loaded datasets
        self._cache = {}
        
        # Metadata for dataset versioning and validation
        self.metadata = self._load_metadata()
        
        logger.info("DataManager initialized")
    
    def _load_metadata(self) -> Dict:
        """
        Load metadata about stored datasets.
        
        Returns:
            dict: Metadata dictionary
        """
        metadata_path = self.output_dir / "data_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
        
        # Initialize new metadata if not found
        return {
            'datasets': {},
            'last_updated': time.time(),
            'pipeline_version': self.config.get('system', {}).get('version', '1.0')
        }
    
    def _save_metadata(self):
        """Save metadata about stored datasets."""
        metadata_path = self.output_dir / "data_metadata.json"
        self.metadata['last_updated'] = time.time()
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def _compute_hash(self, data: Any) -> str:
        """
        Compute hash for dataset to track changes.
        
        Args:
            data: Data to hash
        
        Returns:
            str: Hash of the data
        """
        try:
            if isinstance(data, np.ndarray):
                # Hash numpy array
                return hashlib.md5(data.tobytes()).hexdigest()
            elif isinstance(data, pd.DataFrame):
                # Hash DataFrame
                return hashlib.md5(pd.util.hash_pandas_object(data).values.tobytes()).hexdigest()
            elif isinstance(data, (dict, list)):
                # Hash dictionary or list
                return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
            else:
                # Hash other types
                return hashlib.md5(str(data).encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Error computing hash: {e}")
            return f"error-hash-{time.time()}"
    
    def _get_best_format(self, data: Any) -> str:
        """
        Determine the best format for serializing data.
        
        Args:
            data: Data to serialize
        
        Returns:
            str: Format name ('numpy', 'pickle', or 'json')
        """
        if isinstance(data, np.ndarray):
            return 'numpy'
        elif isinstance(data, pd.DataFrame):
            return 'pickle'
        elif isinstance(data, (dict, list)) and all(isinstance(k, str) for k in data.keys()) if isinstance(data, dict) else True:
            return 'json'
        else:
            return 'pickle'
    
    def _get_file_extension(self, format_name: str) -> str:
        """
        Get file extension for a format.
        
        Args:
            format_name (str): Format name
        
        Returns:
            str: File extension
        """
        extensions = {
            'numpy': '.npy',
            'pickle': '.pkl',
            'json': '.json',
            'csv': '.csv'
        }
        return extensions.get(format_name, '.dat')
    
    def save(self, name: str, data: Any, metadata: Dict = None, format_name: str = None,
             use_memory_mapping: bool = None, stage: str = None) -> str:
        """
        Save data with consistent interface.
        
        Args:
            name (str): Name/identifier for the dataset
            data: Data to save
            metadata (dict, optional): Additional metadata about the dataset
            format_name (str, optional): Format to use ('numpy', 'pickle', 'json', or 'csv')
            use_memory_mapping (bool, optional): Whether to use memory mapping
            stage (str, optional): Pipeline stage that generated this data
        
        Returns:
            str: Path to saved file
        """
        # Determine format if not specified
        if format_name is None:
            format_name = self._get_best_format(data)
        
        # Determine memory mapping setting if not specified
        if use_memory_mapping is None:
            use_memory_mapping = self.config['system']['mode'] == 'prod'
        
        # Create file path
        file_extension = self._get_file_extension(format_name)
        file_path = self.output_dir / f"{name}{file_extension}"
        
        # Create memory-mapped path if needed
        mmap_path = None
        if use_memory_mapping:
            mmap_path = self.temp_dir / f"{name}_mmap{file_extension}"
        
        # Compute data hash for versioning
        data_hash = self._compute_hash(data)
        
        # Save data in the appropriate format
        try:
            if format_name == 'numpy':
                np.save(file_path, data)
                if use_memory_mapping and isinstance(data, np.ndarray):
                    # Also save to memory-mapped file
                    np.save(mmap_path, data)
            
            elif format_name == 'pickle':
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
                
                if use_memory_mapping:
                    # Also save to memory-mapped file
                    with open(mmap_path, 'wb') as f:
                        pickle.dump(data, f)
            
            elif format_name == 'json':
                # Convert numpy arrays and other non-serializable objects
                def convert_numpy(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    else:
                        return obj
                
                if isinstance(data, dict):
                    serializable_data = {k: convert_numpy(v) for k, v in data.items()}
                elif isinstance(data, list):
                    serializable_data = [convert_numpy(item) for item in data]
                else:
                    serializable_data = convert_numpy(data)
                
                with open(file_path, 'w') as f:
                    json.dump(serializable_data, f, indent=2)
            
            elif format_name == 'csv':
                if isinstance(data, pd.DataFrame):
                    data.to_csv(file_path, index=False)
                else:
                    logger.warning(f"Data of type {type(data)} cannot be saved as CSV")
                    return None
            
            else:
                logger.warning(f"Unsupported format: {format_name}")
                return None
            
            # Update metadata
            dataset_metadata = {
                'path': str(file_path),
                'format': format_name,
                'hash': data_hash,
                'timestamp': time.time(),
                'size': os.path.getsize(file_path),
                'memory_mapped': use_memory_mapping,
                'mmap_path': str(mmap_path) if mmap_path else None,
                'stage': stage
            }
            
            # Add additional metadata if provided
            if metadata:
                dataset_metadata.update(metadata)
            
            # Update dataset registry
            self.metadata['datasets'][name] = dataset_metadata
            self._save_metadata()
            
            # Update cache
            self._cache[name] = data
            
            logger.info(f"Saved dataset '{name}' to {file_path}")
            return str(file_path)
        
        except Exception as e:
            logger.error(f"Error saving dataset '{name}': {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def load(self, name: str, use_cache: bool = True) -> Any:
        """
        Load data with consistent interface.
        
        Args:
            name (str): Name/identifier for the dataset
            use_cache (bool, optional): Whether to use cached data if available
        
        Returns:
            Data loaded from storage
        """
        # Check cache first
        if use_cache and name in self._cache:
            logger.debug(f"Loaded dataset '{name}' from cache")
            return self._cache[name]
        
        # Check if dataset exists in metadata
        if name not in self.metadata['datasets']:
            logger.warning(f"Dataset '{name}' not found in metadata")
            
            # Try to find the file directly
            for ext in ['.npy', '.pkl', '.json', '.csv']:
                file_path = self.output_dir / f"{name}{ext}"
                if file_path.exists():
                    logger.info(f"Found dataset file {file_path} not in metadata")
                    break
            else:
                logger.error(f"Dataset '{name}' not found")
                return None
        else:
            # Get file path from metadata
            dataset_info = self.metadata['datasets'][name]
            file_path = Path(dataset_info['path'])
            
            # Check if memory-mapped version should be used
            if dataset_info.get('memory_mapped') and dataset_info.get('mmap_path'):
                mmap_path = Path(dataset_info['mmap_path'])
                if mmap_path.exists():
                    file_path = mmap_path
                    logger.debug(f"Using memory-mapped file for '{name}'")
        
        # Load data based on format
        try:
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None
            
            # Determine format from file extension
            format_name = file_path.suffix[1:]  # Remove leading dot
            
            if format_name == 'npy':
                data = np.load(file_path)
            
            elif format_name in ['pkl', 'pickle']:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            
            elif format_name == 'json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
            
            elif format_name == 'csv':
                data = pd.read_csv(file_path)
            
            else:
                logger.warning(f"Unsupported format: {format_name}")
                return None
            
            # Update cache
            self._cache[name] = data
            
            logger.info(f"Loaded dataset '{name}' from {file_path}")
            return data
        
        except Exception as e:
            logger.error(f"Error loading dataset '{name}': {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def exists(self, name: str) -> bool:
        """
        Check if a dataset exists.
        
        Args:
            name (str): Name/identifier for the dataset
        
        Returns:
            bool: True if dataset exists, False otherwise
        """
        # Check metadata first
        if name in self.metadata['datasets']:
            # Verify file exists
            file_path = Path(self.metadata['datasets'][name]['path'])
            return file_path.exists()
        
        # Check for file directly
        for ext in ['.npy', '.pkl', '.json', '.csv']:
            file_path = self.output_dir / f"{name}{ext}"
            if file_path.exists():
                return True
        
        return False
    
    def get_metadata(self, name: str) -> Dict:
        """
        Get metadata for a dataset.
        
        Args:
            name (str): Name/identifier for the dataset
        
        Returns:
            dict: Dataset metadata or None if not found
        """
        return self.metadata['datasets'].get(name)
    
    def list_datasets(self, stage: str = None) -> List[str]:
        """
        List available datasets, optionally filtered by stage.
        
        Args:
            stage (str, optional): Pipeline stage to filter by
        
        Returns:
            list: List of dataset names
        """
        if stage:
            return [name for name, info in self.metadata['datasets'].items() 
                   if info.get('stage') == stage]
        else:
            return list(self.metadata['datasets'].keys())
    
    def delete(self, name: str) -> bool:
        """
        Delete a dataset.
        
        Args:
            name (str): Name/identifier for the dataset
        
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        if name not in self.metadata['datasets']:
            logger.warning(f"Dataset '{name}' not found")
            return False
        
        try:
            # Get file paths
            dataset_info = self.metadata['datasets'][name]
            file_path = Path(dataset_info['path'])
            mmap_path = Path(dataset_info.get('mmap_path', '')) if dataset_info.get('mmap_path') else None
            
            # Delete files
            if file_path.exists():
                file_path.unlink()
            
            if mmap_path and mmap_path.exists():
                mmap_path.unlink()
            
            # Remove from metadata
            del self.metadata['datasets'][name]
            self._save_metadata()
            
            # Remove from cache
            if name in self._cache:
                del self._cache[name]
            
            logger.info(f"Deleted dataset '{name}'")
            return True
        
        except Exception as e:
            logger.error(f"Error deleting dataset '{name}': {e}")
            return False
    
    def save_dataframe(self, name: str, df: pd.DataFrame, stage: str = None) -> str:
        """
        Save pandas DataFrame with appropriate format.
        
        Args:
            name (str): Name/identifier for the DataFrame
            df (pd.DataFrame): DataFrame to save
            stage (str, optional): Pipeline stage that generated this data
        
        Returns:
            str: Path to saved file
        """
        # Save as CSV for easier inspection
        csv_path = self.save(f"{name}_csv", df, format_name='csv', stage=stage)
        
        # Also save as pickle for efficient loading with metadata
        metadata = {
            'columns': list(df.columns),
            'shape': df.shape,
            'csv_path': csv_path
        }
        
        return self.save(name, df, metadata=metadata, format_name='pickle', stage=stage)
    
    def save_numpy_array(self, name: str, array: np.ndarray, metadata: Dict = None, 
                        stage: str = None, use_memory_mapping: bool = None) -> str:
        """
        Save numpy array with appropriate format and metadata.
        
        Args:
            name (str): Name/identifier for the array
            array (np.ndarray): Array to save
            metadata (dict, optional): Additional metadata about the array
            stage (str, optional): Pipeline stage that generated this data
            use_memory_mapping (bool, optional): Whether to use memory mapping
        
        Returns:
            str: Path to saved file
        """
        array_metadata = {
            'shape': array.shape,
            'dtype': str(array.dtype),
            'size_bytes': array.nbytes
        }
        
        # Update with additional metadata if provided
        if metadata:
            array_metadata.update(metadata)
        
        return self.save(name, array, metadata=array_metadata, format_name='numpy', 
                        use_memory_mapping=use_memory_mapping, stage=stage)
    
    def save_feature_data(self, feature_vectors: np.ndarray, labels: np.ndarray, 
                         feature_names: List[str], stage: str = 'features') -> Dict:
        """
        Save feature data with consistency checks.
        
        Args:
            feature_vectors (np.ndarray): Feature vectors
            labels (np.ndarray): Labels
            feature_names (list): Feature names
            stage (str, optional): Pipeline stage that generated this data
        
        Returns:
            dict: Paths to saved files
        """
        # Verify data consistency
        if len(feature_vectors) != len(labels):
            logger.warning(f"Feature vectors ({len(feature_vectors)}) and labels ({len(labels)}) have different lengths")
            
            # Truncate to shorter length
            min_len = min(len(feature_vectors), len(labels))
            feature_vectors = feature_vectors[:min_len]
            labels = labels[:min_len]
            logger.info(f"Truncated to {min_len} samples for consistency")
        
        # Save feature vectors
        feature_metadata = {
            'feature_count': feature_vectors.shape[1],
            'sample_count': feature_vectors.shape[0],
            'has_labels': True,
        }
        
        vectors_path = self.save_numpy_array('feature_vectors', feature_vectors, 
                                            metadata=feature_metadata, stage=stage)
        
        # Save labels
        labels_path = self.save_numpy_array('labels', labels, 
                                          metadata={'sample_count': len(labels)}, stage=stage)
        
        # Save feature names
        names_path = self.save('feature_names', feature_names, stage=stage)
        
        # Create feature index for easier access
        feature_index = {
            'feature_vectors': vectors_path,
            'labels': labels_path,
            'feature_names': names_path,
            'sample_count': len(labels),
            'feature_count': len(feature_names),
            'timestamp': time.time()
        }
        
        # Save index
        index_path = self.save('feature_index', feature_index, stage=stage)
        
        return {
            'feature_vectors': vectors_path,
            'labels': labels_path,
            'feature_names': names_path,
            'feature_index': index_path
        }
    
    def load_feature_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load feature data with consistency checks.
        
        Returns:
            tuple: (feature_vectors, labels, feature_names)
        """
        # Try to load from index first
        feature_index = self.load('feature_index')
        
        # Load components
        feature_vectors = self.load('feature_vectors')
        labels = self.load('labels')
        feature_names = self.load('feature_names')
        
        # Verify data was loaded
        if feature_vectors is None or labels is None or feature_names is None:
            logger.error("Failed to load feature data")
            return None, None, None
        
        # Verify consistency
        if len(feature_vectors) != len(labels):
            logger.warning(f"Feature vectors ({len(feature_vectors)}) and labels ({len(labels)}) have different lengths")
            
            # Truncate to shorter length
            min_len = min(len(feature_vectors), len(labels))
            feature_vectors = feature_vectors[:min_len]
            labels = labels[:min_len]
            logger.info(f"Truncated to {min_len} samples for consistency")
        
        if feature_vectors.shape[1] != len(feature_names):
            logger.warning(f"Feature vector width ({feature_vectors.shape[1]}) doesn't match feature names count ({len(feature_names)})")
            
            # Adjust feature names to match
            if len(feature_names) > feature_vectors.shape[1]:
                logger.info(f"Truncating feature names to match feature vectors")
                feature_names = feature_names[:feature_vectors.shape[1]]
            else:
                logger.info(f"Extending feature names to match feature vectors")
                feature_names.extend([f"feature_{i}" for i in range(len(feature_names), feature_vectors.shape[1])])
        
        logger.info(f"Loaded feature data: {feature_vectors.shape[0]} samples, {feature_vectors.shape[1]} features")
        return feature_vectors, labels, feature_names
    
    def save_classification_results(self, model, weights: np.ndarray, metrics: Dict,
                                   predictions: np.ndarray = None, probabilities: np.ndarray = None,
                                   stage: str = 'classify') -> Dict:
        """
        Save classification results with proper organization.
        
        Args:
            model: Trained classifier model
            weights (np.ndarray): Model weights
            metrics (dict): Classification metrics
            predictions (np.ndarray, optional): Predictions for test set
            probabilities (np.ndarray, optional): Prediction probabilities
            stage (str, optional): Pipeline stage that generated this data
        
        Returns:
            dict: Paths to saved files
        """
        results = {}
        
        # Save model
        results['model'] = self.save('classifier_model', model, stage=stage, format_name='pickle')
        
        # Save weights
        results['weights'] = self.save_numpy_array('model_weights', weights, stage=stage)
        
        # Save metrics
        results['metrics'] = self.save('classification_metrics', metrics, stage=stage)
        
        # Save predictions if available
        if predictions is not None:
            results['predictions'] = self.save_numpy_array('predictions', predictions, stage=stage)
        
        # Save probabilities if available
        if probabilities is not None:
            results['probabilities'] = self.save_numpy_array('probabilities', probabilities, stage=stage)
        
        # Create classification index
        classification_index = {
            'model': results.get('model'),
            'weights': results.get('weights'),
            'metrics': results.get('metrics'),
            'predictions': results.get('predictions'),
            'probabilities': results.get('probabilities'),
            'timestamp': time.time()
        }
        
        # Save index
        results['classification_index'] = self.save('classification_index', classification_index, stage=stage)
        
        logger.info(f"Saved classification results with {len(results)} components")
        return results
    
    def load_classification_results(self) -> Dict:
        """
        Load classification results with proper organization.
        
        Returns:
            dict: Classification results
        """
        # Try to load from index first
        classification_index = self.load('classification_index')
        
        if classification_index is None:
            logger.warning("Classification index not found, trying to load components directly")
            
            # Try to load components directly
            results = {
                'model': self.load('classifier_model'),
                'weights': self.load('model_weights'),
                'metrics': self.load('classification_metrics'),
                'predictions': self.load('predictions'),
                'probabilities': self.load('probabilities')
            }
            
            # Check if essential components were loaded
            if results['model'] is None or results['weights'] is None or results['metrics'] is None:
                logger.error("Failed to load essential classification results")
                return None
            
            return results
        
        # Load components based on index
        results = {}
        for key, path in classification_index.items():
            if key != 'timestamp' and path:
                name = Path(path).stem
                results[key] = self.load(name)
        
        logger.info(f"Loaded classification results with {len(results)} components")
        return results
    
    def save_cluster_data(self, clusters: List[List[str]], metrics: Dict, 
                         entity_to_cluster: Dict = None, stage: str = 'cluster') -> Dict:
        """
        Save clustering results with proper organization.
        
        Args:
            clusters (list): List of clusters, each containing entity IDs
            metrics (dict): Clustering metrics
            entity_to_cluster (dict, optional): Mapping from entity IDs to cluster IDs
            stage (str, optional): Pipeline stage that generated this data
        
        Returns:
            dict: Paths to saved files
        """
        results = {}
        
        # Save clusters
        results['clusters'] = self.save('entity_clusters', clusters, stage=stage)
        
        # Save metrics
        results['metrics'] = self.save('clustering_metrics', metrics, stage=stage)
        
        # Save entity to cluster mapping if available
        if entity_to_cluster:
            results['entity_to_cluster'] = self.save('entity_to_cluster', entity_to_cluster, stage=stage)
        
        # Create clustering index
        clustering_index = {
            'clusters': results.get('clusters'),
            'metrics': results.get('metrics'),
            'entity_to_cluster': results.get('entity_to_cluster'),
            'timestamp': time.time(),
            'cluster_count': len(clusters),
            'total_entities': sum(len(c) for c in clusters)
        }
        
        # Save index
        results['clustering_index'] = self.save('clustering_index', clustering_index, stage=stage)
        
        logger.info(f"Saved clustering results with {len(clusters)} clusters and {clustering_index['total_entities']} entities")
        return results
    
    def load_cluster_data(self) -> Dict:
        """
        Load clustering results with proper organization.
        
        Returns:
            dict: Clustering results
        """
        # Try to load from index first
        clustering_index = self.load('clustering_index')
        
        if clustering_index is None:
            logger.warning("Clustering index not found, trying to load components directly")
            
            # Try to load components directly
            results = {
                'clusters': self.load('entity_clusters'),
                'metrics': self.load('clustering_metrics'),
                'entity_to_cluster': self.load('entity_to_cluster')
            }
            
            # Check if essential components were loaded
            if results['clusters'] is None or results['metrics'] is None:
                logger.error("Failed to load essential clustering results")
                return None
            
            return results
        
        # Load components based on index
        results = {}
        for key, path in clustering_index.items():
            if key not in ['timestamp', 'cluster_count', 'total_entities'] and path:
                name = Path(path).stem
                results[key] = self.load(name)
        
        logger.info(f"Loaded clustering results with {len(results['clusters']) if 'clusters' in results else 0} clusters")
        return results
    
    def clear_cache(self):
        """Clear the in-memory cache."""
        self._cache = {}
        logger.info("Cleared data cache")
