"""
Utility functions for entity resolution pipeline.

This module provides various utility functions used across the pipeline components.
"""

import os
import logging
import json
import time
import pickle
import psutil
import numpy as np
from pathlib import Path
from prometheus_client import start_http_server, Gauge, Counter, Histogram

logger = logging.getLogger(__name__)

# Prometheus metrics
METRICS = {
    'memory_usage': Gauge('memory_usage_gb', 'Memory usage in GB'),
    'cpu_usage': Gauge('cpu_usage_percent', 'CPU usage in percent'),
    'processed_records': Counter('processed_records_total', 'Total number of records processed'),
    'processing_time': Histogram('processing_time_seconds', 'Time taken to process a batch'),
    'stage_duration': Gauge('stage_duration_seconds', 'Duration of pipeline stage', ['stage']),
    'stage_records': Counter('stage_records_total', 'Records processed in pipeline stage', ['stage']),
    'feature_count': Gauge('feature_count', 'Number of features used'),
    'vector_count': Gauge('vector_count', 'Number of vectors stored'),
    'cluster_count': Gauge('cluster_count', 'Number of entity clusters')
}

class Timer:
    """Simple timer context manager for measuring execution time."""
    
    def __init__(self):
        """Initialize timer."""
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        """Start timer."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timer and calculate duration."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time

def setup_monitoring(config):
    """
    Set up Prometheus monitoring.
    
    Args:
        config (dict): Configuration parameters
    """
    if config['monitoring']['prometheus_enabled']:
        try:
            prometheus_port = config['monitoring']['prometheus_port']
            start_http_server(prometheus_port)
            logger.info("Started Prometheus metrics server on port %d", prometheus_port)
        except Exception as e:
            logger.error("Error starting Prometheus server: %s", str(e))

def update_metrics(metrics_dict=None):
    """
    Update Prometheus metrics.
    
    Args:
        metrics_dict (dict, optional): Custom metrics to update. Defaults to None.
    """
    try:
        # Update system metrics
        METRICS['memory_usage'].set(get_memory_usage())
        METRICS['cpu_usage'].set(psutil.cpu_percent())
        
        # Update custom metrics
        if metrics_dict:
            for key, value in metrics_dict.items():
                if key in METRICS:
                    if isinstance(METRICS[key], Counter):
                        METRICS[key].inc(value)
                    else:
                        METRICS[key].set(value)
    
    except Exception as e:
        logger.error("Error updating metrics: %s", str(e))

def update_stage_metrics(stage_name, metrics_dict):
    """
    Update Prometheus metrics for a pipeline stage.
    
    Args:
        stage_name (str): Name of the pipeline stage
        metrics_dict (dict): Metrics to update
    """
    try:
        # Update standard metrics
        METRICS['memory_usage'].set(get_memory_usage())
        METRICS['cpu_usage'].set(psutil.cpu_percent())
        
        # Update stage-specific metrics
        if 'duration' in metrics_dict:
            METRICS['stage_duration'].labels(stage=stage_name).set(metrics_dict['duration'])
        
        if 'records_processed' in metrics_dict:
            METRICS['stage_records'].labels(stage=stage_name).inc(metrics_dict['records_processed'])
        
        # Update feature metrics
        if 'feature_count' in metrics_dict:
            METRICS['feature_count'].set(metrics_dict['feature_count'])
        
        # Update vector metrics
        if 'strings_embedded' in metrics_dict:
            METRICS['vector_count'].set(metrics_dict['strings_embedded'])
        
        # Update cluster metrics
        if 'clusters' in metrics_dict:
            METRICS['cluster_count'].set(metrics_dict['clusters'])
        
        # Log processing time if batch processing was involved
        if 'batch_durations' in metrics_dict:
            for duration in metrics_dict['batch_durations']:
                METRICS['processing_time'].observe(duration)
    
    except Exception as e:
        logger.error("Error updating stage metrics: %s", str(e))

def get_memory_usage():
    """
    Get current memory usage in GB.
    
    Returns:
        float: Memory usage in GB
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 ** 3)  # Convert bytes to GB

def check_resources(config):
    """
    Check system resources against requirements.
    
    Args:
        config (dict): Configuration parameters
    """
    # Check memory
    total_memory = psutil.virtual_memory().total / (1024 ** 3)  # GB
    memory_limit = config['system']['memory_limit_gb']
    
    if total_memory < memory_limit:
        logger.warning("System has %.2f GB memory, but configuration requires %.2f GB",
                      total_memory, memory_limit)
    
    # Check CPU
    cpu_count = psutil.cpu_count()
    max_workers = config['system']['max_workers']
    
    if cpu_count < max_workers:
        logger.warning("System has %d CPU cores, but configuration uses %d workers",
                      cpu_count, max_workers)
        # Adjust max_workers to match CPU count
        config['system']['max_workers'] = cpu_count
        logger.info("Adjusted max_workers to %d", cpu_count)

def save_checkpoint(state, checkpoint_path):
    """
    Save checkpoint to disk.
    
    Args:
        state (dict): State to save
        checkpoint_path (str or Path): Path to checkpoint file
    """
    try:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert MMapDict instances to regular dictionaries
        serializable_state = {}
        for key, value in state.items():
            if hasattr(value, 'to_dict') and callable(getattr(value, 'to_dict')):
                serializable_state[key] = value.to_dict()
            else:
                serializable_state[key] = value
        
        # Use different serialization based on content type
        if any(isinstance(v, np.ndarray) for v in serializable_state.values()):
            # Use pickle for numpy arrays
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(serializable_state, f)
        else:
            # Use JSON for regular data
            with open(checkpoint_path, 'w') as f:
                json.dump(serializable_state, f)
        
        logger.debug("Saved checkpoint to %s", checkpoint_path)
    
    except Exception as e:
        logger.error("Error saving checkpoint: %s", str(e))

def load_checkpoint(checkpoint_path):
    """
    Load checkpoint from disk.
    
    Args:
        checkpoint_path (str or Path): Path to checkpoint file
        
    Returns:
        dict: Loaded state or empty dict if checkpoint doesn't exist
    """
    try:
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            logger.warning("Checkpoint file not found: %s", checkpoint_path)
            return {}
        
        # Try JSON first
        try:
            with open(checkpoint_path, 'r') as f:
                state = json.load(f)
                return state
        except json.JSONDecodeError:
            # If JSON fails, try pickle
            with open(checkpoint_path, 'rb') as f:
                state = pickle.load(f)
                return state
    
    except Exception as e:
        logger.error("Error loading checkpoint: %s", str(e))
        return {}

def compute_cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1 (list or numpy.ndarray): First vector
        vec2 (list or numpy.ndarray): Second vector
        
    Returns:
        float: Cosine similarity (1.0 = identical, 0.0 = orthogonal)
    """
    # Ensure numpy arrays
    if not isinstance(vec1, np.ndarray):
        vec1 = np.array(vec1)
    
    if not isinstance(vec2, np.ndarray):
        vec2 = np.array(vec2)
    
    # Check for empty vectors
    if vec1.size == 0 or vec2.size == 0:
        return 0.0
    
    # Compute dot product
    dot_product = np.dot(vec1, vec2)
    
    # Compute norms
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # Compute similarity
    if norm1 > 0 and norm2 > 0:
        return dot_product / (norm1 * norm2)
    else:
        return 0.0

def compute_harmonic_mean(x, y):
    """
    Compute harmonic mean of two values.
    
    Args:
        x (float): First value
        y (float): Second value
        
    Returns:
        float: Harmonic mean
    """
    if x > 0 and y > 0:
        return 2 * (x * y) / (x + y)
    else:
        return 0.0

def levenshtein_similarity(str1, str2):
    """
    Compute Levenshtein similarity between two strings.
    
    Args:
        str1 (str): First string
        str2 (str): Second string
        
    Returns:
        float: Similarity (1.0 = identical, 0.0 = completely different)
    """
    try:
        import Levenshtein
        
        # Ensure strings
        if not isinstance(str1, str):
            str1 = str(str1) if str1 is not None else ""
        
        if not isinstance(str2, str):
            str2 = str(str2) if str2 is not None else ""
        
        # Compute Levenshtein distance
        distance = Levenshtein.distance(str1, str2)
        
        # Normalize by maximum length
        max_len = max(len(str1), len(str2))
        
        if max_len > 0:
            return 1.0 - (distance / max_len)
        else:
            return 1.0
    
    except ImportError:
        logger.warning("Levenshtein package not available, using fallback")
        return 0.0 if str1 != str2 else 1.0

def create_output_directories(config):
    """
    Create output directories specified in configuration.
    
    Args:
        config (dict): Configuration parameters
    """
    directories = [
        config['system']['checkpoint_dir'],
        config['system']['output_dir'],
        config['system']['temp_dir']
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug("Created directory: %s", directory)

def validate_config(config):
    """
    Validate configuration parameters.
    
    Args:
        config (dict): Configuration parameters
        
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    required_sections = ['system', 'data', 'embedding', 'weaviate', 'features', 'classification', 'clustering']
    for section in required_sections:
        if section not in config:
            logger.error("Missing required configuration section: %s", section)
            return False
    
    # Validate specific required parameters
    if 'mode' not in config['system']:
        logger.error("Missing required parameter: system.mode")
        return False
    
    # Check system mode
    if config['system']['mode'] not in ['dev', 'prod']:
        logger.error("Invalid system mode: %s. Must be 'dev' or 'prod'", config['system']['mode'])
        return False
    
    # Check embedding model
    if 'model' not in config['embedding']:
        logger.error("Missing required parameter: embedding.model")
        return False
    
    # Check weaviate connection
    if 'host' not in config['weaviate'] or 'port' not in config['weaviate']:
        logger.error("Missing required weaviate connection parameters")
        return False
    
    # Check classification parameters
    if 'algorithm' not in config['classification']:
        logger.error("Missing required parameter: classification.algorithm")
        return False
    
    return True

def generate_uuid5(value):
    """
    Generate UUID5 from a value.
    
    Args:
        value: Value to generate UUID from
        
    Returns:
        str: UUID5 string
    """
    try:
        from uuid import uuid5, NAMESPACE_DNS
        
        # Convert to string if not already
        value_str = str(value)
        
        # Generate UUID
        return str(uuid5(NAMESPACE_DNS, value_str))
    
    except Exception as e:
        logger.error("Error generating UUID: %s", str(e))
        return None
