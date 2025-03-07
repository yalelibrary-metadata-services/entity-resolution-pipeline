#!/usr/bin/env python3
"""
Systematic fix for field hash mapping in entity resolution pipeline.

This script rebuilds the field hash mapping by scanning all records in 
record_field_hashes, ensuring ALL cases where a hash appears in multiple 
field types are properly captured.
"""

import os
import logging
import json
import time
from pathlib import Path
import yaml
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rebuild_field_mapping")

def load_config(config_path="config.yml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return None

def load_mmap_dict(path):
    """
    Load a memory-mapped dictionary if available.
    
    Args:
        path (str): Path to memory-mapped file
        
    Returns:
        dict: Loaded dictionary
    """
    # Add the src directory to the Python path if needed
    import sys
    current_dir = Path(__file__).parent
    src_dir = str(current_dir / "src")
    if os.path.exists(src_dir) and src_dir not in sys.path:
        sys.path.append(src_dir)
    
    # Try to import MMapDict
    try:
        from src.mmap_dict import MMapDict
        return MMapDict(path)
    except ImportError:
        # If import fails, try to load as a pickle file
        import pickle
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading memory-mapped dictionary: {e}")
            return {}

def load_record_field_hashes(config):
    """
    Load record field hashes with support for all storage formats.
    
    Args:
        config (dict): Configuration parameters
        
    Returns:
        dict: Record field hashes
    """
    output_dir = Path(config['system']['output_dir'])
    mmap_dir = Path(config['system']['temp_dir']) / "mmap"
    
    # First try memory-mapped full dataset
    mmap_path = mmap_dir / "record_field_hashes.mmap"
    if mmap_path.exists():
        logger.info(f"Loading record field hashes from memory-mapped file: {mmap_path}")
        return load_mmap_dict(mmap_path)
    
    # Next try record_index.json which points to the full dataset
    record_index_path = output_dir / "record_index.json"
    if record_index_path.exists():
        try:
            with open(record_index_path, 'r') as f:
                record_index = json.load(f)
            
            location = record_index.get('location')
            if location and location != "in-memory" and os.path.exists(location):
                logger.info(f"Loading record field hashes from indexed location: {location}")
                return load_mmap_dict(location)
        except Exception as e:
            logger.error(f"Error loading from record index: {e}")
    
    # Fall back to non-sample file if it exists
    full_path = output_dir / "record_field_hashes.json"
    if full_path.exists():
        logger.info(f"Loading record field hashes from JSON file: {full_path}")
        with open(full_path, 'r') as f:
            return json.load(f)
    
    # Finally, fall back to sample file with warning
    sample_path = output_dir / "record_field_hashes_sample.json"
    if sample_path.exists():
        logger.warning(f"WARNING: Only found sample record field hashes. This is not the full dataset!")
        with open(sample_path, 'r') as f:
            return json.load(f)
    
    logger.error("No record field hashes found! Check preprocessing output.")
    return {}

def rebuild_field_hash_mapping(record_field_hashes):
    """
    Rebuild field hash mapping from record field hashes.
    
    Args:
        record_field_hashes (dict): Record field hashes
        
    Returns:
        dict: Rebuilt field hash mapping
    """
    # Initialize rebuilt field hash mapping
    rebuilt_field_hash_mapping = {}
    
    # Track hash count by field
    field_counts = {}
    
    # Track hashes with multiple field types
    multi_field_hashes = {}
    
    # Process each record
    for record_id, fields in tqdm(record_field_hashes.items(), desc="Processing records"):
        for field, hash_value in fields.items():
            # Skip null values
            if hash_value == 'NULL':
                continue
            
            # Update field hash mapping
            if hash_value not in rebuilt_field_hash_mapping:
                rebuilt_field_hash_mapping[hash_value] = {}
            
            if field not in rebuilt_field_hash_mapping[hash_value]:
                rebuilt_field_hash_mapping[hash_value][field] = 0
            
            rebuilt_field_hash_mapping[hash_value][field] += 1
            
            # Track field counts
            if field not in field_counts:
                field_counts[field] = set()
            
            field_counts[field].add(hash_value)
            
            # Track multi-field hashes
            if len(rebuilt_field_hash_mapping[hash_value]) > 1:
                multi_field_hashes[hash_value] = rebuilt_field_hash_mapping[hash_value]
    
    # Report statistics
    logger.info(f"Rebuilt field hash mapping with {len(rebuilt_field_hash_mapping)} hashes")
    logger.info(f"Field counts:")
    for field, hashes in field_counts.items():
        logger.info(f"  {field}: {len(hashes)} unique hashes")
    
    logger.info(f"Found {len(multi_field_hashes)} hashes that appear in multiple field types")
    
    # Sample a few multi-field hashes
    if multi_field_hashes:
        logger.info("Sample multi-field hashes:")
        for i, (hash_value, fields) in enumerate(multi_field_hashes.items()):
            if i >= 5:
                break
            logger.info(f"  {hash_value}: {fields}")
    
    return rebuilt_field_hash_mapping

def update_field_hash_files(config, rebuilt_field_hash_mapping):
    """
    Update field hash mapping files with rebuilt mapping.
    
    Args:
        config (dict): Configuration parameters
        rebuilt_field_hash_mapping (dict): Rebuilt field hash mapping
    """
    output_dir = Path(config['system']['output_dir'])
    
    # Update field hash index
    field_hash_path = output_dir / "field_hash_index.json"
    
    with open(field_hash_path, 'w') as f:
        json.dump(rebuilt_field_hash_mapping, f, indent=2)
    
    logger.info(f"Updated field hash index with {len(rebuilt_field_hash_mapping)} entries")
    
    # Update field hash mapping sample
    sample_path = output_dir / "field_hash_mapping_sample.json"
    
    # Create a sample of the first 1000 entries
    sample_size = min(1000, len(rebuilt_field_hash_mapping))
    sample_mapping = {}
    
    for i, (hash_value, fields) in enumerate(rebuilt_field_hash_mapping.items()):
        if i >= sample_size:
            break
        sample_mapping[hash_value] = fields
    
    with open(sample_path, 'w') as f:
        json.dump(sample_mapping, f, indent=2)
    
    logger.info(f"Updated field hash mapping sample with {sample_size} entries")
    
    # Update field statistics
    field_stats = {}
    
    for hash_value, fields in rebuilt_field_hash_mapping.items():
        for field, count in fields.items():
            if field not in field_stats:
                field_stats[field] = {
                    'total_occurrences': 0,
                    'unique_values': 0
                }
            
            field_stats[field]['total_occurrences'] += count
            field_stats[field]['unique_values'] += 1
    
    field_stats_path = output_dir / "field_statistics.json"
    
    with open(field_stats_path, 'w') as f:
        json.dump(field_stats, f, indent=2)
    
    logger.info(f"Updated field statistics")
    logger.info(f"Field statistics:")
    for field, stats in field_stats.items():
        logger.info(f"  {field}: {stats['unique_values']} unique values, {stats['total_occurrences']} total occurrences")

def main():
    """Main entry point."""
    start_time = time.time()
    
    # Parse command-line arguments
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yml"
    
    # Load configuration
    config = load_config(config_path)
    if not config:
        return
    
    # Confirm action
    confirm = input("This will rebuild and replace the field hash mapping files. Continue? (y/n): ")
    if confirm.lower() != 'y':
        logger.info("Operation cancelled")
        return
    
    # Load record field hashes
    record_field_hashes = load_record_field_hashes(config)
    logger.info(f"Loaded {len(record_field_hashes)} records")
    
    # Rebuild field hash mapping
    rebuilt_field_hash_mapping = rebuild_field_hash_mapping(record_field_hashes)
    
    # Update field hash files
    update_field_hash_files(config, rebuilt_field_hash_mapping)
    
    # Measure duration
    duration = time.time() - start_time
    logger.info(f"Field hash mapping rebuild completed in {duration:.2f} seconds")
    
    # Provide next steps
    logger.info("\nNext steps:")
    logger.info("1. Run reset_pipeline.py to reset the pipeline")
    logger.info("2. Run the pipeline from the beginning (python main.py --component all)")

if __name__ == "__main__":
    main()
