#!/usr/bin/env python3
"""
Hash debugging utility for entity resolution pipeline.
"""

import os
import logging
import json
import sys
from pathlib import Path
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("hash_debug")

def load_config(config_path="config.yml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return None

def debug_hash(hash_value, config):
    """
    Debug a specific hash value across all pipeline components.
    
    Args:
        hash_value (str): Hash value to debug
        config (dict): Configuration parameters
    """
    output_dir = Path(config['system']['output_dir'])
    
    results = {
        "hash": hash_value,
        "found_in": []
    }
    
    # Check unique strings
    unique_strings_path = output_dir / "unique_strings_sample.json"
    if unique_strings_path.exists():
        with open(unique_strings_path, 'r') as f:
            unique_strings = json.load(f)
            
        if hash_value in unique_strings:
            results["unique_string_value"] = unique_strings[hash_value]
            results["found_in"].append("unique_strings")
    
    # Check field hash mapping
    field_hash_path = output_dir / "field_hash_mapping_sample.json"
    if field_hash_path.exists():
        with open(field_hash_path, 'r') as f:
            field_hash_mapping = json.load(f)
            
        if hash_value in field_hash_mapping:
            results["field_hash_mapping"] = field_hash_mapping[hash_value]
            results["found_in"].append("field_hash_mapping")
    
    # Check embedded hashes
    embedded_hashes_path = output_dir / "embedded_hashes.json"
    if embedded_hashes_path.exists():
        with open(embedded_hashes_path, 'r') as f:
            embedded_hashes = json.load(f)
            
        if hash_value in embedded_hashes:
            results["found_in"].append("embedded_hashes")
    
    # Check indexed hashes
    indexed_hashes_path = output_dir / "indexed_hashes.json"
    if indexed_hashes_path.exists():
        with open(indexed_hashes_path, 'r') as f:
            indexed_hashes = json.load(f)
            
        if hash_value in indexed_hashes:
            results["found_in"].append("indexed_hashes")
    
    # Find records that use this hash
    record_field_hashes_path = output_dir / "record_field_hashes_sample.json"
    if record_field_hashes_path.exists():
        with open(record_field_hashes_path, 'r') as f:
            record_field_hashes = json.load(f)
            
        records_using_hash = []
        for record_id, field_hashes in record_field_hashes.items():
            for field, field_hash in field_hashes.items():
                if field_hash == hash_value:
                    records_using_hash.append({
                        "record_id": record_id,
                        "field": field
                    })
        
        if records_using_hash:
            results["records_using_hash"] = records_using_hash
            results["found_in"].append("record_field_hashes")
    
    # Print results
    logger.info(f"Debug results for hash: {hash_value}")
    for key, value in results.items():
        if key != "found_in":
            logger.info(f"  {key}: {value}")
    
    logger.info(f"  Found in: {', '.join(results['found_in'])}")
    
    return results

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        logger.error("Usage: python hash_debug.py <hash_value> [config_path]")
        return
    
    hash_value = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else "config.yml"
    
    # Load configuration
    config = load_config(config_path)
    if not config:
        return
    
    # Debug hash
    debug_hash(hash_value, config)

if __name__ == "__main__":
    main()
