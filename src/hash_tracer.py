#!/usr/bin/env python3
"""
Pipeline hash tracer for entity resolution.
"""

import os
import logging
import json
import sys
from pathlib import Path
import yaml
import weaviate
from weaviate.classes.query import Filter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("hash_tracer")

def load_config(config_path="config.yml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return None

def trace_hash(hash_value, config):
    """
    Trace a specific hash value through all pipeline components.
    
    Args:
        hash_value (str): Hash value to trace
        config (dict): Configuration parameters
    """
    output_dir = Path(config['system']['output_dir'])
    mmap_dir = Path(config['system']['temp_dir']) / "mmap"
    
    results = {
        "hash": hash_value,
        "stages": {}
    }
    
    # 1. Check record_field_hashes to find records using this hash
    logger.info("Checking record_field_hashes for hash...")
    record_paths = [
        output_dir / "record_field_hashes.json", 
        output_dir / "record_field_hashes_sample.json"
    ]
    
    for path in record_paths:
        if path.exists():
            with open(path, 'r') as f:
                record_data = json.load(f)
            
            record_info = []
            for record_id, fields in record_data.items():
                for field, field_hash in fields.items():
                    if field_hash == hash_value:
                        record_info.append({
                            "record_id": record_id,
                            "field": field
                        })
            
            results["stages"]["record_field_hashes"] = {
                "found": len(record_info) > 0,
                "records": record_info
            }
            
            if record_info:
                logger.info(f"Found {len(record_info)} records using hash: {record_info}")
                break
    
    # 2. Check field_hash_mapping
    logger.info("Checking field_hash_mapping for hash...")
    field_hash_paths = [
        output_dir / "field_hash_index.json",
        output_dir / "field_hash_mapping_sample.json"
    ]
    
    for path in field_hash_paths:
        if path.exists():
            with open(path, 'r') as f:
                field_data = json.load(f)
            
            if hash_value in field_data:
                results["stages"]["field_hash_mapping"] = {
                    "found": True,
                    "fields": field_data[hash_value]
                }
                logger.info(f"Found hash in field_hash_mapping with fields: {field_data[hash_value]}")
                break
            else:
                results["stages"]["field_hash_mapping"] = {
                    "found": False
                }
                logger.info("Hash NOT found in field_hash_mapping")
    
    # 3. Check unique_strings
    logger.info("Checking unique_strings for hash...")
    unique_string_paths = [
        output_dir / "unique_strings_index.json",
        output_dir / "unique_strings_sample.json"
    ]
    
    for path in unique_string_paths:
        if path.exists():
            with open(path, 'r') as f:
                string_data = json.load(f)
            
            if hash_value in string_data:
                results["stages"]["unique_strings"] = {
                    "found": True,
                    "value": string_data[hash_value]
                }
                logger.info(f"Found hash in unique_strings with value: {string_data[hash_value]}")
                break
            else:
                results["stages"]["unique_strings"] = {
                    "found": False
                }
                logger.info("Hash NOT found in unique_strings")
    
    # 4. Check embedded_hashes
    logger.info("Checking embedded_hashes for hash...")
    embedded_path = output_dir / "embedded_hashes.json"
    if embedded_path.exists():
        with open(embedded_path, 'r') as f:
            embedded_data = json.load(f)
        
        if hash_value in embedded_data:
            results["stages"]["embedded_hashes"] = {
                "found": True
            }
            logger.info("Hash found in embedded_hashes")
        else:
            results["stages"]["embedded_hashes"] = {
                "found": False
            }
            logger.info("Hash NOT found in embedded_hashes")
    
    # 5. Check indexed_hashes
    logger.info("Checking indexed_hashes for hash...")
    indexed_path = output_dir / "indexed_hashes.json"
    if indexed_path.exists():
        with open(indexed_path, 'r') as f:
            indexed_data = json.load(f)
        
        if hash_value in indexed_data:
            results["stages"]["indexed_hashes"] = {
                "found": True
            }
            logger.info("Hash found in indexed_hashes")
        else:
            results["stages"]["indexed_hashes"] = {
                "found": False
            }
            logger.info("Hash NOT found in indexed_hashes")
    
    # 6. Check Weaviate for objects with this hash
    logger.info("Checking Weaviate for objects with this hash...")
    try:
        client = weaviate.connect_to_local(
            host=config['weaviate']['host'],
            port=config['weaviate']['port']
        )
        
        collection_name = config['weaviate']['collection_name']
        
        try:
            collection = client.collections.get(collection_name)
            
            # Create filter for hash
            hash_filter = Filter.by_property("hash").equal(hash_value)
            
            # Execute search
            results_obj = collection.query.fetch_objects(
                filters=hash_filter,
                limit=10,
                include_vector=False
            )
            
            weaviate_objects = []
            for obj in results_obj.objects:
                weaviate_objects.append({
                    "field_type": obj.properties.get("field_type"),
                    "value": obj.properties.get("value"),
                    "frequency": obj.properties.get("frequency")
                })
            
            results["stages"]["weaviate"] = {
                "found": len(weaviate_objects) > 0,
                "objects": weaviate_objects
            }
            
            if weaviate_objects:
                logger.info(f"Found {len(weaviate_objects)} objects in Weaviate: {weaviate_objects}")
            else:
                logger.info("Hash NOT found in Weaviate")
            
        except Exception as e:
            logger.error(f"Error querying Weaviate: {e}")
            results["stages"]["weaviate"] = {
                "found": False,
                "error": str(e)
            }
        
        # Close client
        client.close()
        
    except Exception as e:
        logger.error(f"Error connecting to Weaviate: {e}")
        results["stages"]["weaviate"] = {
            "found": False,
            "error": str(e)
        }
    
    # Summary
    logger.info("\n--- Hash Trace Summary ---")
    for stage, info in results["stages"].items():
        found = info.get("found", False)
        logger.info(f"{stage}: {'FOUND' if found else 'NOT FOUND'}")
    
    return results

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        logger.error("Usage: python hash_tracer.py <hash_value> [config_path]")
        return
    
    hash_value = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else "config.yml"
    
    # Load configuration
    config = load_config(config_path)
    if not config:
        return
    
    # Trace hash
    trace_hash(hash_value, config)

if __name__ == "__main__":
    main()
