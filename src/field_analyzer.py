#!/usr/bin/env python3
"""
Weaviate Field Analysis for entity resolution.
"""

import os
import logging
import json
import sys
from pathlib import Path
import yaml
import weaviate
from weaviate.classes.aggregate import GroupByAggregate

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("weaviate_analysis")

def load_config(config_path="config.yml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return None

def analyze_collection_fields(config):
    """
    Analyze field distribution in Weaviate and compare to expected counts.
    
    Args:
        config (dict): Configuration parameters
    """
    try:
        # Connect to Weaviate
        client = weaviate.connect_to_local(
            host=config['weaviate']['host'],
            port=config['weaviate']['port']
        )
        
        collection_name = config['weaviate']['collection_name']
        
        try:
            collection = client.collections.get(collection_name)
            
            # Get total count
            count_result = collection.aggregate.over_all(total_count=True)
            total_count = count_result.total_count
            
            # Get field type distribution
            field_type_result = collection.aggregate.over_all(
                group_by=GroupByAggregate(prop="field_type"),
                total_count=True
            )
            
            weaviate_field_counts = {}
            for group in field_type_result.groups:
                weaviate_field_counts[group.grouped_by.value] = group.total_count
            
            logger.info(f"Weaviate total objects: {total_count}")
            logger.info(f"Weaviate field counts: {weaviate_field_counts}")
            
            # Count directly from record_field_hashes
            output_dir = Path(config['system']['output_dir'])
            record_path = output_dir / "record_field_hashes.json"
            
            direct_counts = {}
            if record_path.exists():
                logger.info("Counting field types directly from record_field_hashes...")
                with open(record_path, 'r') as f:
                    record_data = json.load(f)
                
                # Count by field type
                field_hashes = {}
                for record_id, fields in record_data.items():
                    for field, hash_value in fields.items():
                        if hash_value == 'NULL':
                            continue
                            
                        if field not in field_hashes:
                            field_hashes[field] = set()
                        
                        field_hashes[field].add(hash_value)
                
                # Count unique hashes per field
                for field, hashes in field_hashes.items():
                    direct_counts[field] = len(hashes)
                
                logger.info(f"Direct counts from record_field_hashes: {direct_counts}")
            
            # Compare with expected counts from field_statistics.json
            field_stats_path = output_dir / "field_statistics.json"
            
            if field_stats_path.exists():
                with open(field_stats_path, 'r') as f:
                    field_stats = json.load(f)
                
                expected_counts = {}
                for field, stats in field_stats.items():
                    expected_counts[field] = stats.get('unique_values', 0)
                
                logger.info(f"Expected field counts from field_statistics.json: {expected_counts}")
                
                # Compare counts
                logger.info("\nField Count Comparison:")
                logger.info("Field      | Expected | Direct   | Weaviate | Diff (Direct) | Diff (Weaviate)")
                logger.info("-----------+----------+----------+----------+--------------+----------------")
                
                all_fields = sorted(set(list(weaviate_field_counts.keys()) + list(expected_counts.keys()) + list(direct_counts.keys())))
                
                for field in all_fields:
                    expected = expected_counts.get(field, 0)
                    direct = direct_counts.get(field, 0)
                    actual = weaviate_field_counts.get(field, 0)
                    
                    diff_direct = direct - expected
                    diff_weaviate = actual - expected
                    
                    logger.info(f"{field:10} | {expected:8} | {direct:8} | {actual:8} | {diff_direct:+12} | {diff_weaviate:+14}")
                
                # Find specific examples of missing or problematic hashes
                if direct_counts and any(abs(direct_counts.get(f, 0) - weaviate_field_counts.get(f, 0)) > 0 for f in all_fields):
                    logger.info("\nInvestigating mismatches between direct counts and Weaviate...")
                    
                    # For fields with mismatches, find a sample of hashes
                    for field in all_fields:
                        direct = direct_counts.get(field, 0)
                        actual = weaviate_field_counts.get(field, 0)
                        
                        if direct != actual:
                            logger.info(f"\nField '{field}' has a mismatch: direct={direct}, weaviate={actual}")
                            
                            # Collect hashes used in this field
                            field_hash_set = set()
                            for record_id, fields in record_data.items():
                                if field in fields and fields[field] != 'NULL':
                                    field_hash_set.add(fields[field])
                            
                            # Sample hashes for investigation
                            sample_size = min(5, len(field_hash_set))
                            sample_hashes = list(field_hash_set)[:sample_size]
                            
                            logger.info(f"Sample hashes for field '{field}':")
                            for hash_value in sample_hashes:
                                logger.info(f"  - {hash_value}")
            
        except Exception as e:
            logger.error(f"Error analyzing Weaviate collection: {e}")
        
        # Close client
        client.close()
        
    except Exception as e:
        logger.error(f"Error connecting to Weaviate: {e}")

def main():
    """Main entry point."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yml"
    
    # Load configuration
    config = load_config(config_path)
    if not config:
        return
    
    # Analyze collection fields
    analyze_collection_fields(config)

if __name__ == "__main__":
    main()
