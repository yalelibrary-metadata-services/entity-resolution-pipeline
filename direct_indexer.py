#!/usr/bin/env python3
"""
Complete indexer for entity resolution pipeline.

This script properly handles the full dataset, including memory-mapped files,
to ensure ALL data is indexed correctly in Weaviate.
"""

import os
import logging
import json
import sys
import time
from pathlib import Path
import yaml
import weaviate
import numpy as np
import uuid
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fixed_indexer")

def load_config(config_path="config.yml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return None

def generate_uuid5(value):
    """Generate UUID5 from a value."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(value)))

class MMapDictWrapper:
    """Wrapper for MMapDict to handle missing imports."""
    
    def __init__(self, path):
        """Initialize the wrapper."""
        # Add the src directory to the Python path if needed
        src_dir = str(Path(__file__).parent.parent / "src")
        if src_dir not in sys.path:
            sys.path.append(src_dir)
        
        # Import MMapDict from src.mmap_dict
        try:
            from src.mmap_dict import MMapDict
            self.mmap_dict = MMapDict(path)
        except ImportError:
            # If import fails, create a fallback implementation
            logger.warning(f"Could not import MMapDict, using fallback implementation")
            self.mmap_dict = self._fallback_load(path)
    
    def _fallback_load(self, path):
        """Fallback implementation to load memory-mapped dictionary."""
        import pickle
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading memory-mapped dictionary: {e}")
            return {}
    
    def __getitem__(self, key):
        """Get item from dictionary."""
        return self.mmap_dict[key]
    
    def __contains__(self, key):
        """Check if key exists in dictionary."""
        return key in self.mmap_dict
    
    def keys(self):
        """Get dictionary keys."""
        return self.mmap_dict.keys()
    
    def values(self):
        """Get dictionary values."""
        return self.mmap_dict.values()
    
    def items(self):
        """Get dictionary items."""
        return self.mmap_dict.items()
    
    def get(self, key, default=None):
        """Get item with default."""
        return self.mmap_dict.get(key, default)

def load_full_record_data(config):
    """
    Load the FULL record data, handling both memory-mapped and JSON formats.
    
    Args:
        config (dict): Configuration parameters
        
    Returns:
        dict: Complete record data
    """
    output_dir = Path(config['system']['output_dir'])
    mmap_dir = Path(config['system']['temp_dir']) / "mmap"
    
    # First try memory-mapped full dataset
    mmap_path = mmap_dir / "record_field_hashes.mmap"
    if mmap_path.exists():
        logger.info(f"Loading FULL dataset from memory-mapped file: {mmap_path}")
        return MMapDictWrapper(mmap_path)
    
    # Next try record_index.json which points to the full dataset
    record_index_path = output_dir / "record_index.json"
    if record_index_path.exists():
        with open(record_index_path, 'r') as f:
            record_index = json.load(f)
        
        location = record_index.get('location')
        if location and location != "in-memory" and os.path.exists(location):
            logger.info(f"Loading FULL dataset from indexed location: {location}")
            return MMapDictWrapper(location)
    
    # Fall back to non-sample file if it exists
    full_path = output_dir / "record_field_hashes.json"
    if full_path.exists():
        logger.info(f"Loading FULL dataset from JSON file: {full_path}")
        with open(full_path, 'r') as f:
            return json.load(f)
    
    # Finally, fall back to sample file with warning
    sample_path = output_dir / "record_field_hashes_sample.json"
    if sample_path.exists():
        logger.warning(f"WARNING: Only found SAMPLE data file. This is not the full dataset!")
        with open(sample_path, 'r') as f:
            return json.load(f)
    
    logger.error("No record data found! Check preprocessing output.")
    return {}

def load_full_unique_strings(config):
    """
    Load the FULL unique strings data, handling both memory-mapped and JSON formats.
    
    Args:
        config (dict): Configuration parameters
        
    Returns:
        dict: Complete unique strings data
    """
    output_dir = Path(config['system']['output_dir'])
    mmap_dir = Path(config['system']['temp_dir']) / "mmap"
    
    # First try memory-mapped full dataset
    mmap_path = mmap_dir / "unique_strings.mmap"
    if mmap_path.exists():
        logger.info(f"Loading FULL unique strings from memory-mapped file: {mmap_path}")
        return MMapDictWrapper(mmap_path)
    
    # Next try unique_strings_index.json which points to the full dataset
    index_path = output_dir / "unique_strings_index.json"
    if index_path.exists():
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        location = index.get('location')
        if location and location != "in-memory" and os.path.exists(location):
            logger.info(f"Loading FULL unique strings from indexed location: {location}")
            return MMapDictWrapper(location)
    
    # Fall back to full file if it exists
    full_path = output_dir / "unique_strings.json"
    if full_path.exists():
        logger.info(f"Loading FULL unique strings from JSON file: {full_path}")
        with open(full_path, 'r') as f:
            return json.load(f)
    
    # Finally, fall back to sample file with warning
    sample_path = output_dir / "unique_strings_sample.json"
    if sample_path.exists():
        logger.warning(f"WARNING: Only found SAMPLE unique strings. This is not the full dataset!")
        with open(sample_path, 'r') as f:
            return json.load(f)
    
    logger.error("No unique strings found! Check preprocessing output.")
    return {}

def load_full_embeddings(config):
    """
    Load the FULL embeddings data, handling both memory-mapped and JSON formats.
    
    Args:
        config (dict): Configuration parameters
        
    Returns:
        dict: Complete embeddings data
    """
    output_dir = Path(config['system']['output_dir'])
    mmap_dir = Path(config['system']['temp_dir']) / "mmap"
    
    # Try to load from embedding index first (enhanced preprocessor)
    embedding_index_path = output_dir / "embedding_index.json"
    if embedding_index_path.exists():
        with open(embedding_index_path, 'r') as f:
            embedding_index = json.load(f)
        
        location = embedding_index.get('location')
        
        if location != "in-memory" and os.path.exists(location):
            logger.info(f"Loading FULL embeddings from indexed location: {location}")
            return MMapDictWrapper(location)
    
    # Try to load from embedding checkpoint
    checkpoint_dir = Path(config['system']['checkpoint_dir'])
    embedding_checkpoint = checkpoint_dir / "embedding_final.ckpt"
    
    if embedding_checkpoint.exists():
        logger.info(f"Loading embeddings from checkpoint: {embedding_checkpoint}")
        try:
            import pickle
            with open(embedding_checkpoint, 'rb') as f:
                state = pickle.load(f)
                
                if 'embeddings' in state:
                    logger.info("Found embeddings in checkpoint state")
                    return state['embeddings']
                elif 'embeddings_location' in state and os.path.exists(state['embeddings_location']):
                    logger.info(f"Loading embeddings from location in checkpoint: {state['embeddings_location']}")
                    return MMapDictWrapper(state['embeddings_location'])
        except Exception as e:
            logger.error(f"Error loading embeddings from checkpoint: {e}")
    
    # Fall back to embeddings_sample.json if it exists
    embeddings_sample_path = output_dir / "embeddings_sample.json"
    if embeddings_sample_path.exists():
        logger.warning("WARNING: Only found SAMPLE embeddings. This is not the full dataset!")
        with open(embeddings_sample_path, 'r') as f:
            return json.load(f)
    
    logger.error("No embeddings found! Check embedding output.")
    return {}

def fixed_index(config):
    """
    Completely index ALL objects in Weaviate based on full record data.
    
    Args:
        config (dict): Configuration parameters
    """
    # Load FULL record data
    logger.info("Loading FULL record data...")
    record_data = load_full_record_data(config)
    #logger.info(f"Loaded {len(record_data)} records")
    
    # Load FULL unique strings
    logger.info("Loading FULL unique strings...")
    unique_strings = load_full_unique_strings(config)
    logger.info(f"Loaded unique strings data")
    
    # Load FULL embeddings
    logger.info("Loading FULL embeddings...")
    embeddings = load_full_embeddings(config)
    logger.info(f"Loaded embeddings data")
    
    # Collect all unique field types and hashes
    logger.info("Building field hash map...")
    field_hash_map = {}
    
    for record_id, fields in tqdm(record_data.items(), desc="Processing records"):
        for field, hash_value in fields.items():
            if hash_value == 'NULL':
                continue
                
            if hash_value not in field_hash_map:
                field_hash_map[hash_value] = {}
            
            if field not in field_hash_map[hash_value]:
                field_hash_map[hash_value][field] = 0
            
            field_hash_map[hash_value][field] += 1
    
    logger.info(f"Found {len(field_hash_map)} unique hashes across all records")
    
    # Log field statistics by counting unique hashes per field
    field_stats = {}
    for hash_value, fields in field_hash_map.items():
        for field, count in fields.items():
            if field not in field_stats:
                field_stats[field] = set()
            field_stats[field].add(hash_value)
    
    logger.info("Field statistics (unique values per field):")
    for field, hashes in field_stats.items():
        logger.info(f"  {field}: {len(hashes)} unique values")
    
    # Connect to Weaviate
    logger.info("Connecting to Weaviate...")
    client = weaviate.connect_to_local(
        host=config['weaviate']['host'],
        port=config['weaviate']['port']
    )
    
    # Check if collection exists and create if needed
    collection_name = config['weaviate']['collection_name']
    collections = client.collections.list_all()
    
    if collection_name in collections:
        logger.info(f"Collection {collection_name} already exists. Deleting...")
        client.collections.delete(collection_name)
    
    # Create collection with optimized schema
    logger.info(f"Creating collection: {collection_name}")
    
    # Import required enums and config classes
    from weaviate.classes.config import Configure, Property, DataType, VectorDistances
    
    # Configure named vectors for each field type
    field_types = ["composite", "person", "title", "provision", "subjects"]
    
    vector_configs = []
    for field_type in field_types:
        vector_configs.append(
            Configure.NamedVectors.none(
                name=field_type,
                vector_index_config=Configure.VectorIndex.hnsw(
                    ef=config['weaviate']['ef'],
                    max_connections=config['weaviate']['max_connections'],
                    ef_construction=config['weaviate']['ef_construction'],
                    distance_metric=VectorDistances.COSINE,
                )
            )
        )
    
    # Define properties
    properties = [
        Property(name="hash", data_type=DataType.TEXT, index_filterable=True, 
                 description="Hash of the text value"),
        Property(name="value", data_type=DataType.TEXT, index_filterable=True, 
                 description="Original text value"),
        Property(name="field_type", data_type=DataType.TEXT, index_filterable=True, 
                 description="Type of field (composite, person, title, etc.)"),
        Property(name="frequency", data_type=DataType.NUMBER, index_filterable=True, 
                 description="Frequency of occurrence in the dataset")
    ]
    
    # Create collection
    collection = client.collections.create(
        name=collection_name,
        description="Entity resolution vectors collection",
        vectorizer_config=vector_configs,
        properties=properties
    )
    
    logger.info(f"Collection created: {collection_name}")
    
    # Prepare objects for indexing
    logger.info("Preparing objects for indexing...")
    objects_to_index = []
    missing_embeddings = []
    missing_strings = []
    problem_hashes = []
    
    # The problematic hash you specifically mentioned
    debug_hash = "149694241953673352241727113736816909148"
    
    # Loop through each unique hash and its field types
    for hash_value, fields in tqdm(field_hash_map.items(), desc="Preparing objects"):
        # Special handling for debug hash
        if hash_value == debug_hash:
            logger.info(f"Processing debug hash {debug_hash}")
            logger.info(f"  Fields: {fields}")
        
        # Get string value
        string_value = None
        if hash_value in unique_strings:
            string_value = unique_strings[hash_value]
        else:
            missing_strings.append(hash_value)
            logger.warning(f"Missing string for hash: {hash_value}")
            # Try to find the string value in record data
            for record_id, record_fields in record_data.items():
                for field, field_hash in record_fields.items():
                    if field_hash == hash_value:
                        logger.info(f"Found hash {hash_value} in record {record_id}, field {field}")
                        # We don't have the string value, use a placeholder
                        string_value = f"[Unknown string for hash {hash_value}]"
                        break
                if string_value:
                    break
            
            if not string_value:
                logger.error(f"Cannot find string value for hash {hash_value}")
                continue
        
        # Get embedding vector
        embedding_vector = None
        if hash_value in embeddings:
            embedding_vector = embeddings[hash_value]
        else:
            missing_embeddings.append(hash_value)
            logger.warning(f"Missing embedding for hash: {hash_value}")
            
            # For hashes without embeddings, use a zero vector
            embedding_vector = [0.0] * 1536  # Use standard OpenAI embedding size
        
        # Create an object for each field type
        for field_type, count in fields.items():
            # For debug hash
            if hash_value == debug_hash:
                logger.info(f"  Creating object for field {field_type} with count {count}")
                
            # Check if this is a field type we support vectors for
            if field_type not in field_types:
                logger.warning(f"Unsupported field type: {field_type}")
                problem_hashes.append((hash_value, field_type))
                # Skip for now
                continue
                
            obj = {
                'hash': hash_value,
                'value': string_value,
                'field_type': field_type,
                'frequency': count,
                'vector': embedding_vector
            }
            
            objects_to_index.append(obj)
    
    logger.info(f"Prepared {len(objects_to_index)} objects to index")
    logger.info(f"Missing strings: {len(missing_strings)}")
    logger.info(f"Missing embeddings: {len(missing_embeddings)}")
    logger.info(f"Problem hashes with unsupported field types: {len(problem_hashes)}")
    
    # Index objects in batches
    batch_size = config['weaviate']['batch_size']
    total_indexed = 0
    errors = []
    
    # Track unique field type counts
    field_type_counts = {}
    
    logger.info(f"Indexing {len(objects_to_index)} objects in batches of {batch_size}")
    for i in range(0, len(objects_to_index), batch_size):
        batch = objects_to_index[i:i+batch_size]
        
        with collection.batch.dynamic() as batch_executor:
            for obj in batch:
                try:
                    # Generate UUID from hash and field type for idempotency
                    obj_uuid = generate_uuid5(f"{obj['hash']}_{obj['field_type']}")
                    
                    # Prepare object properties and vector
                    properties = {
                        'hash': obj['hash'],
                        'value': obj['value'],
                        'field_type': obj['field_type'],
                        'frequency': obj['frequency']
                    }
                    
                    # Track field type counts
                    field_type = obj['field_type']
                    if field_type not in field_type_counts:
                        field_type_counts[field_type] = 0
                    field_type_counts[field_type] += 1
                    
                    # Prepare vector - important to use the right field type as the named vector
                    vector = {
                        obj['field_type']: obj['vector']
                    }
                    
                    # Add object to batch
                    batch_executor.add_object(
                        properties=properties,
                        uuid=obj_uuid,
                        vector=vector
                    )
                    
                    total_indexed += 1
                    
                    # Special handling for debug hash
                    if obj['hash'] == debug_hash:
                        logger.info(f"Successfully indexed object for debug hash in field {obj['field_type']}")
                except Exception as e:
                    errors.append({
                        'hash': obj['hash'],
                        'field_type': obj['field_type'],
                        'error': str(e)
                    })
        
        logger.info(f"Indexed {i+len(batch)}/{len(objects_to_index)} objects")
    
    logger.info(f"Indexing completed: {total_indexed}/{len(objects_to_index)} objects indexed successfully")
    logger.info(f"Errors: {len(errors)}")
    
    if errors:
        error_sample = errors[:3]
        logger.info(f"First few errors: {error_sample}")
        # Write all errors to file for analysis
        error_file = Path(config['system']['output_dir']) / "indexing_errors.json"
        with open(error_file, 'w') as f:
            json.dump(errors, f, indent=2)
        logger.info(f"All errors written to {error_file}")
    
    # Verify indexing
    from weaviate.classes.aggregate import GroupByAggregate
    
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
    logger.info(f"Expected field counts: {field_type_counts}")
    
    # Compare counts
    logger.info("\nField Count Comparison:")
    logger.info("Field      | Expected | Weaviate | Difference")
    logger.info("-----------+----------+----------+-----------")
    
    for field in sorted(set(list(weaviate_field_counts.keys()) + list(field_type_counts.keys()))):
        expected = field_type_counts.get(field, 0)
        actual = weaviate_field_counts.get(field, 0)
        diff = actual - expected
        
        logger.info(f"{field:10} | {expected:8} | {actual:8} | {diff:+10}")
    
    # Check for the specific hash
    logger.info(f"\nVerifying debug hash: {debug_hash}")
    
    try:
        from weaviate.classes.query import Filter
        
        # Create filter for hash
        hash_filter = Filter.by_property("hash").equal(debug_hash)
        
        # Execute search
        results = collection.query.fetch_objects(
            filters=hash_filter,
            limit=10,
            include_vector=False
        )
        
        logger.info(f"Found {len(results.objects)} objects for debug hash:")
        for obj in results.objects:
            logger.info(f"  Field type: {obj.properties.get('field_type')}, Frequency: {obj.properties.get('frequency')}")
    except Exception as e:
        logger.error(f"Error checking debug hash: {e}")
    
    # Close client
    client.close()
    
    return True

def main():
    """Main entry point."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yml"
    
    # Load configuration
    config = load_config(config_path)
    if not config:
        return
    
    # Confirm action
    answer = input("This will DELETE the existing collection and rebuild it with ALL data. Are you sure? (yes/no): ")
    if answer.lower() != 'yes':
        logger.info("Fixed indexing aborted")
        return
    
    # Run the fixed indexer
    start_time = time.time()
    success = fixed_index(config)
    duration = time.time() - start_time
    
    if success:
        logger.info(f"Fixed indexing completed in {duration:.2f} seconds")
    else:
        logger.error(f"Fixed indexing failed")

if __name__ == "__main__":
    main()