#!/usr/bin/env python3
"""
Fix imputation stage for entity resolution pipeline.

This script:
1. Checks what collections exist in Weaviate
2. Ensures the collection name is consistent across stages
3. Creates the necessary collection if it's missing
"""

import os
import logging
import json
import sys
from pathlib import Path
import yaml
import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fix_imputation")

def load_config(config_path="config.yml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return None

def check_weaviate_collections(config):
    """
    Check what collections exist in Weaviate.
    
    Args:
        config (dict): Configuration parameters
        
    Returns:
        list: List of collection names
    """
    try:
        # Connect to Weaviate
        client = weaviate.connect_to_local(
            host=config['weaviate']['host'],
            port=config['weaviate']['port']
        )
        
        # Get list of collections
        collections = client.collections.list_all()
        
        logger.info(f"Found {len(collections)} collections in Weaviate:")
        for collection in collections:
            logger.info(f"  - {collection}")
        
        # Close client
        client.close()
        
        return collections
    
    except Exception as e:
        logger.error(f"Error connecting to Weaviate: {e}")
        return []

def fix_collection_name(config):
    """
    Fix collection name across configuration files and code.
    
    Args:
        config (dict): Configuration parameters
        
    Returns:
        bool: True if successful, False otherwise
    """
    collection_name = config['weaviate']['collection_name']
    logger.info(f"Collection name in config: {collection_name}")
    
    # Check if the collection exists
    collections = check_weaviate_collections(config)
    
    # If the collection doesn't exist, create it
    if collection_name not in collections:
        logger.info(f"Collection {collection_name} does not exist in Weaviate. Creating...")
        
        success = create_collection(config, collection_name)
        if not success:
            logger.error(f"Failed to create collection {collection_name}")
            return False
    
    return True

def create_collection(config, collection_name):
    """
    Create a collection in Weaviate.
    
    Args:
        config (dict): Configuration parameters
        collection_name (str): Name of collection to create
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Connect to Weaviate
        client = weaviate.connect_to_local(
            host=config['weaviate']['host'],
            port=config['weaviate']['port']
        )
        
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
        
        logger.info(f"Collection {collection_name} created successfully")
        
        # Close client
        client.close()
        
        return True
    
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def load_embeddings_and_index(config):
    """
    Load embeddings and index them in Weaviate.
    
    Args:
        config (dict): Configuration parameters
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Connect to Weaviate
        client = weaviate.connect_to_local(
            host=config['weaviate']['host'],
            port=config['weaviate']['port']
        )
        
        collection_name = config['weaviate']['collection_name']
        collection = client.collections.get(collection_name)
        
        # Load embeddings
        output_dir = Path(config['system']['output_dir'])
        embedding_index_path = output_dir / "embedding_index.json"
        
        if not embedding_index_path.exists():
            logger.error(f"Embedding index not found at {embedding_index_path}")
            return False
        
        with open(embedding_index_path, 'r') as f:
            embedding_index = json.load(f)
        
        location = embedding_index.get('location')
        
        if not location or location == "in-memory":
            logger.error("Embeddings are in-memory, cannot load")
            return False
        
        # Add the src directory to the Python path if needed
        src_dir = "src"
        if src_dir not in sys.path:
            sys.path.append(src_dir)
        
        # Try to import MMapDict
        try:
            from src.mmap_dict import MMapDict
            embeddings = MMapDict(location)
            logger.info(f"Loaded embeddings from {location}")
        except ImportError:
            # If import fails, try to load as a pickle file
            import pickle
            with open(location, 'rb') as f:
                embeddings = pickle.load(f)
                logger.info(f"Loaded embeddings using pickle from {location}")
        
        # Load unique strings
        unique_strings_path = output_dir / "unique_strings_sample.json"
        if not unique_strings_path.exists():
            logger.error(f"Unique strings not found at {unique_strings_path}")
            return False
        
        with open(unique_strings_path, 'r') as f:
            unique_strings = json.load(f)
        
        # Load field hash mapping
        field_hash_path = output_dir / "field_hash_index.json"
        if not field_hash_path.exists():
            logger.error(f"Field hash mapping not found at {field_hash_path}")
            return False
        
        with open(field_hash_path, 'r') as f:
            field_hash_mapping = json.load(f)
        
        # Index a subset of embeddings
        logger.info("Indexing a subset of embeddings...")
        
        # Get sample hashes
        sample_size = min(100, len(embeddings))
        sample_hashes = list(embeddings.keys())[:sample_size]
        
        indexed_count = 0
        
        with collection.batch.dynamic() as batch:
            for hash_value in sample_hashes:
                # Skip if no field types
                if hash_value not in field_hash_mapping:
                    continue
                
                # Skip if not in unique strings
                if hash_value not in unique_strings:
                    continue
                
                # Get string value and embedding
                string_value = unique_strings[hash_value]
                embedding_vector = embeddings[hash_value]
                
                # Create an object for each field type
                for field_type, count in field_hash_mapping[hash_value].items():
                    # Skip fields that don't have vector configurations
                    if field_type not in ["composite", "person", "title", "provision", "subjects"]:
                        continue
                    
                    # Generate UUID
                    import uuid
                    obj_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{hash_value}_{field_type}"))
                    
                    # Prepare properties
                    properties = {
                        'hash': hash_value,
                        'value': string_value,
                        'field_type': field_type,
                        'frequency': count
                    }
                    
                    # Prepare vector
                    vector = {
                        field_type: embedding_vector
                    }
                    
                    # Add object to batch
                    batch.add_object(
                        properties=properties,
                        uuid=obj_uuid,
                        vector=vector
                    )
                    
                    indexed_count += 1
        
        logger.info(f"Indexed {indexed_count} objects")
        
        # Verify indexing
        count_result = collection.aggregate.over_all(total_count=True)
        logger.info(f"Collection now contains {count_result.total_count} objects")
        
        # Close client
        client.close()
        
        return True
    
    except Exception as e:
        logger.error(f"Error indexing embeddings: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main entry point."""
    # Parse command-line arguments
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yml"
    
    # Load configuration
    config = load_config(config_path)
    if not config:
        return
    
    # Check and fix collection name
    if not fix_collection_name(config):
        logger.error("Failed to fix collection name")
        return
    
    # If collection was created, index some embeddings
    collections = check_weaviate_collections(config)
    collection_name = config['weaviate']['collection_name']
    
    if collection_name in collections:
        # Load and index embeddings
        if not load_embeddings_and_index(config):
            logger.error("Failed to index embeddings")
            return
    
    logger.info("Imputation stage fixed!")
    logger.info("\nNext steps:")
    logger.info("1. Run the imputation stage: python main.py --component impute")
    logger.info("2. Continue with the rest of the pipeline")

if __name__ == "__main__":
    main()
