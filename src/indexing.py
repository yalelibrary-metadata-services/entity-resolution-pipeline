"""
Weaviate indexing module for large-scale entity resolution.
"""
import os
import logging
import json
import time
import uuid
from pathlib import Path
from tqdm import tqdm
import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.query import MetadataQuery, Filter
from weaviate.util import generate_uuid5
from tenacity import retry, stop_after_attempt, wait_exponential
import numpy as np
import pickle

from src.utils import save_checkpoint, load_checkpoint, Timer, get_memory_usage
from src.mmap_dict import MMapDict

logger = logging.getLogger(__name__)

class Indexer:
    """
    Handles indexing of vector embeddings in Weaviate for large datasets.
    
    Features:
    - Efficient schema optimization for vector search
    - Batch indexing with automatic retries
    - Support for named vectors by field type
    - Idempotent operations for re-indexing
    """
    
    def __init__(self, config):
        """
        Initialize the indexer with configuration parameters.
        """
        self.config = config
        
        # Weaviate connection parameters
        self.weaviate_host = config['weaviate']['host']
        self.weaviate_port = config['weaviate']['port']
        self.collection_name = config['weaviate']['collection_name']
        
        # Connect to Weaviate
        self.client = self._connect_to_weaviate()
        
        # Batch size for indexing
        self.batch_size = config['weaviate']['batch_size']
        
        # Weaviate schema parameters
        self.ef = config['weaviate']['ef']
        self.max_connections = config['weaviate']['max_connections']
        self.ef_construction = config['weaviate']['ef_construction']
        self.distance_metric = config['weaviate']['distance_metric']
        
        # Memory-mapped file paths
        self.mmap_dir = Path(self.config['system']['temp_dir']) / "mmap"
        
        logger.info("Indexer initialized with Weaviate at %s:%s", 
                   self.weaviate_host, self.weaviate_port)

    def execute(self, checkpoint=None):
        """Execute indexing of embeddings in Weaviate with scalable approach."""
        # Load checkpoint if provided
        if checkpoint and os.path.exists(checkpoint):
            state = load_checkpoint(checkpoint)
            indexed_hashes = set(state.get('indexed_hashes', []))
            logger.info(f"Resumed indexing from checkpoint: {checkpoint} with {len(indexed_hashes)} indexed hashes")
        else:
            # Check for existing checkpoints
            checkpoint_dir = Path(self.config['system']['checkpoint_dir'])
            indexing_checkpoint = checkpoint_dir / "indexing_final.ckpt"
            
            if indexing_checkpoint.exists():
                state = load_checkpoint(indexing_checkpoint)
                indexed_hashes = set(state.get('indexed_hashes', []))
                logger.info(f"Found existing indexing checkpoint with {len(indexed_hashes)} indexed hashes")
            else:
                indexed_hashes = set()
        
        # Check if indexing is already complete
        if self._check_indexing_complete():
            logger.info("Indexing is already complete, skipping indexing")
            stats = self._get_collection_stats()
            return {
                'objects_indexed': stats.get('object_count', 0),
                'total_embedded': stats.get('object_count', 0),
                'completion_percentage': 100.0,
                'total_in_collection': stats.get('object_count', 0),
                'duration': 0.0,
                'skipped': True
            }
        
        # Verify or create schema
        self._create_or_update_schema()
        
        # Load embedded hashes and unique strings
        embedded_hashes, unique_strings, field_hash_mapping = self._load_data()
        
        # Filter hashes that haven't been indexed yet
        hashes_to_index = [h for h in embedded_hashes if h not in indexed_hashes]
        
        # If all hashes are already indexed, return early
        if not hashes_to_index:
            stats = self._get_collection_stats()
            logger.info(f"All {len(embedded_hashes)} hashes are already indexed")
            return {
                'objects_indexed': 0,
                'total_embedded': len(embedded_hashes),
                'completion_percentage': 100.0,
                'total_in_collection': stats.get('object_count', 0),
                'duration': 0.0,
                'skipped': True
            }
        
        logger.info(f"Indexing {len(hashes_to_index)}/{len(embedded_hashes)} embedded hashes")
        
        if self.config['system']['mode'] == 'dev':
            # In dev mode, limit the number of hashes to index
            dev_sample_size = min(self.config['system']['dev_sample_size'], len(hashes_to_index))
            hashes_to_index = hashes_to_index[:dev_sample_size]
            logger.info("Dev mode: limited to %d hashes", len(hashes_to_index))
        
        # Load embeddings
        embeddings = self._load_embeddings()
        
        # Create batches of objects to index
        batches = self._create_batches(
            hashes_to_index, 
            embeddings, 
            unique_strings, 
            field_hash_mapping
        )
        
        # Index objects in batches
        total_indexed = 0
        batch_durations = []
        
        with Timer() as timer:
            # Get the collection
            collection = self._execute_weaviate_operation(
                lambda: self.client.collections.get(self.collection_name)
            )
            
            for batch_idx, batch in enumerate(tqdm(batches, desc="Indexing batches")):
                batch_start = time.time()
                try:
                    # Index batch
                    self._index_batch(collection, batch)
                    
                    # Update indexed hashes
                    batch_hashes = [obj['hash'] for obj in batch]
                    indexed_hashes.update(batch_hashes)
                    total_indexed += len(batch)
                    
                    # Record batch duration
                    batch_duration = time.time() - batch_start
                    batch_durations.append(batch_duration)
                    
                    # Log progress
                    if (batch_idx + 1) % 10 == 0:
                        logger.info("Indexed %d/%d batches, %d objects, %.2f seconds/batch", 
                                   batch_idx + 1, len(batches), total_indexed, 
                                   sum(batch_durations[-10:]) / 10)
                        logger.info("Memory usage: %.2f GB", get_memory_usage())
                    
                    # Save checkpoint periodically
                    if self.config['data']['checkpoints_enabled'] and (batch_idx + 1) % 50 == 0:
                        self._save_checkpoint(indexed_hashes, batch_idx)
                
                except Exception as e:
                    logger.error("Error indexing batch %d: %s", batch_idx, str(e))
                    
                    # Save checkpoint on error
                    self._save_checkpoint(indexed_hashes, f"error_{batch_idx}")
                    
                    # Continue with next batch after a short delay
                    time.sleep(5)
                    continue
        
        # Save final results
        self._save_results(indexed_hashes)
        
        # Get collection statistics
        collection_stats = self._get_collection_stats()
        
        # Calculate completion percentage
        completion_pct = (len(indexed_hashes) / len(embedded_hashes)) * 100 if embedded_hashes else 0
        
        results = {
            'objects_indexed': total_indexed,
            'total_embedded': len(embedded_hashes),
            'completion_percentage': completion_pct,
            'total_in_collection': collection_stats.get('object_count', 0),
            'duration': timer.duration,
            'batch_durations': batch_durations
        }
        
        logger.info("Indexing completed: %d objects indexed (%.2f%%), %.2f seconds",
                   total_indexed, completion_pct, timer.duration)
        
        return results

    def _check_indexing_complete(self):
        """
        Check if indexing is already complete by comparing object count to expected count.
        
        Returns:
            bool: True if indexing is complete, False otherwise
        """
        try:
            # Check if collection exists
            collections = self._execute_weaviate_operation(
                lambda: self.client.collections.list_all()
            )
            
            if self.collection_name not in collections:
                logger.info(f"Collection {self.collection_name} does not exist in Weaviate")
                return False
            
            # Get collection statistics
            collection = self._execute_weaviate_operation(
                lambda: self.client.collections.get(self.collection_name)
            )
            
            count_result = self._execute_weaviate_operation(
                lambda: collection.aggregate.over_all(total_count=True)
            )
            
            # Get field distribution
            from weaviate.classes.aggregate import GroupByAggregate
            
            field_type_result = self._execute_weaviate_operation(
                lambda: collection.aggregate.over_all(
                    group_by=GroupByAggregate(prop="field_type"),
                    total_count=True
                )
            )
            
            # Load embedded hashes count from checkpoint
            checkpoint_dir = Path(self.config['system']['checkpoint_dir'])
            embedding_checkpoint = checkpoint_dir / "embedding_final.ckpt"
            
            if embedding_checkpoint.exists():
                state = load_checkpoint(embedding_checkpoint)
                processed_hashes_count = len(state.get('processed_hashes', []))
                
                # Estimate expected object count (one object per field type per hash)
                # This is a rough estimate since we don't know how many field types each hash has
                # Assume average of 3 field types per hash
                expected_count = processed_hashes_count * 3
                
                # If we have at least 90% of expected objects, consider indexing as complete
                completion_percentage = (count_result.total_count / max(1, expected_count)) * 100
                logger.info(f"Found {count_result.total_count} objects in Weaviate ({completion_percentage:.2f}% of expected)")
                
                return completion_percentage >= 90.0
            
            return False
        
        except Exception as e:
            logger.error(f"Error checking indexing completion: {str(e)}")
            return False

    def _connect_to_weaviate(self):
        """
        Connect to Weaviate instance with retry logic.
        """
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        def connect_with_retry():
            try:
                # Connect to Weaviate
                client = weaviate.connect_to_local(
                    host=self.weaviate_host,
                    port=self.weaviate_port
                )
                
                # Test connection
                client.is_ready()
                logger.info("Connected to Weaviate at %s:%s", 
                           self.weaviate_host, self.weaviate_port)
                
                return client
            
            except Exception as e:
                logger.error("Error connecting to Weaviate: %s", str(e))
                raise
        
        return connect_with_retry()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _execute_weaviate_operation(self, operation_func):
        """
        Execute a Weaviate operation with retry logic.
        """
        try:
            return operation_func()
        except Exception as e:
            logger.error("Error executing Weaviate operation: %s", str(e))
            raise

    def _create_or_update_schema(self):
        """
        Create or update Weaviate schema for entity resolution.
        """
        try:
            # Check if collection exists
            collections = self._execute_weaviate_operation(
                lambda: self.client.collections.list_all()
            )
            
            collection_exists = self.collection_name in collections
            
            if collection_exists:
                logger.info("Collection %s already exists", self.collection_name)
                
                # Check if schema needs updates
                collection = self._execute_weaviate_operation(
                    lambda: self.client.collections.get(self.collection_name)
                )
                
                # Check if properties exist and add missing ones
                # This is a simplified version - in production you'd want
                # to check property types and vector index settings too
                
                # For now, we'll just log that we're keeping the existing schema
                logger.info("Using existing collection schema")
                return
            
            logger.info("Collection %s does not exist in Weaviate", self.collection_name)
            
            # Create collection with named vectors
            field_types = ["composite", "person", "title", "provision", "subjects"]
            
            # Import required enums and config classes
            from weaviate.classes.config import Configure, Property, DataType, VectorDistances
            from weaviate.classes.config import StopwordsPreset
            
            # Configure named vectors for each field type
            vector_configs = []
            for field_type in field_types:
                vector_configs.append(
                    Configure.NamedVectors.none(
                        name=field_type,
                        vector_index_config=Configure.VectorIndex.hnsw(
                            ef=self.ef,
                            max_connections=self.max_connections,
                            ef_construction=self.ef_construction,
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
            
            # Create collection with optimized settings - use Configure helpers
            inverted_index = Configure.inverted_index(
                index_timestamps=True,
                index_property_length=True,
                index_null_state=True                
            )
            
            try:
                # Create collection with updated configuration
                self._execute_weaviate_operation(
                    lambda: self.client.collections.create(
                        name=self.collection_name,
                        description="Entity resolution vectors collection",
                        vectorizer_config=vector_configs,
                        properties=properties,
                        inverted_index_config=inverted_index
                    )
                )
                
                logger.info("Created Weaviate collection: %s with optimized schema", 
                        self.collection_name)
                        
            except Exception as e:
                logger.error(f"Error creating collection with configured schema: {e}")
                
                # Fallback to simpler schema without inverted index config
                logger.info("Trying with simplified schema...")
                self._execute_weaviate_operation(
                    lambda: self.client.collections.create(
                        name=self.collection_name,
                        description="Entity resolution vectors collection",
                        vectorizer_config=vector_configs,
                        properties=properties
                    )
                )
                
                logger.info("Created Weaviate collection with simplified schema")
        
        except Exception as e:
            logger.error("Error creating Weaviate schema: %s", str(e))
            raise

    def _load_data(self):
        """
        Load embedded hashes, unique strings, and field hash mapping.
        """
        try:
            # Load embedded hashes
            output_dir = Path(self.config['system']['output_dir'])
            
            # Try to load from embedding index first
            embedding_index_path = output_dir / "embedding_index.json"
            if embedding_index_path.exists():
                with open(embedding_index_path, 'r') as f:
                    embedding_index = json.load(f)
                
                location = embedding_index.get('location')
                
                if location != "in-memory" and os.path.exists(location):
                    # For large datasets, load hash arrays instead of full embeddings
                    embedded_mmap = MMapDict(location)
                    embedded_hashes = list(embedded_mmap.keys())
                else:
                    # Fall back to embedded_hashes.json
                    with open(output_dir / "embedded_hashes.json", 'r') as f:
                        embedded_hashes = json.load(f)
            else:
                # Fall back to embedded_hashes.json
                with open(output_dir / "embedded_hashes.json", 'r') as f:
                    embedded_hashes = json.load(f)
            
            # Load unique strings
            unique_strings_index_path = output_dir / "unique_strings_index.json"
            if unique_strings_index_path.exists():
                with open(unique_strings_index_path, 'r') as f:
                    unique_strings_index = json.load(f)
                
                location = unique_strings_index.get('location')
                
                if location != "in-memory" and os.path.exists(location):
                    # For large datasets, use memory-mapped dictionary
                    unique_strings = MMapDict(location)
                else:
                    # Fall back to sample
                    with open(output_dir / "unique_strings_sample.json", 'r') as f:
                        unique_strings = json.load(f)
            else:
                # Fall back to sample
                with open(output_dir / "unique_strings_sample.json", 'r') as f:
                    unique_strings = json.load(f)
            
            # Load field hash mapping
            field_hash_index_path = output_dir / "field_hash_index.json"
            if field_hash_index_path.exists():
                with open(field_hash_index_path, 'r') as f:
                    field_hash_index = json.load(f)
                
                location = field_hash_index.get('location')
                
                if location != "in-memory" and os.path.exists(location):
                    # For large datasets, use memory-mapped dictionary
                    field_hash_mapping = MMapDict(location)
                else:
                    # Fall back to sample
                    with open(output_dir / "field_hash_mapping_sample.json", 'r') as f:
                        field_hash_mapping = json.load(f)
            else:
                # Fall back to sample
                with open(output_dir / "field_hash_mapping_sample.json", 'r') as f:
                    field_hash_mapping = json.load(f)
            
            logger.info("Loaded %d embedded hashes, %d unique strings, %d field mappings",
                       len(embedded_hashes), len(unique_strings), len(field_hash_mapping))
            
            return embedded_hashes, unique_strings, field_hash_mapping
        
        except Exception as e:
            logger.error("Error loading data: %s", str(e))
            return [], {}, {}

    def _load_embeddings(self):
        """
        Load embeddings with support for memory-mapped storage.
        """
        try:
            output_dir = Path(self.config['system']['output_dir'])
            
            # Try to load from embedding index first
            embedding_index_path = output_dir / "embedding_index.json"
            if embedding_index_path.exists():
                with open(embedding_index_path, 'r') as f:
                    embedding_index = json.load(f)
                
                location = embedding_index.get('location')
                
                if location != "in-memory" and os.path.exists(location):
                    # For large datasets, use memory-mapped dictionary
                    return MMapDict(location)
            
            # Try to load from embedding checkpoint
            checkpoint_dir = Path(self.config['system']['checkpoint_dir'])
            embedding_checkpoint = checkpoint_dir / "embedding_final.ckpt"
            
            if embedding_checkpoint.exists():
                state = load_checkpoint(embedding_checkpoint)
                if 'embeddings' in state:
                    return state['embeddings']
                elif 'embeddings_location' in state and os.path.exists(state['embeddings_location']):
                    return MMapDict(state['embeddings_location'])
            
            # Fall back to embeddings_sample.json if it exists
            embeddings_sample_path = output_dir / "embeddings_sample.json"
            if embeddings_sample_path.exists():
                with open(embeddings_sample_path, 'r') as f:
                    return json.load(f)
            
            # If all else fails, return empty dict
            logger.warning("Could not load embeddings, returning empty dictionary")
            return {}
        
        except Exception as e:
            logger.error("Error loading embeddings: %s", str(e))
            return {}

    def _create_batches(self, hashes, embeddings, unique_strings, field_hash_mapping):
        """
        Create batches of objects to index with optimized memory usage and improved field handling.
        """
        batch_size = self.batch_size
        batches = []
        current_batch = []
        
        # Debugging hash that needs to be tracked (you mentioned it was missing)
        debug_hash = "149694241953673352241727113736816909148"
        if debug_hash in hashes:
            logger.info(f"Debug hash {debug_hash} is in hashes to index")
            if debug_hash in field_hash_mapping:
                logger.info(f"Debug hash field types: {field_hash_mapping[debug_hash]}")
            else:
                logger.error(f"Debug hash not found in field_hash_mapping!")
        
        # Process in smaller chunks to avoid loading all embeddings at once
        for i in range(0, len(hashes), 1000):
            chunk_hashes = hashes[i:i+1000]
            
            # Process each hash in the chunk
            for hash_value in chunk_hashes:
                # Skip if embedding is missing
                if hash_value not in embeddings:
                    logger.debug(f"Skipping hash {hash_value}: No embedding found")
                    continue
                
                # Skip if not in unique strings (shouldn't happen but check anyway)
                if hash_value not in unique_strings:
                    logger.debug(f"Skipping hash {hash_value}: No unique string found")
                    continue
                
                # IMPORTANT FIX: Check if not in field_hash_mapping and log the error
                if hash_value not in field_hash_mapping:
                    logger.error(f"Critical: Hash {hash_value} not found in field_hash_mapping but exists in unique_strings!")
                    # We can build a fallback field mapping to fix it
                    field_counts = self._find_field_usage_for_hash(hash_value)
                    if field_counts:
                        logger.info(f"Built fallback field mapping for {hash_value}: {field_counts}")
                        field_hash_mapping[hash_value] = field_counts
                    else:
                        logger.warning(f"Unable to determine field types for hash {hash_value}, using generic 'value'")
                        # Add a default field type to allow processing
                        field_hash_mapping[hash_value] = {"value": 1}
                
                # Get string value and embedding
                string_value = unique_strings[hash_value]
                embedding_vector = embeddings[hash_value]
                
                # Get field types and counts
                field_types = field_hash_mapping[hash_value]
                
                # Calculate total frequency
                total_frequency = sum(field_types.values())
                
                # Create an object for each field type
                for field_type, count in field_types.items():
                    # Special debugging for our problematic hash
                    if hash_value == debug_hash:
                        logger.info(f"Creating object for debug hash with field_type={field_type}, count={count}")
                    
                    obj = {
                        'hash': hash_value,
                        'value': string_value,
                        'field_type': field_type,
                        'frequency': count,
                        'vector': embedding_vector
                    }
                    
                    current_batch.append(obj)
                    
                    # If batch is full, add to batches and start a new one
                    if len(current_batch) >= batch_size:
                        batches.append(current_batch)
                        current_batch = []
            
            # Log progress
            logger.debug(f"Processed {i + len(chunk_hashes)}/{len(hashes)} hashes, created {len(batches)} batches")
        
        # Add last batch if not empty
        if current_batch:
            batches.append(current_batch)
        
        return batches

    def _find_field_usage_for_hash(self, hash_value):
        """
        Search record_field_hashes to find where a hash is used.
        This is a fallback to rebuild field mappings for missing hashes.
        
        Args:
            hash_value (str): Hash to search for
            
        Returns:
            dict: Field counts for this hash
        """
        field_counts = {}
        
        # Load record field hashes directory from output dir
        output_dir = Path(self.config['system']['output_dir'])
        record_field_hashes_path = output_dir / "record_field_hashes.json"
        
        try:
            # First try to load full data if available
            if record_field_hashes_path.exists():
                with open(record_field_hashes_path, 'r') as f:
                    record_field_hashes = json.load(f)
                    
                # Search for hash in all records and all fields
                for record_id, fields in record_field_hashes.items():
                    for field, field_hash in fields.items():
                        if field_hash == hash_value:
                            if field not in field_counts:
                                field_counts[field] = 0
                            field_counts[field] += 1
            else:
                # If not available, try the sample file
                record_field_hashes_path = output_dir / "record_field_hashes_sample.json"
                if record_field_hashes_path.exists():
                    with open(record_field_hashes_path, 'r') as f:
                        record_field_hashes = json.load(f)
                        
                    # Search for hash in all records and all fields
                    for record_id, fields in record_field_hashes.items():
                        for field, field_hash in fields.items():
                            if field_hash == hash_value:
                                if field not in field_counts:
                                    field_counts[field] = 0
                                field_counts[field] += 1
        
        except Exception as e:
            logger.error(f"Error finding field usage for hash: {e}")
        
        return field_counts

    def _index_batch(self, collection, batch):
        """
        Index a batch of objects in Weaviate with batching and error handling.
        """
        try:
            # Use Weaviate's batch import
            with collection.batch.dynamic() as batch_executor:
                for obj in batch:
                    # Generate UUID from hash and field type for idempotency
                    # This ensures that re-running the indexing won't create duplicates
                    obj_uuid = generate_uuid5(f"{obj['hash']}_{obj['field_type']}")
                    
                    # Prepare object properties (excluding vector)
                    properties = {
                        'hash': obj['hash'],
                        'value': obj['value'],
                        'field_type': obj['field_type'],
                        'frequency': obj['frequency']
                    }
                    
                    # Prepare vector
                    vector = {
                        obj['field_type']: obj['vector']
                    }
                    
                    # Add object to batch
                    batch_executor.add_object(
                        properties=properties,
                        uuid=obj_uuid,
                        vector=vector
                    )
        
        except Exception as e:
            logger.error("Error indexing batch: %s", str(e))
            raise

    def _save_checkpoint(self, indexed_hashes, batch_idx):
        """
        Save checkpoint for indexing progress.
        """
        try:
            checkpoint_dir = Path(self.config['system']['checkpoint_dir'])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = checkpoint_dir / f"indexing_{batch_idx}.ckpt"
            
            # Save checkpoint
            save_checkpoint({
                'indexed_hashes': list(indexed_hashes)
            }, checkpoint_path)
            
            logger.info("Saved indexing checkpoint at %s with %d indexed hashes", 
                       checkpoint_path, len(indexed_hashes))
        
        except Exception as e:
            logger.error("Error saving checkpoint: %s", str(e))

    def _get_collection_stats(self):
        """
        Get statistics for the Weaviate collection.
        """
        try:
            collection = self._execute_weaviate_operation(
                lambda: self.client.collections.get(self.collection_name)
            )
            
            # Get object count
            count_result = self._execute_weaviate_operation(
                lambda: collection.aggregate.over_all(total_count=True)
            )
            
            # Get field type distribution
            from weaviate.classes.aggregate import GroupByAggregate
            
            field_type_result = self._execute_weaviate_operation(
                lambda: collection.aggregate.over_all(
                    group_by=GroupByAggregate(prop="field_type"),
                    total_count=True
                )
            )
            
            field_counts = {}
            for group in field_type_result.groups:
                field_counts[group.grouped_by.value] = group.total_count
            
            stats = {
                'object_count': count_result.total_count,
                'field_counts': field_counts
            }
            
            return stats
        
        except Exception as e:
            logger.error("Error getting collection statistics: %s", str(e))
            return {'object_count': 0, 'field_counts': {}}

    def _save_results(self, indexed_hashes):
        """
        Save indexing results.
        """
        output_dir = Path(self.config['system']['output_dir'])
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save list of indexed hashes (up to 10,000 for reference)
        sample_hashes = list(indexed_hashes)[:10000]
        with open(output_dir / "indexed_hashes.json", 'w') as f:
            json.dump(sample_hashes, f)
        
        # Save collection statistics
        stats = self._get_collection_stats()
        with open(output_dir / "collection_statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save indexing metadata
        indexing_metadata = {
            'collection_name': self.collection_name,
            'indexed_count': len(indexed_hashes),
            'collection_count': stats.get('object_count', 0),
            'field_distribution': stats.get('field_counts', {}),
            'index_parameters': {
                'ef': self.ef,
                'max_connections': self.max_connections,
                'ef_construction': self.ef_construction,
                'distance_metric': self.distance_metric
            }
        }
        
        with open(output_dir / "indexing_metadata.json", 'w') as f:
            json.dump(indexing_metadata, f, indent=2)
        
        # Save final checkpoint
        checkpoint_path = Path(self.config['system']['checkpoint_dir']) / "indexing_final.ckpt"
        save_checkpoint({
            'indexed_hashes': list(indexed_hashes)[:100000]  # Limit to 100K for checkpoint size
        }, checkpoint_path)
        
        logger.info("Indexing results saved to %s", output_dir)
        debug_stats = debug_weaviate_collections()
        logger.info(f"Weaviate collection stats: {debug_stats}")

    def search_by_vector(self, vector, field_type, limit=100, threshold=0.7):
        """
        Search for similar objects by vector.
        """
        try:
            collection = self.client.collections.get(self.collection_name)
            
            # Create filter for field type
            field_filter = Filter.by_property("field_type").equal(field_type)
            
            # Execute search
            results = collection.query.near_vector(
                near_vector={field_type: vector},
                limit=limit,
                filters=field_filter,
                return_metadata=MetadataQuery(distance=True),
                include_vector=True
            )
            
            # Filter by threshold
            filtered_results = []
            for obj in results.objects:
                similarity = 1.0 - obj.metadata.distance
                if similarity >= threshold:
                    filtered_results.append({
                        'hash': obj.properties.get('hash'),
                        'value': obj.properties.get('value'),
                        'field_type': obj.properties.get('field_type'),
                        'similarity': similarity,
                        'vector': obj.vector.get(field_type) if obj.vector else None
                    })
            
            return filtered_results
        
        except Exception as e:
            logger.error("Error searching by vector: %s", str(e))
            return []

    def verify_index_consistency(self):
        """
        Verify consistency between indexed objects and embeddings.
        """
        try:
            # Load embedded and indexed hashes
            output_dir = Path(self.config['system']['output_dir'])
            
            with open(output_dir / "embedded_hashes.json", 'r') as f:
                embedded_hashes = json.load(f)
            
            with open(output_dir / "indexed_hashes.json", 'r') as f:
                indexed_hashes = json.load(f)
            
            # Get collection statistics
            collection = self.client.collections.get(self.collection_name)
            count_result = collection.aggregate.over_all(total_count=True)
            
            # Calculate metrics
            embedded_count = len(embedded_hashes)
            indexed_count = len(indexed_hashes)
            collection_count = count_result.total_count
            
            # Note: collection_count should be higher than indexed_count
            # because we create multiple objects per hash (one per field type)
            
            missing_count = embedded_count - indexed_count
            
            consistency_check = {
                'embedded_count': embedded_count,
                'indexed_count': indexed_count,
                'collection_count': collection_count,
                'missing_count': missing_count,
                'missing_percentage': (missing_count / embedded_count * 100) if embedded_count > 0 else 0,
                'is_consistent': missing_count == 0
            }
            
            return consistency_check
        
        except Exception as e:
            logger.error("Error verifying index consistency: %s", str(e))
            return {
                'error': str(e),
                'is_consistent': False
            }

    def __del__(self):
        """
        Clean up resources when object is garbage collected.
        """
        try:
            # Close Weaviate client if it exists
            if hasattr(self, 'client') and self.client:
                logger.debug("Closing Weaviate client connection")
                self.client.close()
                self.client = None
        except Exception as e:
            logger.error("Error closing Weaviate client: %s", str(e))

def debug_weaviate_collections():
    """
    Debug function to analyze the content of Weaviate collections.
    """
    try:
        # Connect to Weaviate
        client = weaviate.connect_to_local()
        
        # Get list of collections
        collections = client.collections.list_all()
        logger.info(f"Collections in Weaviate: {collections}")
        
        # Get the collection
        collection_name = "UniqueStringsByField"  # Update with your actual collection name
        if collection_name in collections:
            collection = client.collections.get(collection_name)
            
            # Get total count
            count_result = collection.aggregate.over_all(total_count=True)
            logger.info(f"Total objects in collection: {count_result.total_count}")
            
            # Analyze distribution by field_type
            from weaviate.classes.aggregate import GroupByAggregate
            
            field_type_result = collection.aggregate.over_all(
                group_by=GroupByAggregate(prop="field_type"),
                total_count=True
            )
            
            logger.info("Field type distribution:")
            for group in field_type_result.groups:
                logger.info(f"  {group.grouped_by.value}: {group.total_count}")
            
            # Sample a few objects to verify
            
            
            results = collection.query.fetch_objects(limit=5, include_vector=True)
            logger.info(f"Sample objects: {len(results.objects)}")
            
            for obj in results.objects:
                logger.info(f"Object fields: {obj.properties}")
                logger.info(f"Vector dimensions: {len(next(iter(obj.vector.values()))) if obj.vector else 'No vector'}")
            
            # Close client
            client.close()
            
            return {
                "total_count": count_result.total_count,
                "field_distribution": {
                    group.grouped_by.value: group.total_count for group in field_type_result.groups
                }
            }
        else:
            logger.warning(f"Collection {collection_name} not found")
            return {}
    
    except Exception as e:
        logger.error(f"Error analyzing Weaviate collections: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}