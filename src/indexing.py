"""
Weaviate indexing module for entity resolution.

This module provides the Indexer class, which handles the indexing of
vector embeddings in Weaviate for efficient similarity search.
"""

import os
import logging
import json
import time
import uuid
from pathlib import Path
from tqdm import tqdm
import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances, Tokenization
from weaviate.classes.query import MetadataQuery, Filter
from weaviate.util import generate_uuid5
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils import save_checkpoint, load_checkpoint, Timer, update_stage_metrics

logger = logging.getLogger(__name__)

class Indexer:
    """
    Handles indexing of vector embeddings in Weaviate.
    
    Features:
    - Creation of Weaviate schema
    - Batch indexing of embeddings
    - Named vector indexing for each field
    - Checkpointing for resuming interrupted indexing
    """
    
    def __init__(self, config):
        """
        Initialize the indexer with configuration parameters.
        
        Args:
            config (dict): Configuration parameters
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
        
        logger.info("Indexer initialized with Weaviate at %s:%s", 
                   self.weaviate_host, self.weaviate_port)

    def execute(self, checkpoint=None):
        """
        Execute indexing of embeddings in Weaviate.
        
        Args:
            checkpoint (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Indexing results
        """
        # Load checkpoint if provided
        if checkpoint and os.path.exists(checkpoint):
            state = load_checkpoint(checkpoint)
            indexed_hashes = set(state.get('indexed_hashes', []))
            logger.info("Resumed indexing from checkpoint: %s", checkpoint)
        else:
            indexed_hashes = set()
        
        # Check if collection exists, create if not
        self._create_or_update_schema()
        
        # Load embeddings and string data
        embeddings, unique_strings, field_hash_mapping = self._load_data()
        
        # Filter embeddings that haven't been indexed yet
        embeddings_to_index = {h: embeddings[h] for h in embeddings if h not in indexed_hashes}
        
        logger.info("Indexing %d/%d embeddings", len(embeddings_to_index), len(embeddings))
        
        if self.config['system']['mode'] == 'dev':
            # In dev mode, limit the number of embeddings to index
            dev_sample_size = min(self.config['system']['dev_sample_size'], len(embeddings_to_index))
            hash_sample = list(embeddings_to_index.keys())[:dev_sample_size]
            embeddings_to_index = {h: embeddings_to_index[h] for h in hash_sample}
            logger.info("Dev mode: limited to %d embeddings", len(embeddings_to_index))
        
        # Index embeddings in batches
        total_indexed = 0
        batch_durations = []
        
        with Timer() as timer:
            # Get the collection
            collection = self._execute_weaviate_operation(
                lambda: self.client.collections.get(self.collection_name)
            )
            
            # Create batches of objects to index
            batches = self._create_batches(embeddings_to_index, unique_strings, field_hash_mapping)
            
            for batch_idx, batch in enumerate(tqdm(batches, desc="Indexing batches")):
                batch_start = time.time()
                try:
                    # Index batch
                    with collection.batch.dynamic() as batch_executor:
                        for obj in batch:
                            # Generate UUID from hash
                            obj_uuid = generate_uuid5(obj['hash'])
                            
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
                    
                    # Update indexed hashes
                    batch_hashes = [obj['hash'] for obj in batch]
                    indexed_hashes.update(batch_hashes)
                    total_indexed += len(batch)
                    
                    # Record batch duration
                    batch_duration = time.time() - batch_start
                    batch_durations.append(batch_duration)
                    
                    # Log progress
                    if (batch_idx + 1) % 10 == 0:
                        logger.info("Indexed %d/%d batches, %d objects", 
                                   batch_idx + 1, len(batches), total_indexed)
                    
                    # Save checkpoint
                    if self.config['data']['checkpoints_enabled'] and (batch_idx + 1) % 50 == 0:
                        checkpoint_path = Path(self.config['system']['checkpoint_dir']) / f"indexing_{batch_idx}.ckpt"
                        save_checkpoint({
                            'indexed_hashes': list(indexed_hashes)
                        }, checkpoint_path)
                
                except Exception as e:
                    logger.error("Error indexing batch %d: %s", batch_idx, str(e))
                    
                    # Save checkpoint on error
                    error_checkpoint = Path(self.config['system']['checkpoint_dir']) / f"indexing_error_{batch_idx}.ckpt"
                    save_checkpoint({
                        'indexed_hashes': list(indexed_hashes)
                    }, error_checkpoint)
                    
                    # Continue with next batch
                    continue
        
        # Save final results
        self._save_results(indexed_hashes)
        
        # Get collection statistics
        collection_stats = self._get_collection_stats()
        
        results = {
            'objects_indexed': total_indexed,
            'total_in_collection': collection_stats.get('object_count', 0),
            'duration': timer.duration,
            'batch_durations': batch_durations
        }
        
        # Update monitoring metrics
        update_stage_metrics('index', results)
        
        logger.info("Indexing completed: %d objects indexed in %.2f seconds",
                   total_indexed, timer.duration)
        
        return results

    def _connect_to_weaviate(self):
        """
        Connect to Weaviate instance with retry logic.
        
        Returns:
            weaviate.Client: Weaviate client
        """
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        def connect_with_retry():
            try:
                # Construct Weaviate connection URL
                connection_url = f"http://{self.weaviate_host}:{self.weaviate_port}"
                
                # Connect to Weaviate
                client = weaviate.connect_to_local(
                    # host=self.weaviate_host,
                    # port=self.weaviate_port,
                    # grpc_port=None,  # Use default
                    # headers={}
                )
                
                # Test connection
                client.is_ready()
                logger.info("Connected to Weaviate at %s", connection_url)
                
                return client
            
            except Exception as e:
                logger.error("Error connecting to Weaviate: %s", str(e))
                raise
        
        return connect_with_retry()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _execute_weaviate_operation(self, operation_func):
        """
        Execute a Weaviate operation with retry logic.
        
        Args:
            operation_func (callable): Function to execute
            
        Returns:
            Any: Result of the operation
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
            
            collection_exists = any(c == self.collection_name for c in collections)
            
            if collection_exists:
                logger.info("Collection %s already exists", self.collection_name)
                # TODO: Check if schema needs updates
                return
            
            # Create collection with named vectors
            field_types = ["composite", "person", "title", "provision", "subjects"]
            
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
                Property(name="hash", data_type=DataType.TEXT, index_filterable=True),
                Property(name="value", data_type=DataType.TEXT, index_filterable=True, tokenization=Tokenization.FIELD),
                Property(name="field_type", data_type=DataType.TEXT, index_filterable=True),
                Property(name="frequency", data_type=DataType.NUMBER, index_filterable=True)
            ]
            
            # Create collection
            self._execute_weaviate_operation(
                lambda: self.client.collections.create(
                    name=self.collection_name,
                    vectorizer_config=vector_configs,
                    properties=properties
                )
            )
            
            logger.info("Created Weaviate collection: %s", self.collection_name)
        
        except Exception as e:
            logger.error("Error creating Weaviate schema: %s", str(e))
            raise

    def _load_data(self):
        """
        Load embeddings and related data.
        
        Returns:
            tuple: (embeddings, unique_strings, field_hash_mapping)
        """
        try:
            # Load embeddings from checkpoint
            embedding_checkpoint = Path(self.config['system']['checkpoint_dir']) / "embedding_final.ckpt"
            if os.path.exists(embedding_checkpoint):
                embedding_state = load_checkpoint(embedding_checkpoint)
                embeddings = embedding_state.get('embeddings', {})
            else:
                # Alternative: Load sample for development
                output_dir = Path(self.config['system']['output_dir'])
                with open(output_dir / "embeddings_sample.json", 'r') as f:
                    embeddings = json.load(f)
            
            # Load unique strings
            output_dir = Path(self.config['system']['output_dir'])
            with open(output_dir / "unique_strings_sample.json", 'r') as f:
                unique_strings = json.load(f)
            
            # Load field hash mapping
            with open(output_dir / "field_hash_mapping_sample.json", 'r') as f:
                field_hash_mapping = json.load(f)
            
            logger.info("Loaded %d embeddings, %d unique strings, %d field mappings",
                       len(embeddings), len(unique_strings), len(field_hash_mapping))
            
            return embeddings, unique_strings, field_hash_mapping
        
        except Exception as e:
            logger.error("Error loading data: %s", str(e))
            raise

    def _create_batches(self, embeddings, unique_strings, field_hash_mapping):
        """
        Create batches of objects to index.
        
        Args:
            embeddings (dict): Dictionary of hash -> embedding vector
            unique_strings (dict): Dictionary of hash -> string value
            field_hash_mapping (dict): Dictionary of hash -> {field -> count}
            
        Returns:
            list: List of batches, where each batch is a list of objects
        """
        batch_size = self.batch_size
        objects = []
        
        # Prepare objects to index
        for hash_value, vector in embeddings.items():
            # Get string value
            string_value = unique_strings.get(hash_value, "")
            
            # Get field types and counts
            field_types = field_hash_mapping.get(hash_value, {})
            
            # Calculate total frequency
            total_frequency = sum(field_types.values())
            
            # Create an object for each field type
            for field_type, count in field_types.items():
                objects.append({
                    'hash': hash_value,
                    'value': string_value,
                    'field_type': field_type,
                    'frequency': count,
                    'vector': vector
                })
        
        # Create batches
        return [objects[i:i + batch_size] for i in range(0, len(objects), batch_size)]

    def _get_collection_stats(self):
        """
        Get statistics for the Weaviate collection.
        
        Returns:
            dict: Collection statistics
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
        
        Args:
            indexed_hashes (set): Set of indexed hash values
        """
        output_dir = Path(self.config['system']['output_dir'])
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save list of indexed hashes
        with open(output_dir / "indexed_hashes.json", 'w') as f:
            json.dump(list(indexed_hashes), f)
        
        # Save collection statistics
        stats = self._get_collection_stats()
        with open(output_dir / "collection_statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save final checkpoint
        checkpoint_path = Path(self.config['system']['checkpoint_dir']) / "indexing_final.ckpt"
        save_checkpoint({
            'indexed_hashes': list(indexed_hashes)
        }, checkpoint_path)
        
        logger.info("Indexing results saved to %s", output_dir)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def search_by_vector(self, vector, field_type, limit=10):
        """
        Search for similar objects by vector.
        
        Args:
            vector (list): Query vector
            field_type (str): Field type to search
            limit (int, optional): Maximum number of results. Defaults to 10.
            
        Returns:
            list: Search results
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
            
            return results.objects
        
        except Exception as e:
            logger.error("Error searching by vector: %s", str(e))
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def search_by_hash(self, hash_value, field_type=None):
        """
        Search for objects by hash value.
        
        Args:
            hash_value (str): Hash value to search for
            field_type (str, optional): Field type to filter by. Defaults to None.
            
        Returns:
            list: Search results
        """
        try:
            collection = self.client.collections.get(self.collection_name)
            
            # Create filter for hash
            if field_type:
                # Filter by hash and field type
                hash_filter = Filter.by_property("hash").equal(hash_value)
                field_filter = Filter.by_property("field_type").equal(field_type)
                combined_filter = Filter.all_of([hash_filter, field_filter])
                filter_obj = combined_filter
            else:
                # Filter by hash only
                filter_obj = Filter.by_property("hash").equal(hash_value)
            
            # Execute search
            results = collection.query.fetch_objects(
                filters=filter_obj,
                limit=10,
                include_vector=True
            )
            
            return results.objects
        
        except Exception as e:
            logger.error("Error searching by hash: %s", str(e))
            raise

    def __del__(self):
        """
        Cleanup resources when object is garbage collected.
        """
        try:
            # Close Weaviate client if it exists
            if hasattr(self, 'client') and self.client:
                logger.debug("Closing Weaviate client connection")
                self.client.close()  # Properly close the client
                self.client = None   # Then set to None
        except Exception as e:
            logger.error(f"Error closing Weaviate client: {e}")
