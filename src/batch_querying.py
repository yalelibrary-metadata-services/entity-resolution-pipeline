"""
Batch querying module for entity resolution.

This module provides the QueryEngine class, which handles retrieval of
match candidates using vector similarity.
"""

import os
import logging
import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import weaviate
from weaviate.classes.query import Filter, MetadataQuery

from src.utils import save_checkpoint, load_checkpoint, Timer, get_memory_usage

logger = logging.getLogger(__name__)

class QueryEngine:
    """
    Handles retrieval of match candidates using vector similarity.
    
    Features:
    - Uses vector similarity to identify candidate matches
    - Batched and parallel processing for efficiency
    - Configurable similarity threshold and candidate count
    - Checkpointing for resuming interrupted querying
    """
    
    def __init__(self, config):
        """
        Initialize the query engine with configuration parameters.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        
        # Connect to Weaviate
        self.client = self._connect_to_weaviate()
        self.collection_name = config['weaviate']['collection_name']
        
        # Querying parameters
        self.similarity_threshold = config['imputation']['vector_similarity_threshold']
        
        # Initialize candidate pairs
        self.candidate_pairs = []
        
        # Initialize ground truth data
        self.ground_truth = self._load_ground_truth()
        
        logger.info("QueryEngine initialized with similarity threshold: %s", 
                   self.similarity_threshold)

    def execute(self, checkpoint=None):
        """
        Execute candidate retrieval.
        
        Args:
            checkpoint (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Querying results
        """
        # Load checkpoint if provided
        if checkpoint and os.path.exists(checkpoint):
            state = load_checkpoint(checkpoint)
            self.candidate_pairs = state.get('candidate_pairs', [])
            processed_records = set(state.get('processed_records', []))
            logger.info("Resumed querying from checkpoint: %s", checkpoint)
        else:
            processed_records = set()
        
        # Load record field hashes
        record_field_hashes = self._load_record_field_hashes()
        
        # Get records to process
        records_to_process = {}
        for record_id, field_hashes in record_field_hashes.items():
            # Skip if already processed
            if record_id in processed_records:
                continue
            
            # Check if person field is available
            if 'person' in field_hashes and field_hashes['person'] != 'NULL':
                records_to_process[record_id] = field_hashes
        
        logger.info("Processing %d/%d records", len(records_to_process), len(record_field_hashes))
        
        if self.config['system']['mode'] == 'dev':
            # In dev mode, limit the number of records to process
            dev_sample_size = min(self.config['system']['dev_sample_size'], len(records_to_process))
            record_sample = list(records_to_process.keys())[:dev_sample_size]
            records_to_process = {r: records_to_process[r] for r in record_sample}
            logger.info("Dev mode: limited to %d records", len(records_to_process))
        
        # Process records in batches
        total_candidates = 0
        batch_size = self.config['system']['batch_size']
        max_workers = self.config['system']['max_workers']
        
        with Timer() as timer:
            # Create batches of records
            record_batches = self._create_batches(list(records_to_process.items()), batch_size)
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                batch_results = []
                
                for batch_idx, batch in enumerate(tqdm(record_batches, desc="Submitting batches")):
                    # Convert batch to dictionary for processing
                    batch_dict = dict(batch)
                    
                    # Submit batch for processing
                    future = executor.submit(
                        self._process_batch,
                        batch_dict,
                        self.similarity_threshold
                    )
                    
                    batch_results.append(future)
                
                # Process results
                for future in tqdm(batch_results, desc="Processing results"):
                    try:
                        batch_candidates = future.result()
                        
                        if batch_candidates:
                            self.candidate_pairs.extend(batch_candidates)
                            total_candidates += len(batch_candidates)
                            
                            # Update processed records
                            batch_records = set()
                            for candidate in batch_candidates:
                                batch_records.add(candidate['record1_id'])
                            
                            processed_records.update(batch_records)
                    
                    except Exception as e:
                        logger.error("Error processing batch result: %s", str(e))
                
                # Save checkpoint periodically
                if self.config['data']['checkpoints_enabled'] and len(processed_records) % 1000 == 0:
                    checkpoint_path = Path(self.config['system']['checkpoint_dir']) / f"querying_{len(processed_records)}.ckpt"
                    save_checkpoint({
                        'candidate_pairs': self.candidate_pairs,
                        'processed_records': list(processed_records)
                    }, checkpoint_path)
                
                logger.info("Processed %d records, found %d candidate pairs", 
                           len(processed_records), total_candidates)
                logger.info("Memory usage: %.2f GB", get_memory_usage())
        
        # Save final results
        self._save_results()
        
        results = {
            'records_processed': len(processed_records),
            'candidate_pairs': len(self.candidate_pairs),
            'duration': timer.duration
        }
        
        logger.info("Querying completed: %d records, %d candidate pairs, %.2f seconds",
                   len(processed_records), len(self.candidate_pairs), timer.duration)
        
        return results

    def _connect_to_weaviate(self):
        """
        Connect to Weaviate instance.
        
        Returns:
            weaviate.Client: Weaviate client
        """
        try:
            # Connect to Weaviate
            client = weaviate.connect_to_local(
                # host=self.config['weaviate']['host'],
                # port=self.config['weaviate']['port'],
                # grpc_port=None,
                # headers={}
            )
            
            # Test connection
            client.is_ready()
            
            return client
        
        except Exception as e:
            logger.error("Error connecting to Weaviate: %s", str(e))
            raise

    def _load_record_field_hashes(self):
        """
        Load record field hashes from preprocessing results.
        
        Returns:
            dict: Dictionary of record ID -> field hashes
        """
        try:
            output_dir = Path(self.config['system']['output_dir'])
            with open(output_dir / "record_field_hashes_sample.json", 'r') as f:
                record_field_hashes = json.load(f)
            
            return record_field_hashes
        
        except Exception as e:
            logger.error("Error loading record field hashes: %s", str(e))
            return {}

    def _load_ground_truth(self):
        """
        Load ground truth data for evaluation.
        
        Returns:
            dict: Dictionary of record pair -> match status
        """
        try:
            ground_truth_file = Path(self.config['data']['ground_truth_file'])
            
            if not ground_truth_file.exists():
                logger.warning("Ground truth file not found: %s", ground_truth_file)
                return {}
            
            ground_truth = {}
            
            with open(ground_truth_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                
                for row in reader:
                    if len(row) >= 3:
                        left_id, right_id, match = row
                        
                        # Ensure consistent ordering of IDs
                        if left_id > right_id:
                            left_id, right_id = right_id, left_id
                        
                        pair_key = f"{left_id}|{right_id}"
                        ground_truth[pair_key] = match.lower() == 'true'
            
            logger.info("Loaded %d ground truth pairs", len(ground_truth))
            return ground_truth
        
        except Exception as e:
            logger.error("Error loading ground truth: %s", str(e))
            return {}

    def _create_batches(self, items, batch_size):
        """
        Create batches of items.
        
        Args:
            items (list): List of items to batch
            batch_size (int): Batch size
            
        Returns:
            list: List of batches
        """
        return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    def _process_batch(self, batch_records, similarity_threshold):
        """
        Process a batch of records to find candidate matches.
        
        Args:
            batch_records (dict): Dictionary of record ID -> field hashes
            similarity_threshold (float): Similarity threshold for candidates
            
        Returns:
            list: List of candidate pairs
        """
        try:
            # Connect to Weaviate (separate connection for each process)
            client = weaviate.connect_to_local(
                host=self.config['weaviate']['host'],
                port=self.config['weaviate']['port']
            )
            
            collection = client.collections.get(self.collection_name)
            candidates = []
            
            for record_id, field_hashes in batch_records.items():
                # Get person hash
                person_hash = field_hashes.get('person')
                
                if not person_hash or person_hash == 'NULL':
                    continue
                
                # Get person vector
                person_vector = self._get_vector_by_hash(collection, person_hash, 'person')
                
                if not person_vector:
                    continue
                
                # Find similar person vectors
                similar_persons = self._find_similar_vectors(
                    collection, 
                    person_vector, 
                    'person', 
                    limit=100,
                    threshold=similarity_threshold
                )
                
                # Create candidate pairs
                for similar in similar_persons:
                    # Ensure different records
                    if similar['hash'] == person_hash:
                        continue
                    
                    # Add candidate pair
                    candidates.append({
                        'record1_id': record_id,
                        'record1_hash': person_hash,
                        'record2_hash': similar['hash'],
                        'similarity': similar['similarity'],
                        'record2_id': None  # Will be filled later
                    })
            
            return candidates
        
        except Exception as e:
            logger.error("Error processing batch: %s", str(e))
            return []

    def _get_vector_by_hash(self, collection, hash_value, field_type):
        """
        Get vector for a hash value and field type.
        
        Args:
            collection: Weaviate collection
            hash_value (str): Hash value
            field_type (str): Field type
            
        Returns:
            list: Vector or None if not found
        """
        try:
            # Create filter for hash and field type
            hash_filter = Filter.by_property("hash").equal(hash_value)
            field_filter = Filter.by_property("field_type").equal(field_type)
            combined_filter = Filter.all_of([hash_filter, field_filter])
            
            # Execute search
            results = collection.query.fetch_objects(
                filters=combined_filter,
                limit=1,
                include_vector=True
            )
            
            if results.objects:
                # Extract vector
                return results.objects[0].vector.get(field_type)
            
            return None
        
        except Exception as e:
            logger.error("Error getting vector for hash %s, field %s: %s", 
                        hash_value, field_type, str(e))
            return None

    def _find_similar_vectors(self, collection, query_vector, field_type, limit=100, threshold=0.7):
        """
        Find similar vectors.
        
        Args:
            collection: Weaviate collection
            query_vector (list): Query vector
            field_type (str): Field type
            limit (int): Maximum number of results
            threshold (float): Similarity threshold
            
        Returns:
            list: List of similar vectors
        """
        try:
            # Create filter for field type
            field_filter = Filter.by_property("field_type").equal(field_type)
            
            # Execute search
            results = collection.query.near_vector(
                near_vector={field_type: query_vector},
                filters=field_filter,
                limit=limit,
                return_metadata=MetadataQuery(distance=True),
                include_vector=True
            )
            
            # Process results
            similar_vectors = []
            
            for obj in results.objects:
                # Convert distance to similarity (1 - distance)
                similarity = 1.0 - obj.metadata.distance
                
                if similarity >= threshold:
                    similar_vectors.append({
                        'hash': obj.properties['hash'],
                        'value': obj.properties['value'],
                        'similarity': similarity
                    })
            
            return similar_vectors
        
        except Exception as e:
            logger.error("Error finding similar vectors: %s", str(e))
            return []

    def _save_results(self):
        """
        Save querying results.
        """
        output_dir = Path(self.config['system']['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        # Save candidate pairs
        with open(output_dir / "candidate_pairs.json", 'w') as f:
            json.dump(self.candidate_pairs, f, indent=2)
        
        # Save statistics
        stats = {
            'total_candidates': len(self.candidate_pairs),
            'unique_record1': len(set(c['record1_id'] for c in self.candidate_pairs)),
            'similarity_distribution': {
                '0.7-0.8': sum(1 for c in self.candidate_pairs if 0.7 <= c['similarity'] < 0.8),
                '0.8-0.9': sum(1 for c in self.candidate_pairs if 0.8 <= c['similarity'] < 0.9),
                '0.9-1.0': sum(1 for c in self.candidate_pairs if 0.9 <= c['similarity'] <= 1.0)
            }
        }
        
        with open(output_dir / "candidate_statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save final checkpoint
        checkpoint_path = Path(self.config['system']['checkpoint_dir']) / "querying_final.ckpt"
        save_checkpoint({
            'candidate_pairs': self.candidate_pairs,
            'processed_records': list(set(c['record1_id'] for c in self.candidate_pairs))
        }, checkpoint_path)
        
        logger.info("Querying results saved to %s", output_dir)

    def get_candidates_for_record(self, record_id):
        """
        Get candidate matches for a record.
        
        Args:
            record_id (str): Record ID
            
        Returns:
            list: List of candidate pairs
        """
        return [c for c in self.candidate_pairs if c['record1_id'] == record_id]

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