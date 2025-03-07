"""
Batch querying module for large-scale entity resolution.
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
import pickle

from src.utils import save_checkpoint, load_checkpoint, Timer, get_memory_usage
from src.mmap_dict import MMapDict

logger = logging.getLogger(__name__)

class QueryEngine:
    """
    Handles retrieval of match candidates using vector similarity for large datasets.
    
    Features:
    - Scalable batch processing for millions of records
    - Blocking strategies for efficient candidate retrieval
    - Robust error handling and retry logic
    - Memory-efficient data structures
    """
    
    def __init__(self, config):
        """
        Initialize the query engine with configuration parameters.
        """
        self.config = config
        
        # Connect to Weaviate
        self.client = self._connect_to_weaviate()
        self.collection_name = config['weaviate']['collection_name']
        
        # Querying parameters
        self.similarity_threshold = config['imputation']['vector_similarity_threshold']
        
        # Initialize candidate pairs storage
        self.candidate_pairs = []
        self.use_mmap = self.config['system']['mode'] == 'prod'
        
        if self.use_mmap:
            self.mmap_dir = Path(self.config['system']['temp_dir']) / "mmap"
            self.mmap_dir.mkdir(exist_ok=True, parents=True)
            self.candidate_pairs_file = self.mmap_dir / "candidate_pairs.pkl"
        
        # Initialize ground truth data
        self.ground_truth = self._load_ground_truth()
        
        logger.info("QueryEngine initialized with similarity threshold: %s", 
                   self.similarity_threshold)

    def execute(self, checkpoint=None):
        """
        Execute candidate retrieval for large datasets.
        """
        # Load checkpoint if provided
        if checkpoint and os.path.exists(checkpoint):
            state = load_checkpoint(checkpoint)
            
            if self.use_mmap and 'candidate_pairs_file' in state:
                # For memory-mapped storage, we only store the file path
                self.candidate_pairs_file = Path(state['candidate_pairs_file'])
                # Count entries for reporting
                if self.candidate_pairs_file.exists():
                    try:
                        with open(self.candidate_pairs_file, 'rb') as f:
                            self.candidate_pairs = []
                            while True:
                                try:
                                    pair = pickle.load(f)
                                    self.candidate_pairs.append(None)  # Just for counting
                                except EOFError:
                                    break
                    except Exception as e:
                        logger.error("Error loading candidate pairs: %s", str(e))
                        self.candidate_pairs = []
            else:
                # For in-memory storage, load the actual pairs
                self.candidate_pairs = state.get('candidate_pairs', [])
            
            processed_records = set(state.get('processed_records', []))
            logger.info("Resumed querying from checkpoint: %s with %d processed records, %d candidate pairs", 
                       checkpoint, len(processed_records), len(self.candidate_pairs))
        else:
            processed_records = set()
            # Clear candidate pairs file if it exists
            if self.use_mmap and self.candidate_pairs_file.exists():
                os.remove(self.candidate_pairs_file)
                self.candidate_pairs = []
        
        # Load record field hashes
        record_field_hashes = self._load_record_field_hashes()
        logger.info("Loaded %d records from record_field_hashes", len(record_field_hashes))
        
        # Get records to process
        records_to_process = {}
        for record_id, field_hashes in record_field_hashes.items():
            # Skip if already processed
            if record_id in processed_records:
                continue
            
            # Check if person field is available
            if 'person' in field_hashes and field_hashes['person'] != 'NULL':
                records_to_process[record_id] = field_hashes
        
        logger.info("Processing %d/%d records after filtering", 
                   len(records_to_process), len(record_field_hashes))
        
        # Log reasons for filtering
        no_person_field = sum(1 for _, fields in record_field_hashes.items() 
                            if 'person' not in fields or fields['person'] == 'NULL')
        logger.info("Records without valid person field: %d", no_person_field)
        
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
            
            # Log batch information
            logger.info("Created %d batches with batch size %d", 
                       len(record_batches), batch_size)
            
            # Determine processing mode based on configuration
            debug_mode = os.environ.get('DEBUG_MODE', '').lower() == 'true'
            
            if debug_mode:
                # Process synchronously for debugging
                all_candidates = []
                logger.info("DEBUG MODE: Processing synchronously")
                
                for batch_idx, batch in enumerate(tqdm(record_batches, desc="Processing batches")):
                    batch_dict = dict(batch)
                    batch_candidates = self._process_batch_sync(batch_dict, self.similarity_threshold)
                    
                    if batch_candidates:
                        if self.use_mmap:
                            # Append to file
                            with open(self.candidate_pairs_file, 'ab') as f:
                                for candidate in batch_candidates:
                                    pickle.dump(candidate, f)
                            # Just count for reporting
                            total_candidates += len(batch_candidates)
                        else:
                            self.candidate_pairs.extend(batch_candidates)
                            total_candidates += len(batch_candidates)
                        
                        # Update processed records
                        batch_records = set(batch_dict.keys())
                        processed_records.update(batch_records)
                    
                    # Save checkpoint periodically
                    if self.config['data']['checkpoints_enabled'] and (batch_idx + 1) % 10 == 0:
                        self._save_checkpoint(processed_records, batch_idx)
                        logger.info("Checkpoint saved at batch %d: %d records, %d candidates", 
                                   batch_idx + 1, len(processed_records), total_candidates)
            else:
                # Use parallel processing
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    batch_results = []
                    
                    # Submit all batches for processing
                    for batch_idx, batch in enumerate(tqdm(record_batches, desc="Submitting batches")):
                        # Convert batch to serializable format
                        batch_dict = {}
                        for record_id, field_hashes in batch:
                            # Ensure all values are simple types
                            serializable_hashes = {}
                            for field, hash_value in field_hashes.items():
                                serializable_hashes[str(field)] = str(hash_value)
                            batch_dict[str(record_id)] = serializable_hashes
                        
                        # Submit batch for processing
                        future = executor.submit(
                            process_batch_worker,
                            batch_dict,
                            self.config,
                            self.similarity_threshold
                        )
                        
                        batch_results.append((future, set(batch_dict.keys())))
                    
                    # Process results as they complete
                    for future, batch_records in tqdm(batch_results, desc="Processing results"):
                        try:
                            batch_candidates = future.result()
                            
                            if batch_candidates:
                                if self.use_mmap:
                                    # Append to file
                                    with open(self.candidate_pairs_file, 'ab') as f:
                                        for candidate in batch_candidates:
                                            pickle.dump(candidate, f)
                                    # Just count for reporting
                                    total_candidates += len(batch_candidates)
                                else:
                                    self.candidate_pairs.extend(batch_candidates)
                                    total_candidates += len(batch_candidates)
                                
                                # Update processed records
                                processed_records.update(batch_records)
                        
                        except Exception as e:
                            logger.error("Error processing batch result: %s", str(e))
                            import traceback
                            logger.error(traceback.format_exc())
        
        # Save final results
        self._save_results()
        
        results = {
            'records_processed': len(processed_records),
            'candidate_pairs': total_candidates,
            'duration': timer.duration
        }
        
        logger.info("Querying completed: %d records, %d candidate pairs, %.2f seconds",
                   len(processed_records), total_candidates, timer.duration)
        
        return results

    def _connect_to_weaviate(self):
        """
        Connect to Weaviate instance with retry logic.
        """
        from tenacity import retry, stop_after_attempt, wait_exponential
        
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        def connect_with_retry():
            try:
                # Connect to Weaviate
                client = weaviate.connect_to_local(
                    host=self.config['weaviate']['host'],
                    port=self.config['weaviate']['port']
                )
                
                # Test connection
                client.is_ready()
                
                return client
            
            except Exception as e:
                logger.error("Error connecting to Weaviate: %s", str(e))
                raise
        
        return connect_with_retry()

    def _load_record_field_hashes(self):
        """
        Load record field hashes from preprocessing results.
        """
        try:
            output_dir = Path(self.config['system']['output_dir'])
            
            # Try to load from record index first (enhanced preprocessor)
            record_index_path = output_dir / "record_index.json"
            if record_index_path.exists():
                with open(record_index_path, 'r') as f:
                    record_index = json.load(f)
                
                location = record_index.get('location')
                
                if location != "in-memory" and os.path.exists(location):
                    # For large datasets, use memory-mapped dictionary
                    return MMapDict(location)
                else:
                    # Fall back to sample file
                    with open(output_dir / "record_field_hashes_sample.json", 'r') as f:
                        return json.load(f)
            else:
                # Fall back to original approach
                with open(output_dir / "record_field_hashes_sample.json", 'r') as f:
                    return json.load(f)
        
        except Exception as e:
            logger.error("Error loading record field hashes: %s", str(e))
            return {}

    def _load_ground_truth(self):
        """
        Load ground truth data for evaluation.
        """
        try:
            ground_truth_file = Path(self.config['data']['ground_truth_file'])
            
            if not ground_truth_file.exists():
                logger.warning("Ground truth file not found: %s", ground_truth_file)
                return {}
            
            ground_truth = {}
            
            with open(ground_truth_file, 'r') as f:
                import csv
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                
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
        Create batches of items in a memory-efficient way.
        """
        return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    def _process_batch_sync(self, batch_records, similarity_threshold):
        """
        Process a batch of records synchronously (for debugging).
        """
        logger.info("Processing %d records synchronously", len(batch_records))
        try:
            # Use the same Weaviate client as the main thread
            collection = self.client.collections.get(self.collection_name)
            candidates = []
            
            for record_id, field_hashes in batch_records.items():
                # Get person hash
                person_hash = field_hashes.get('person')
                
                if not person_hash or person_hash == 'NULL':
                    logger.debug("Skipping record %s: No valid person hash", record_id)
                    continue
                
                # Get person vector
                person_vector = self._get_vector_by_hash(collection, person_hash, 'person')
                
                if not person_vector:
                    logger.debug("Skipping record %s: No person vector found", record_id)
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
                    
                    # Use only primitive data types in the returned dictionary
                    candidates.append({
                        'record1_id': str(record_id),
                        'record1_hash': str(person_hash),
                        'record2_hash': str(similar['hash']),
                        'similarity': float(similar['similarity']),
                        'record2_id': None  # Will be filled later
                    })
            
            return candidates
        
        except Exception as e:
            logger.error("Error in synchronous processing: %s", str(e))
            import traceback
            logger.error(traceback.format_exc())
            return []

    def _get_vector_by_hash(self, collection, hash_value, field_type):
        """
        Get vector for a hash value and field type.
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
        Find similar vectors with proper serialization.
        """
        try:
            # Create filter for field type
            field_filter = Filter.by_property("field_type").equal(field_type)
            
            # Execute search
            results = collection.query.near_vector(
                near_vector=query_vector,
                target_vector=[field_type],
                filters=field_filter,
                limit=limit,
                return_metadata=MetadataQuery(distance=True),
                include_vector=False  # Avoid returning vectors to reduce size
            )
            
            # Process results
            similar_vectors = []
            
            for obj in results.objects:
                # Convert distance to similarity (1 - distance)
                similarity = float(1.0 - obj.metadata.distance)
                
                if similarity >= threshold:
                    # Only use simple Python types
                    similar_vectors.append({
                        'hash': str(obj.properties.get('hash', '')),
                        'value': str(obj.properties.get('value', '')),
                        'similarity': similarity
                    })
            
            return similar_vectors
        
        except Exception as e:
            logger.error("Error finding similar vectors: %s", str(e))
            import traceback
            logger.error(traceback.format_exc())
            return []

    def _save_checkpoint(self, processed_records, batch_idx):
        """
        Save checkpoint for querying progress.
        """
        try:
            checkpoint_dir = Path(self.config['system']['checkpoint_dir'])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = checkpoint_dir / f"querying_{batch_idx}.ckpt"
            
            # For memory-mapped storage, only save the file path
            if self.use_mmap:
                checkpoint_data = {
                    'candidate_pairs_file': str(self.candidate_pairs_file),
                    'processed_records': list(processed_records)
                }
            else:
                checkpoint_data = {
                    'candidate_pairs': self.candidate_pairs,
                    'processed_records': list(processed_records)
                }
            
            # Use pickle for large checkpoint data
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            logger.info("Saved querying checkpoint at %s with %d records", 
                       checkpoint_path, len(processed_records))
        
        except Exception as e:
            logger.error("Error saving checkpoint: %s", str(e))

    def _save_results(self):
        """
        Save querying results.
        """
        output_dir = Path(self.config['system']['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        # Save candidate pairs - handle both in-memory and memory-mapped
        if self.use_mmap and self.candidate_pairs_file.exists():
            # Count candidates
            candidate_count = 0
            candidates_sample = []
            
            with open(self.candidate_pairs_file, 'rb') as f:
                try:
                    while True:
                        pair = pickle.load(f)
                        candidate_count += 1
                        if len(candidates_sample) < 1000:
                            candidates_sample.append(pair)
                except EOFError:
                    pass
            
            # Save sample of candidate pairs
            with open(output_dir / "candidate_pairs_sample.json", 'w') as f:
                json.dump(candidates_sample, f, indent=2)
            
            # Save a reference to the full file
            with open(output_dir / "candidate_pairs_info.json", 'w') as f:
                json.dump({
                    'count': candidate_count,
                    'file': str(self.candidate_pairs_file)
                }, f, indent=2)
        else:
            # Save in-memory candidates directly
            with open(output_dir / "candidate_pairs.json", 'w') as f:
                json.dump(self.candidate_pairs, f, indent=2)
        
        # Save statistics
        count = len(self.candidate_pairs) if not self.use_mmap else candidate_count
        
        stats = {
            'total_candidates': count,
            'unique_record1': count  # Simplified for now
        }
        
        with open(output_dir / "candidate_statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save final checkpoint
        checkpoint_path = Path(self.config['system']['checkpoint_dir']) / "querying_final.ckpt"
        if self.use_mmap:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'candidate_pairs_file': str(self.candidate_pairs_file),
                    'processed_records': []  # Skip for final checkpoint to save space
                }, f)
        else:
            save_checkpoint({
                'candidate_pairs': self.candidate_pairs,
                'processed_records': []  # Skip for final checkpoint to save space
            }, checkpoint_path)
        
        logger.info("Querying results saved to %s", output_dir)

    def get_candidates_for_record(self, record_id):
        """
        Get candidate matches for a record.
        """
        if self.use_mmap and self.candidate_pairs_file.exists():
            # Scan the file for matching candidates
            matching_pairs = []
            
            with open(self.candidate_pairs_file, 'rb') as f:
                try:
                    while True:
                        pair = pickle.load(f)
                        if pair['record1_id'] == record_id:
                            matching_pairs.append(pair)
                except EOFError:
                    pass
            
            return matching_pairs
        else:
            # In-memory lookup
            return [c for c in self.candidate_pairs if c['record1_id'] == record_id]

    def __del__(self):
        """
        Cleanup resources when object is garbage collected.
        """
        try:
            # Close Weaviate client if it exists
            if hasattr(self, 'client') and self.client:
                logger.debug("Closing Weaviate client connection")
                self.client.close()
                self.client = None
        except Exception as e:
            logger.error("Error closing Weaviate client: %s", str(e))


def process_batch_worker(batch_dict, config, similarity_threshold):
    """
    Worker function for parallel batch processing.
    This must be a top-level function to be picklable.
    """
    try:
        # Connect to Weaviate
        client = weaviate.connect_to_local(
            host=config['weaviate']['host'],
            port=config['weaviate']['port']
        )
        
        collection = client.collections.get(config['weaviate']['collection_name'])
        candidates = []
        
        for record_id, field_hashes in batch_dict.items():
            # Get person hash
            person_hash = field_hashes.get('person')
            
            if not person_hash or person_hash == 'NULL':
                continue
            
            # Get person vector
            # Create filter for hash and field type
            hash_filter = Filter.by_property("hash").equal(person_hash)
            field_filter = Filter.by_property("field_type").equal('person')
            combined_filter = Filter.all_of([hash_filter, field_filter])
            
            # Execute search
            results = collection.query.fetch_objects(
                filters=combined_filter,
                limit=1,
                include_vector=True
            )
            
            if not results.objects:
                continue
            
            # Extract vector
            person_vector = results.objects[0].vector.get('person')
            
            if not person_vector:
                continue
            
            # Find similar person vectors
            # Create filter for field type
            field_filter = Filter.by_property("field_type").equal('person')
            
            # Execute search
            
            similar_results = collection.query.near_vector(
                near_vector=person_vector,
                filters=field_filter,
                limit=100,
                return_metadata=MetadataQuery(distance=True),
                target_vector=['person'],
                include_vector=False  # Avoid returning vectors to reduce size
            )
            
            # Process results
            for obj in similar_results.objects:
                # Convert distance to similarity (1 - distance)
                similarity = float(1.0 - obj.metadata.distance)
                
                if similarity >= similarity_threshold:
                    similar_hash = obj.properties.get('hash', '')
                    
                    # Ensure different records
                    if similar_hash == person_hash:
                        continue
                    
                    # Use only primitive data types in the returned dictionary
                    candidates.append({
                        'record1_id': str(record_id),
                        'record1_hash': str(person_hash),
                        'record2_hash': str(similar_hash),
                        'similarity': similarity,
                        'record2_id': None  # Will be filled later
                    })
        
        # Close client
        client.close()
        
        return candidates
    
    except Exception as e:
        import traceback
        error_str = f"Error in worker: {str(e)}\n{traceback.format_exc()}"
        print(error_str)  # Print for parallel debugging
        return []
