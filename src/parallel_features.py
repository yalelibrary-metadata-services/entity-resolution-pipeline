"""
Parallel feature engineering module for entity resolution.

This module provides the FeatureEngineer class, which handles construction
of feature vectors for record pairs.
"""

import os
import logging
import json
import csv
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import weaviate
from weaviate.classes.query import Filter
from scipy.spatial.distance import cosine
from rapidfuzz import fuzz
import Levenshtein
import re

from src.utils import save_checkpoint, load_checkpoint, Timer, get_memory_usage, update_stage_metrics
from src.birth_death_regexes import BirthDeathYearExtractor

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Handles construction of feature vectors for record pairs.
    
    Features:
    - Computes vector cosine similarity and additional similarity metrics
    - Constructs feature vectors for classifier training and prediction
    - Implements sophisticated interaction features
    - Supports batched and parallel processing
    """
    
    def __init__(self, config):
        """
        Initialize the feature engineer with configuration parameters.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        
        # Connect to Weaviate
        self.client = self._connect_to_weaviate()
        self.collection_name = config['weaviate']['collection_name']
        
        # Initialize birth/death year parser
        self.birth_death_parser = BirthDeathYearExtractor()
        
        # Initialize feature vectors and labels
        self.feature_vectors = []
        self.labels = []
        self.feature_names = []
        
        # Initialize prefiltered pairs
        self.prefiltered_true = []
        self.prefiltered_false = []
        
        # Load ground truth data
        self.ground_truth = self._load_ground_truth()
        
        # Initialize unique strings
        self.unique_strings = self._load_unique_strings()
        
        # Initialize record field hashes
        self.record_field_hashes = self._load_record_field_hashes()
        
        # Initialize record-hash mapping
        self.record_hash_mapping = self._build_record_hash_mapping()
        
        # Initialize feature configuration
        self.initialize_feature_config()
        
        logger.info("FeatureEngineer initialized with %d feature types", 
                   len(self.feature_names))

    def execute(self, checkpoint=None):
        """
        Execute feature engineering.
        
        Args:
            checkpoint (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Feature engineering results
        """
        # Load checkpoint if provided
        if checkpoint and os.path.exists(checkpoint):
            state = load_checkpoint(checkpoint)
            self.feature_vectors = state.get('feature_vectors', [])
            self.labels = state.get('labels', [])
            self.feature_names = state.get('feature_names', [])
            self.prefiltered_true = state.get('prefiltered_true', [])
            self.prefiltered_false = state.get('prefiltered_false', [])
            processed_pairs = set(state.get('processed_pairs', []))
            logger.info("Resumed feature engineering from checkpoint: %s", checkpoint)
        else:
            processed_pairs = set()
        
        # Get candidate pairs
        candidate_pairs = self._load_candidate_pairs()
        
        # Filter pairs that haven't been processed yet
        pairs_to_process = []
        for pair in candidate_pairs:
            pair_id = self._get_pair_id(pair['record1_id'], pair.get('record2_id', ''))
            
            if pair_id not in processed_pairs:
                pairs_to_process.append(pair)
        
        logger.info("Processing %d/%d candidate pairs", 
                   len(pairs_to_process), len(candidate_pairs))
        
        if self.config['system']['mode'] == 'dev':
            # In dev mode, limit the number of pairs to process
            dev_sample_size = min(self.config['system']['dev_sample_size'], len(pairs_to_process))
            pairs_to_process = pairs_to_process[:dev_sample_size]
            logger.info("Dev mode: limited to %d pairs", len(pairs_to_process))
        
        # Process pairs in batches
        batch_size = self.config['system']['batch_size']
        max_workers = self.config['system']['max_workers']
        
        with Timer() as timer:
            # Create batches of pairs
            pair_batches = self._create_batches(pairs_to_process, batch_size)
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                batch_results = []
                
                for batch_idx, batch in enumerate(tqdm(pair_batches, desc="Submitting batches")):
                    # Submit batch for processing
                    future = executor.submit(
                        self._process_batch,
                        batch,
                        self.feature_names,
                        self.record_field_hashes,
                        self.unique_strings
                    )
                    
                    batch_results.append(future)
                
                # Process results
                for future in tqdm(batch_results, desc="Processing results"):
                    try:
                        batch_results = future.result()
                        
                        if batch_results:
                            # Extract results
                            batch_vectors = batch_results['vectors']
                            batch_labels = batch_results['labels']
                            batch_prefiltered_true = batch_results['prefiltered_true']
                            batch_prefiltered_false = batch_results['prefiltered_false']
                            batch_pair_ids = batch_results['pair_ids']
                            
                            # Validate feature vectors
                            if self._validate_feature_vectors(batch_vectors):
                                # Update feature vectors and labels
                                self.feature_vectors.extend(batch_vectors)
                                self.labels.extend(batch_labels)
                                self.prefiltered_true.extend(batch_prefiltered_true)
                                self.prefiltered_false.extend(batch_prefiltered_false)
                                
                                # Update processed pairs
                                processed_pairs.update(batch_pair_ids)
                    
                    except Exception as e:
                        logger.error("Error processing batch result: %s", str(e))
                
                # Save checkpoint periodically
                if self.config['data']['checkpoints_enabled'] and len(processed_pairs) % 1000 == 0:
                    checkpoint_path = Path(self.config['system']['checkpoint_dir']) / f"features_{len(processed_pairs)}.ckpt"
                    save_checkpoint({
                        'feature_vectors': self.feature_vectors,
                        'labels': self.labels,
                        'feature_names': self.feature_names,
                        'prefiltered_true': self.prefiltered_true,
                        'prefiltered_false': self.prefiltered_false,
                        'processed_pairs': list(processed_pairs)
                    }, checkpoint_path)
                
                logger.info("Processed %d pairs, created %d feature vectors", 
                           len(processed_pairs), len(self.feature_vectors))
                logger.info("Prefiltered: %d true, %d false", 
                           len(self.prefiltered_true), len(self.prefiltered_false))
                logger.info("Memory usage: %.2f GB", get_memory_usage())
        
        # Save final results
        self._save_results()
        
        results = {
            'pairs_processed': len(processed_pairs),
            'feature_vectors': len(self.feature_vectors),
            'prefiltered_true': len(self.prefiltered_true),
            'prefiltered_false': len(self.prefiltered_false),
            'feature_count': len(self.feature_names),
            'duration': timer.duration
        }
        
        # Update monitoring metrics
        update_stage_metrics('features', results)
        
        logger.info("Feature engineering completed: %d pairs, %d vectors, %.2f seconds",
                   len(processed_pairs), len(self.feature_vectors), timer.duration)
        
        return results

    def initialize_feature_config(self):
        """
        Initialize feature configuration.
        """
        # Start with empty feature names
        self.feature_names = []
        
        # Add cosine similarity features
        cosine_similarities = self.config['features']['cosine_similarities']
        for field in cosine_similarities:
            self.feature_names.append(f"{field}_cosine")
        
        # Add string similarity features
        string_similarities = self.config['features']['string_similarities']
        for sim_config in string_similarities:
            field = sim_config['field']
            metrics = sim_config['metrics']
            
            for metric in metrics:
                self.feature_names.append(f"{field}_{metric}")
        
        # Add harmonic mean features
        harmonic_means = self.config['features']['harmonic_means']
        for field_pair in harmonic_means:
            field1, field2 = field_pair
            self.feature_names.append(f"{field1}_{field2}_harmonic")
        
        # Add additional interaction features
        additional_interactions = self.config['features']['additional_interactions']
        for interaction in additional_interactions:
            interaction_type = interaction['type']
            fields = interaction['fields']
            self.feature_names.append(f"{fields[0]}_{fields[1]}_{interaction_type}")
        
        # Add birth/death year match feature
        self.feature_names.append("birth_year_match")
        self.feature_names.append("death_year_match")
        self.feature_names.append("has_birth_death_years")
        
        logger.info("Initialized %d feature names", len(self.feature_names))

    def _connect_to_weaviate(self):
        """
        Connect to Weaviate instance with retry logic.
        
        Returns:
            weaviate.Client: Weaviate client
        """
        from tenacity import retry, stop_after_attempt, wait_exponential

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        def connect_with_retry():
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

        return connect_with_retry()

    def _load_candidate_pairs(self):
        """
        Load candidate pairs.
        
        Returns:
            list: List of candidate pairs
        """
        try:
            output_dir = Path(self.config['system']['output_dir'])
            with open(output_dir / "candidate_pairs.json", 'r') as f:
                candidate_pairs = json.load(f)
            
            return candidate_pairs
        
        except Exception as e:
            logger.error("Error loading candidate pairs: %s", str(e))
            return []

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
                next(reader, None)  # Skip header if present
                
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

    def _load_unique_strings(self):
        """
        Load unique strings from preprocessing results.
        
        Returns:
            dict: Dictionary of hash -> string value
        """
        try:
            output_dir = Path(self.config['system']['output_dir'])
            with open(output_dir / "unique_strings_sample.json", 'r') as f:
                unique_strings = json.load(f)
            
            return unique_strings
        
        except Exception as e:
            logger.error("Error loading unique strings: %s", str(e))
            return {}

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

    def _build_record_hash_mapping(self):
        """
        Build mapping from hash to record IDs.
        
        Returns:
            dict: Dictionary of hash -> list of record IDs
        """
        mapping = {}
        
        for record_id, field_hashes in self.record_field_hashes.items():
            for field, hash_value in field_hashes.items():
                if hash_value not in mapping:
                    mapping[hash_value] = []
                
                mapping[hash_value].append(record_id)
        
        return mapping

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

    def _get_pair_id(self, record1_id, record2_id):
        """
        Get unique identifier for a record pair.
        
        Args:
            record1_id (str): First record ID
            record2_id (str): Second record ID
            
        Returns:
            str: Pair identifier
        """
        # Ensure consistent ordering
        if record1_id > record2_id:
            record1_id, record2_id = record2_id, record1_id
        
        return f"{record1_id}|{record2_id}"

    def _validate_feature_vectors(self, vectors):
        """
        Validate feature vectors for consistency.
        
        Args:
            vectors (list): List of feature vectors
            
        Returns:
            bool: True if vectors are valid, False otherwise
        """
        if not vectors:
            return False
        
        # Convert to numpy array if needed
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)
        
        # Check for empty feature vectors
        if len(vectors) == 0:
            logger.warning("Empty feature vector set")
            return False
        
        # Check for NaN or infinity values
        if np.isnan(vectors).any() or np.isinf(vectors).any():
            logger.error("Feature vectors contain NaN or infinity values")
            return False
        
        # Check for consistent dimensions
        expected_dims = len(self.feature_names)
        if vectors.shape[1] != expected_dims:
            logger.error("Feature vector dimension mismatch: %d vs expected %d", 
                         vectors.shape[1], expected_dims)
            return False
        
        return True

    def _process_batch(self, batch_pairs, feature_names, record_field_hashes, unique_strings):
        """
        Process a batch of candidate pairs.
        
        Args:
            batch_pairs (list): List of candidate pairs
            feature_names (list): List of feature names
            record_field_hashes (dict): Dictionary of record ID -> field hashes
            unique_strings (dict): Dictionary of hash -> string value
            
        Returns:
            dict: Batch processing results
        """
        try:
            # Connect to Weaviate (separate connection for each process)
            client = weaviate.connect_to_local(
                host=self.config['weaviate']['host'],
                port=self.config['weaviate']['port']
            )
            
            collection = client.collections.get(self.config['weaviate']['collection_name'])
            
            # Initialize results
            vectors = []
            labels = []
            prefiltered_true = []
            prefiltered_false = []
            pair_ids = []
            
            # Initialize birth/death parser
            birth_death_parser = BirthDeathYearExtractor()
            
            for pair in batch_pairs:
                record1_id = pair['record1_id']
                record1_hash = pair['record1_hash']
                
                # Find record2_id using hash
                record2_hash = pair['record2_hash']
                record2_candidates = self._find_records_by_hash(record2_hash, 'person', record_field_hashes)
                
                if not record2_candidates:
                    continue
                
                # Use first candidate as record2_id
                record2_id = record2_candidates[0]
                
                # Skip self-matches
                if record1_id == record2_id:
                    continue
                
                # Get pair ID
                pair_id = self._get_pair_id(record1_id, record2_id)
                
                # Get field hashes for both records
                record1_fields = record_field_hashes.get(record1_id, {})
                record2_fields = record_field_hashes.get(record2_id, {})
                
                # Skip if missing essential fields
                if not record1_fields or not record2_fields:
                    continue
                
                # Apply prefilters
                prefilter_result = self._apply_prefilters(
                    record1_id, record2_id,
                    record1_fields, record2_fields,
                    unique_strings, collection,
                    birth_death_parser
                )
                
                if prefilter_result == 'true':
                    # Automatically classified as true match
                    prefiltered_true.append({
                        'record1_id': record1_id,
                        'record2_id': record2_id,
                        'reason': 'prefilter'
                    })
                    continue
                
                elif prefilter_result == 'false':
                    # Automatically classified as false match
                    prefiltered_false.append({
                        'record1_id': record1_id,
                        'record2_id': record2_id,
                        'reason': 'prefilter'
                    })
                    continue
                
                # Get ground truth label if available
                label = None
                if pair_id in self.ground_truth:
                    label = 1 if self.ground_truth[pair_id] else 0
                
                # Skip if no label (for training/testing)
                if label is None:
                    continue
                
                # Construct feature vector
                feature_vector = self._construct_feature_vector(
                    record1_id, record2_id,
                    record1_fields, record2_fields,
                    unique_strings, collection,
                    feature_names, birth_death_parser
                )
                
                if feature_vector:
                    vectors.append(feature_vector)
                    labels.append(label)
                    pair_ids.append(pair_id)
            
            return {
                'vectors': vectors,
                'labels': labels,
                'prefiltered_true': prefiltered_true,
                'prefiltered_false': prefiltered_false,
                'pair_ids': pair_ids
            }
        
        except Exception as e:
            logger.error("Error processing batch: %s", str(e))
            return None

    def _find_records_by_hash(self, hash_value, field_type, record_field_hashes):
        """
        Find records containing a hash value for a specific field.
        
        Args:
            hash_value (str): Hash value
            field_type (str): Field type
            record_field_hashes (dict): Dictionary of record ID -> field hashes
            
        Returns:
            list: List of record IDs
        """
        matching_records = []
        
        for record_id, field_hashes in record_field_hashes.items():
            if field_type in field_hashes and field_hashes[field_type] == hash_value:
                matching_records.append(record_id)
        
        return matching_records

    def _apply_prefilters(self, record1_id, record2_id, record1_fields, record2_fields, 
                         unique_strings, collection, birth_death_parser):
        """
        Apply prefilters to automatically classify candidate pairs.
        
        Args:
            record1_id (str): First record ID
            record2_id (str): Second record ID
            record1_fields (dict): Field hashes for first record
            record2_fields (dict): Field hashes for second record
            unique_strings (dict): Dictionary of hash -> string value
            collection: Weaviate collection
            birth_death_parser: Birth/death year parser
            
        Returns:
            str: Prefilter result ('true', 'false', or None)
        """
        # Check exact name with birth/death years prefilter
        if self.config['features']['prefilters']['exact_name_birth_death_prefilter']:
            if 'person' in record1_fields and 'person' in record2_fields:
                person1_hash = record1_fields['person']
                person2_hash = record2_fields['person']
                
                if person1_hash == person2_hash and person1_hash in unique_strings:
                    person_name = unique_strings[person1_hash]
                    
                    # Check if name contains birth/death years
                    birth1, death1 = birth_death_parser.parse(person_name)
                    
                    if birth1 is not None or death1 is not None:
                        return 'true'
        
        # Check composite cosine prefilter
        if self.config['features']['prefilters']['composite_cosine_prefilter']['enabled']:
            threshold = self.config['features']['prefilters']['composite_cosine_prefilter']['threshold']
            
            if 'composite' in record1_fields and 'composite' in record2_fields:
                composite1_hash = record1_fields['composite']
                composite2_hash = record2_fields['composite']
                
                if composite1_hash in unique_strings and composite2_hash in unique_strings:
                    composite1_vector = self._get_vector_by_hash(collection, composite1_hash, 'composite')
                    composite2_vector = self._get_vector_by_hash(collection, composite2_hash, 'composite')
                    
                    if composite1_vector is not None and composite2_vector is not None:
                        similarity = 1.0 - cosine(composite1_vector, composite2_vector)
                        
                        if similarity >= threshold:
                            return 'true'
        
        # Check person cosine prefilter
        if self.config['features']['prefilters']['person_cosine_prefilter']['enabled']:
            threshold = self.config['features']['prefilters']['person_cosine_prefilter']['threshold']
            
            if 'person' in record1_fields and 'person' in record2_fields:
                person1_hash = record1_fields['person']
                person2_hash = record2_fields['person']
                
                if person1_hash in unique_strings and person2_hash in unique_strings:
                    person1_vector = self._get_vector_by_hash(collection, person1_hash, 'person')
                    person2_vector = self._get_vector_by_hash(collection, person2_hash, 'person')
                    
                    if person1_vector is not None and person2_vector is not None:
                        similarity = 1.0 - cosine(person1_vector, person2_vector)
                        
                        if similarity < threshold:
                            return 'false'
        
        # No prefilter matched
        return None

    def _construct_feature_vector(self, record1_id, record2_id, record1_fields, record2_fields, 
                                 unique_strings, collection, feature_names, birth_death_parser):
        """
        Construct feature vector for a record pair.
        
        Args:
            record1_id (str): First record ID
            record2_id (str): Second record ID
            record1_fields (dict): Field hashes for first record
            record2_fields (dict): Field hashes for second record
            unique_strings (dict): Dictionary of hash -> string value
            collection: Weaviate collection
            feature_names (list): List of feature names
            birth_death_parser: Birth/death year parser
            
        Returns:
            list: Feature vector
        """
        # Initialize feature vector with zeros
        feature_vector = [0.0] * len(feature_names)
        feature_computed = [False] * len(feature_names)
        
        # Get vectors for fields
        field_vectors = {}
        
        for field in ['composite', 'person', 'title', 'provision', 'subjects']:
            # Get field hashes
            field1_hash = record1_fields.get(field, 'NULL')
            field2_hash = record2_fields.get(field, 'NULL')
            
            # Skip if either field is missing
            if field1_hash == 'NULL' or field2_hash == 'NULL':
                continue
            
            # Get vectors
            field1_vector = self._get_vector_by_hash(collection, field1_hash, field)
            field2_vector = self._get_vector_by_hash(collection, field2_hash, field)
            
            if field1_vector is not None and field2_vector is not None:
                field_vectors[field] = (field1_vector, field2_vector)
        
        # Compute cosine similarity features
        for field in ['composite', 'person', 'title', 'provision', 'subjects']:
            feature_index = feature_names.index(f"{field}_cosine") if f"{field}_cosine" in feature_names else -1
            
            if feature_index >= 0 and field in field_vectors:
                vector1, vector2 = field_vectors[field]
                similarity = 1.0 - cosine(vector1, vector2)
                feature_vector[feature_index] = similarity
                feature_computed[feature_index] = True
        
        # Compute string similarity features
        for field in ['person']:
            for metric in ['levenshtein', 'jaro_winkler']:
                feature_index = feature_names.index(f"{field}_{metric}") if f"{field}_{metric}" in feature_names else -1
                
                if feature_index >= 0:
                    field1_hash = record1_fields.get(field, 'NULL')
                    field2_hash = record2_fields.get(field, 'NULL')
                    
                    if field1_hash != 'NULL' and field2_hash != 'NULL' and field1_hash in unique_strings and field2_hash in unique_strings:
                        string1 = unique_strings[field1_hash]
                        string2 = unique_strings[field2_hash]
                        
                        if metric == 'levenshtein':
                            # Normalize by maximum length
                            max_len = max(len(string1), len(string2))
                            if max_len > 0:
                                distance = Levenshtein.distance(string1, string2)
                                similarity = 1.0 - (distance / max_len)
                            else:
                                similarity = 1.0
                        
                        elif metric == 'jaro_winkler':
                            similarity = fuzz.token_sort_ratio(string1, string2) / 100.0
                        
                        feature_vector[feature_index] = similarity
                        feature_computed[feature_index] = True
        
        # Compute harmonic mean features
        for field_pair in [['person', 'title'], ['person', 'provision'], ['person', 'subjects'],
                          ['title', 'subjects'], ['title', 'provision'], ['provision', 'subjects']]:
            field1, field2 = field_pair
            feature_index = feature_names.index(f"{field1}_{field2}_harmonic") if f"{field1}_{field2}_harmonic" in feature_names else -1
            
            if feature_index >= 0:
                cosine1_index = feature_names.index(f"{field1}_cosine") if f"{field1}_cosine" in feature_names else -1
                cosine2_index = feature_names.index(f"{field2}_cosine") if f"{field2}_cosine" in feature_names else -1
                
                if cosine1_index >= 0 and cosine2_index >= 0 and feature_computed[cosine1_index] and feature_computed[cosine2_index]:
                    cosine1 = feature_vector[cosine1_index]
                    cosine2 = feature_vector[cosine2_index]
                    
                    # Compute harmonic mean
                    if cosine1 > 0 and cosine2 > 0:
                        harmonic_mean = 2 * (cosine1 * cosine2) / (cosine1 + cosine2)
                    else:
                        harmonic_mean = 0.0
                    
                    feature_vector[feature_index] = harmonic_mean
                    feature_computed[feature_index] = True
        
        # Compute additional interaction features
        # Product features
        for field_pair in [['person', 'subjects']]:
            field1, field2 = field_pair
            feature_index = feature_names.index(f"{field1}_{field2}_product") if f"{field1}_{field2}_product" in feature_names else -1
            
            if feature_index >= 0:
                cosine1_index = feature_names.index(f"{field1}_cosine") if f"{field1}_cosine" in feature_names else -1
                cosine2_index = feature_names.index(f"{field2}_cosine") if f"{field2}_cosine" in feature_names else -1
                
                if cosine1_index >= 0 and cosine2_index >= 0 and feature_computed[cosine1_index] and feature_computed[cosine2_index]:
                    cosine1 = feature_vector[cosine1_index]
                    cosine2 = feature_vector[cosine2_index]
                    
                    # Compute product
                    product = cosine1 * cosine2
                    
                    feature_vector[feature_index] = product
                    feature_computed[feature_index] = True
        
        # Ratio features
        for field_pair in [['composite', 'subjects']]:
            field1, field2 = field_pair
            feature_index = feature_names.index(f"{field1}_{field2}_ratio") if f"{field1}_{field2}_ratio" in feature_names else -1
            
            if feature_index >= 0:
                cosine1_index = feature_names.index(f"{field1}_cosine") if f"{field1}_cosine" in feature_names else -1
                cosine2_index = feature_names.index(f"{field2}_cosine") if f"{field2}_cosine" in feature_names else -1
                
                if cosine1_index >= 0 and cosine2_index >= 0 and feature_computed[cosine1_index] and feature_computed[cosine2_index]:
                    cosine1 = feature_vector[cosine1_index]
                    cosine2 = feature_vector[cosine2_index]
                    
                    # Compute ratio
                    if cosine2 > 0:
                        ratio = cosine1 / cosine2
                    else:
                        ratio = 0.0
                    
                    feature_vector[feature_index] = min(ratio, 10.0)  # Cap at 10.0
                    feature_computed[feature_index] = True
        
        # Compute birth/death year match features
        birth_year_match_index = feature_names.index("birth_year_match") if "birth_year_match" in feature_names else -1
        death_year_match_index = feature_names.index("death_year_match") if "death_year_match" in feature_names else -1
        has_birth_death_years_index = feature_names.index("has_birth_death_years") if "has_birth_death_years" in feature_names else -1
        
        if (birth_year_match_index >= 0 or death_year_match_index >= 0 or has_birth_death_years_index >= 0) and 'person' in record1_fields and 'person' in record2_fields:
            person1_hash = record1_fields['person']
            person2_hash = record2_fields['person']
            
            if person1_hash in unique_strings and person2_hash in unique_strings:
                person1_name = unique_strings[person1_hash]
                person2_name = unique_strings[person2_hash]
                
                # Extract birth/death years
                birth1, death1 = birth_death_parser.parse(person1_name)
                birth2, death2 = birth_death_parser.parse(person2_name)
                
                # Set birth year match feature
                if birth_year_match_index >= 0:
                    if birth1 is not None and birth2 is not None:
                        feature_vector[birth_year_match_index] = 1.0 if birth1 == birth2 else 0.0
                    else:
                        feature_vector[birth_year_match_index] = 0.0
                    
                    feature_computed[birth_year_match_index] = True
                
                # Set death year match feature
                if death_year_match_index >= 0:
                    if death1 is not None and death2 is not None:
                        feature_vector[death_year_match_index] = 1.0 if death1 == death2 else 0.0
                    else:
                        feature_vector[death_year_match_index] = 0.0
                    
                    feature_computed[death_year_match_index] = True
                
                # Set has birth/death years feature
                if has_birth_death_years_index >= 0:
                    has_years = (birth1 is not None or death1 is not None) and (birth2 is not None or death2 is not None)
                    feature_vector[has_birth_death_years_index] = 1.0 if has_years else 0.0
                    feature_computed[has_birth_death_years_index] = True
        
        # Set missing features to 0.0
        for i in range(len(feature_names)):
            if not feature_computed[i]:
                feature_vector[i] = 0.0
        
        return feature_vector

    def _get_vector_by_hash(self, collection, hash_value, field_type):
        """
        Get vector for a hash value and field type with retry logic.
        
        Args:
            collection: Weaviate collection
            hash_value (str): Hash value
            field_type (str): Field type
            
        Returns:
            list: Vector or None if not found
        """
        from tenacity import retry, stop_after_attempt, wait_exponential
        
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        def get_vector_with_retry():
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
                raise
        
        return get_vector_with_retry()

    def _save_results(self):
        """
        Save feature engineering results with enhanced error handling.
        """
        output_dir = Path(self.config['system']['output_dir'])
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Log feature vector status before saving
        logger.info(f"Saving {len(self.feature_vectors)} feature vectors")
        logger.info(f"Saving {len(self.labels)} labels")
        logger.info(f"Saving {len(self.feature_names)} feature names")
        
        # Convert feature vectors and labels to numpy arrays
        if len(self.feature_vectors) > 0:
            feature_vectors_np = np.array(self.feature_vectors)
            labels_np = np.array(self.labels)
            
            # Save feature vectors and labels
            feature_vectors_path = output_dir / "feature_vectors.npy"
            labels_path = output_dir / "labels.npy"
            
            np.save(feature_vectors_path, feature_vectors_np)
            np.save(labels_path, labels_np)
            
            logger.info(f"Saved feature vectors to {feature_vectors_path} with shape {feature_vectors_np.shape}")
            logger.info(f"Saved labels to {labels_path} with shape {labels_np.shape}")
        else:
            logger.warning("No feature vectors to save")
        
        # Save feature names
        with open(output_dir / "feature_names.json", 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        
        # Save prefiltered pairs
        with open(output_dir / "prefiltered_true.json", 'w') as f:
            json.dump(self.prefiltered_true, f, indent=2)
        
        with open(output_dir / "prefiltered_false.json", 'w') as f:
            json.dump(self.prefiltered_false, f, indent=2)
        
        # Save statistics
        stats = {
            'total_vectors': len(self.feature_vectors),
            'positive_examples': sum(self.labels),
            'negative_examples': len(self.labels) - sum(self.labels),
            'prefiltered_true': len(self.prefiltered_true),
            'prefiltered_false': len(self.prefiltered_false),
            'feature_count': len(self.feature_names)
        }
        
        with open(output_dir / "feature_statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save final checkpoint
        checkpoint_path = Path(self.config['system']['checkpoint_dir']) / "features_final.ckpt"
        save_checkpoint({
            'feature_vectors': self.feature_vectors,
            'labels': self.labels,
            'feature_names': self.feature_names,
            'prefiltered_true': self.prefiltered_true,
            'prefiltered_false': self.prefiltered_false,
            'processed_pairs': []  # Empty list to save space
        }, checkpoint_path)
        
        logger.info("Feature engineering results saved to %s", output_dir)

    def normalize_features(self, feature_vectors):
        """
        Normalize feature vectors.
        
        Args:
            feature_vectors (list or numpy.ndarray): Feature vectors
            
        Returns:
            numpy.ndarray: Normalized feature vectors
        """
        # Convert to numpy array if needed
        if not isinstance(feature_vectors, np.ndarray):
            feature_vectors = np.array(feature_vectors)
        
        # Calculate mean and standard deviation for each feature
        means = np.mean(feature_vectors, axis=0)
        stds = np.std(feature_vectors, axis=0)
        
        # Handle zero standard deviation
        stds[stds == 0] = 1.0
        
        # Normalize features
        normalized_vectors = (feature_vectors - means) / stds
        
        return normalized_vectors

    def get_feature_vector(self, record1_id, record2_id):
        """
        Get feature vector for a record pair.
        
        Args:
            record1_id (str): First record ID
            record2_id (str): Second record ID
            
        Returns:
            list: Feature vector or None if not found
        """
        # TODO: Implement retrieval of precomputed feature vectors
        return None

    def _check_feature_files(self):
        """
        Check the state of feature files to help diagnose issues.
        """
        output_dir = Path(self.config['system']['output_dir'])
        
        # Check feature vectors
        feature_vectors_path = output_dir / "feature_vectors.npy"
        if feature_vectors_path.exists():
            try:
                feature_vectors = np.load(feature_vectors_path)
                logger.info(f"Feature vectors file exists with shape: {feature_vectors.shape}")
            except Exception as e:
                logger.error(f"Error loading feature vectors: {e}")
        else:
            logger.error(f"Feature vectors file does not exist: {feature_vectors_path}")
        
        # Check labels
        labels_path = output_dir / "labels.npy"
        if labels_path.exists():
            try:
                labels = np.load(labels_path)
                logger.info(f"Labels file exists with shape: {labels.shape}")
            except Exception as e:
                logger.error(f"Error loading labels: {e}")
        else:
            logger.error(f"Labels file does not exist: {labels_path}")
        
        # Check feature names
        feature_names_path = output_dir / "feature_names.json"
        if feature_names_path.exists():
            try:
                with open(feature_names_path, 'r') as f:
                    feature_names = json.load(f)
                logger.info(f"Feature names file exists with {len(feature_names)} features")
            except Exception as e:
                logger.error(f"Error loading feature names: {e}")
        else:
            logger.error(f"Feature names file does not exist: {feature_names_path}")

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