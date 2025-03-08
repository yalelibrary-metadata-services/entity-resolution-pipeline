"""
Parallel feature engineering module for large-scale entity resolution.
"""
import os
import logging
import json
import csv
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor
import pickle
import weaviate
from weaviate.classes.query import Filter
from scipy.spatial.distance import cosine
from rapidfuzz import fuzz
import Levenshtein
import re

from src.utils import save_checkpoint, load_checkpoint, Timer, get_memory_usage
from src.birth_death_regexes import BirthDeathYearExtractor
from src.mmap_dict import MMapDict

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Handles construction of feature vectors for large-scale entity resolution.
    
    Features:
    - Robust serialization for parallel processing
    - Memory-mapped storage for large datasets
    - Sophisticated feature engineering with interaction features
    - Built-in prefiltering for efficient classification
    """
    
    def __init__(self, config):
        """
        Initialize the feature engineer with configuration parameters.
        """
        self.config = config
        
        # Initialize birth/death year parser
        self.birth_death_parser = BirthDeathYearExtractor()
        
        # Determine storage approach based on dataset size and mode
        self.use_mmap = self.config['system']['mode'] == 'prod'
        self.mmap_dir = Path(self.config['system']['temp_dir']) / "mmap"
        self.mmap_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize storage
        self._initialize_storage()
        
        # Load ground truth data
        self.ground_truth = self._load_ground_truth()
        
        # Initialize unique strings and record field hashes
        self.unique_strings = self._load_unique_strings()
        self.record_field_hashes = self._load_record_field_hashes()
        
        # Initialize feature configuration
        self.feature_names = self._initialize_feature_config()
        
        logger.info("FeatureEngineer initialized with %d feature types", 
                   len(self.feature_names))

    def _initialize_storage(self):
        """
        Initialize storage based on configuration.
        """
        if self.use_mmap:
            # For large datasets, use memory-mapped files
            self.feature_vectors_file = self.mmap_dir / "feature_vectors.npy"
            self.labels_file = self.mmap_dir / "labels.npy"
            self.prefiltered_true_file = self.mmap_dir / "prefiltered_true.pkl"
            self.prefiltered_false_file = self.mmap_dir / "prefiltered_false.pkl"
            
            # Initialize empty data structures
            self.feature_vectors = []
            self.labels = []
            self.prefiltered_true = []
            self.prefiltered_false = []
            
            # Create empty memory-mapped array for feature_vectors and labels
            # We'll resize it as needed
            if not self.feature_vectors_file.exists():
                # Initial empty array
                np.save(self.feature_vectors_file, np.array([]))
            
            if not self.labels_file.exists():
                # Initial empty array
                np.save(self.labels_file, np.array([]))
        else:
            # For small datasets, use in-memory lists
            self.feature_vectors = []
            self.labels = []
            self.prefiltered_true = []
            self.prefiltered_false = []

    def _initialize_feature_config(self):
        """
        Initialize feature configuration from config settings.
        """
        # Start with empty feature names
        feature_names = []
        
        # Add cosine similarity features
        cosine_similarities = self.config['features']['cosine_similarities']
        for field in cosine_similarities:
            feature_names.append(f"{field}_cosine")
        
        # Add string similarity features
        string_similarities = self.config['features']['string_similarities']
        for sim_config in string_similarities:
            field = sim_config['field']
            metrics = sim_config['metrics']
            
            for metric in metrics:
                feature_names.append(f"{field}_{metric}")
        
        # Add harmonic mean features
        harmonic_means = self.config['features']['harmonic_means']
        for field_pair in harmonic_means:
            field1, field2 = field_pair
            feature_names.append(f"{field1}_{field2}_harmonic")
        
        # Add additional interaction features
        additional_interactions = self.config['features']['additional_interactions']
        for interaction in additional_interactions:
            interaction_type = interaction['type']
            fields = interaction['fields']
            feature_names.append(f"{fields[0]}_{fields[1]}_{interaction_type}")
        
        # Add birth/death year match feature
        feature_names.append("birth_year_match")
        feature_names.append("death_year_match")
        feature_names.append("has_birth_death_years")
        
        logger.info("Initialized %d feature names", len(feature_names))
        return feature_names

    def execute(self, checkpoint=None):
        """
        Execute feature engineering using ground truth pairs directly.
        
        Instead of building neighborhoods, this approach:
        1. Reads pairs directly from the ground truth file
        2. Retrieves record information for each pair
        3. Constructs feature vectors for those pairs
        4. Uses these for model training
        
        Args:
            checkpoint (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Feature engineering results
        """
        # Load ground truth data
        self.ground_truth = self._load_ground_truth()
        logger.info(f"Loaded {len(self.ground_truth)} ground truth pairs")
        
        if len(self.ground_truth) == 0:
            logger.error("No ground truth pairs found! Check ground truth file path.")
            return {
                'pairs_processed': 0,
                'feature_vectors': 0,
                'feature_count': len(self.feature_names),
                'duration': 0.0
            }
        
        # Initialize storage for feature vectors and labels
        feature_vectors = []
        labels = []
        processed_pairs = set()
        
        # Connect to Weaviate for vector operations if needed
        weaviate_client = None
        if self.config['weaviate']['host'] and self.config['weaviate']['port']:
            try:
                import weaviate
                weaviate_client = weaviate.connect_to_local(
                    host=self.config['weaviate']['host'],
                    port=self.config['weaviate']['port']
                )
                collection_name = self.config['weaviate']['collection_name']
                collection = weaviate_client.collections.get(collection_name)
                logger.info(f"Connected to Weaviate collection: {collection_name}")
            except Exception as e:
                logger.warning(f"Could not connect to Weaviate: {e}. Will proceed without vector operations.")
                collection = None
        else:
            collection = None
        
        # Track progress
        start_time = time.time()
        logger.info(f"Starting feature engineering on {len(self.ground_truth)} ground truth pairs")
        
        # Process each pair in the ground truth file
        for pair_id, is_match in tqdm(self.ground_truth.items(), desc="Processing ground truth pairs"):
            try:
                # Split pair ID to get record IDs
                record_ids = pair_id.split('|')
                if len(record_ids) != 2:
                    logger.warning(f"Invalid pair ID format: {pair_id}")
                    continue
                
                record1_id, record2_id = record_ids
                
                # Get field hashes for records
                record1_fields = self.record_field_hashes.get(record1_id, {})
                record2_fields = self.record_field_hashes.get(record2_id, {})
                
                # Skip if missing essential fields
                if not record1_fields or not record2_fields:
                    logger.warning(f"Missing field data for pair {pair_id}")
                    continue
                
                # Construct feature vector
                feature_vector = self._construct_feature_vector(
                    record1_id, record2_id,
                    record1_fields, record2_fields,
                    self.unique_strings, collection,
                    self.feature_names
                )
                
                if feature_vector:
                    feature_vectors.append(feature_vector)
                    labels.append(1 if is_match else 0)
                    processed_pairs.add(pair_id)
                else:
                    logger.warning(f"Could not construct feature vector for pair {pair_id}")
            
            except Exception as e:
                logger.error(f"Error processing pair {pair_id}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Check if we generated any feature vectors
        if len(feature_vectors) == 0:
            logger.error("No feature vectors generated! Check ground truth and record data.")
            return {
                'pairs_processed': 0,
                'feature_vectors': 0,
                'feature_count': len(self.feature_names),
                'duration': time.time() - start_time
            }
        
        # Run diagnostic check on generated features
        self._diagnostic_feature_check(feature_vectors)
        
        # Save results
        output_dir = Path(self.config['system']['output_dir'])
        
        # Save feature names
        with open(output_dir / "feature_names.json", 'w') as f:
            json.dump(self.feature_names, f)
        
        # Save feature vectors and labels as numpy arrays
        feature_vectors_np = np.array(feature_vectors)
        labels_np = np.array(labels)
        
        np.save(output_dir / "feature_vectors.npy", feature_vectors_np)
        np.save(output_dir / "labels.npy", labels_np)
        
        # Save a reference to memory-mapped files if using them
        if self.use_mmap:
            # Save to memory-mapped files
            np.save(self.mmap_dir / "feature_vectors.npy", feature_vectors_np)
            np.save(self.mmap_dir / "labels.npy", labels_np)
            
            with open(output_dir / "feature_vectors_info.json", 'w') as f:
                json.dump({
                    'count': len(feature_vectors),
                    'feature_count': len(self.feature_names),
                    'feature_vectors_file': str(self.mmap_dir / "feature_vectors.npy"),
                    'labels_file': str(self.mmap_dir / "labels.npy")
                }, f)
        
        # Save processed pairs for reference
        with open(output_dir / "processed_pairs.json", 'w') as f:
            json.dump(list(processed_pairs), f)
        
        # Close Weaviate client if it was opened
        if weaviate_client:
            try:
                weaviate_client.close()
            except:
                pass
        
        duration = time.time() - start_time
        logger.info(f"Feature engineering completed: {len(processed_pairs)} pairs processed, "
                f"{len(feature_vectors)} feature vectors, {duration:.2f} seconds")
        
        return {
            'pairs_processed': len(processed_pairs),
            'feature_vectors': len(feature_vectors),
            'feature_count': len(self.feature_names),
            'duration': duration
        }

    def _load_unique_strings(self):
        """
        Load unique strings with support for large datasets.
        """
        try:
            output_dir = Path(self.config['system']['output_dir'])
            
            # Try to load from index first (enhanced preprocessor)
            index_path = output_dir / "unique_strings_index.json"
            if index_path.exists():
                with open(index_path, 'r') as f:
                    index = json.load(f)
                
                location = index.get('location')
                
                if location != "in-memory" and os.path.exists(location):
                    # For large datasets, use memory-mapped dictionary
                    return MMapDict(location)
                else:
                    # Fall back to sample file
                    with open(output_dir / "unique_strings_sample.json", 'r') as f:
                        return json.load(f)
            else:
                # Fall back to original approach
                with open(output_dir / "unique_strings_sample.json", 'r') as f:
                    return json.load(f)
        
        except Exception as e:
            logger.error("Error loading unique strings: %s", str(e))
            return {}

    def _load_record_field_hashes(self):
        """
        Load record field hashes with support for large datasets.
        """
        try:
            output_dir = Path(self.config['system']['output_dir'])
            
            # Try to load from index first (enhanced preprocessor)
            index_path = output_dir / "record_index.json"
            if index_path.exists():
                with open(index_path, 'r') as f:
                    index = json.load(f)
                
                location = index.get('location')
                
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
        Load ground truth data with enhanced error handling and diagnostics.
        
        Returns:
            dict: Dictionary mapping pair keys to match status (True/False)
        """
        try:
            ground_truth_file = Path(self.config['data']['ground_truth_file'])
            
            if not ground_truth_file.exists():
                logger.error(f"Ground truth file not found: {ground_truth_file}")
                return {}
            
            ground_truth = {}
            error_count = 0
            
            with open(ground_truth_file, 'r') as f:
                # Try to detect file format
                first_line = f.readline().strip()
                f.seek(0)  # Reset file pointer
                
                # Check if this is a CSV file with header
                if first_line and ',' in first_line and any(header in first_line.lower() 
                                                        for header in ['left', 'right', 'match']):
                    logger.info(f"Detected CSV format with header for ground truth file")
                    reader = csv.reader(f)
                    next(reader, None)  # Skip header
                    
                    for row in reader:
                        if len(row) >= 3:
                            left_id, right_id, match = row
                            
                            # Ensure IDs are trimmed of any whitespace
                            left_id = left_id.strip()
                            right_id = right_id.strip()
                            
                            # Ensure consistent ordering of IDs
                            if left_id > right_id:
                                left_id, right_id = right_id, left_id
                            
                            # Normalize match value, handling various formats
                            match_value = str(match).strip().lower()
                            is_match = match_value in ['true', 'yes', 't', 'y', '1', 'match']
                            
                            pair_key = f"{left_id}|{right_id}"
                            ground_truth[pair_key] = is_match
                        else:
                            error_count += 1
                else:
                    # Try alternate formats (JSON, etc.)
                    try:
                        # Check if it's a JSON file
                        f.seek(0)  # Reset file pointer
                        data = json.load(f)
                        
                        if isinstance(data, list):
                            # List of pair objects
                            for pair in data:
                                if 'left' in pair and 'right' in pair and 'match' in pair:
                                    left_id = str(pair['left']).strip()
                                    right_id = str(pair['right']).strip()
                                    
                                    # Ensure consistent ordering
                                    if left_id > right_id:
                                        left_id, right_id = right_id, left_id
                                    
                                    pair_key = f"{left_id}|{right_id}"
                                    ground_truth[pair_key] = bool(pair['match'])
                        elif isinstance(data, dict):
                            # Dictionary mapping pair keys to match status
                            for pair_key, is_match in data.items():
                                ids = pair_key.split('|')
                                if len(ids) == 2:
                                    left_id = ids[0].strip()
                                    right_id = ids[1].strip()
                                    
                                    # Ensure consistent ordering
                                    if left_id > right_id:
                                        pair_key = f"{right_id}|{left_id}"
                                    
                                    ground_truth[pair_key] = bool(is_match)
                    except json.JSONDecodeError:
                        # Not a JSON file, try custom format parsing
                        f.seek(0)  # Reset file pointer
                        for line in f:
                            parts = line.strip().split(',')
                            if len(parts) >= 3:
                                left_id = parts[0].strip()
                                right_id = parts[1].strip()
                                match = parts[2].strip().lower()
                                
                                # Ensure consistent ordering
                                if left_id > right_id:
                                    left_id, right_id = right_id, left_id
                                
                                pair_key = f"{left_id}|{right_id}"
                                ground_truth[pair_key] = match in ['true', 'yes', 't', 'y', '1', 'match']
            
            # Log statistics
            logger.info(f"Loaded {len(ground_truth)} ground truth pairs")
            if error_count > 0:
                logger.warning(f"Encountered {error_count} errors while parsing ground truth file")
            
            # Log match/non-match distribution
            match_count = sum(1 for is_match in ground_truth.values() if is_match)
            non_match_count = len(ground_truth) - match_count
            match_percentage = (match_count / len(ground_truth) * 100) if ground_truth else 0
            
            logger.info(f"Ground truth distribution: {match_count} matches ({match_percentage:.1f}%), "
                    f"{non_match_count} non-matches ({100 - match_percentage:.1f}%)")
            
            # Sample some entries for verification
            if ground_truth:
                sample_entries = list(ground_truth.items())[:5]
                logger.info(f"Sample ground truth entries:")
                for pair_key, is_match in sample_entries:
                    logger.info(f"  {pair_key}: {is_match}")
            
            return ground_truth
        
        except Exception as e:
            logger.error(f"Error loading ground truth: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def _load_candidate_pairs(self):
        """
        Load candidate pairs with support for both file and memory-mapped storage.
        """
        try:
            output_dir = Path(self.config['system']['output_dir'])
            
            # Try to load from candidate pairs info first (for large datasets)
            info_path = output_dir / "candidate_pairs_info.json"
            if info_path.exists():
                with open(info_path, 'r') as f:
                    info = json.load(f)
                
                file_path = info.get('file')
                
                if file_path and os.path.exists(file_path):
                    # Load pairs from pickle file
                    pairs = []
                    with open(file_path, 'rb') as f:
                        try:
                            while True:
                                pair = pickle.load(f)
                                pairs.append(pair)
                        except EOFError:
                            pass
                    
                    return pairs
            
            # Fall back to standard JSON file
            pairs_path = output_dir / "candidate_pairs.json"
            if pairs_path.exists():
                with open(pairs_path, 'r') as f:
                    return json.load(f)
            
            # Try sample file as last resort
            sample_path = output_dir / "candidate_pairs_sample.json"
            if sample_path.exists():
                with open(sample_path, 'r') as f:
                    return json.load(f)
            
            logger.warning("No candidate pairs found")
            return []
        
        except Exception as e:
            logger.error("Error loading candidate pairs: %s", str(e))
            return []

    def _create_batches(self, items, batch_size):
        """
        Create batches of items for parallel processing.
        """
        return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    def _get_pair_id(self, record1_id, record2_id):
        """
        Get unique identifier for a record pair with consistent ordering.
        """
        # Convert None or non-string values to strings
        record1_id = str(record1_id) if record1_id is not None else ""
        record2_id = str(record2_id) if record2_id is not None else ""
        
        # Ensure consistent ordering
        if record1_id > record2_id:
            record1_id, record2_id = record2_id, record1_id
        
        return f"{record1_id}|{record2_id}"

    def _process_batch_sync(self, batch, collection, feature_names, unique_strings, record_field_hashes):
        """
        Process a batch of candidate pairs synchronously.
        """
        # Initialize results
        vectors = []
        labels = []
        prefiltered_true = []
        prefiltered_false = []
        pair_ids = []
        
        # Process each pair
        for pair in batch:
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

            pair_id_reversed = f"{record2_id}|{record1_id}"
            if pair_id in self.ground_truth:
                label = 1 if self.ground_truth[pair_id] else 0
            elif pair_id_reversed in self.ground_truth:
                label = 1 if self.ground_truth[pair_id_reversed] else 0
            else:
                label = None
            
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
                unique_strings
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
                feature_names
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

    def _process_batch_direct(self, batch):
        """
        Process a batch of candidate pairs with direct hash-to-ID mapping.
        """
        # Initialize results
        vectors = []
        labels = []
        prefiltered_true = []
        prefiltered_false = []
        pair_ids = []
        
        # Build a hash-to-id mapping from ground truth for faster lookups
        hash_to_id_map = {}
        
        # Collect hashes from records for quick lookup
        for record_id, field_hashes in self.record_field_hashes.items():
            if 'person' in field_hashes:
                person_hash = field_hashes['person']
                if person_hash != 'NULL':
                    # Multiple records might have the same person hash
                    if person_hash not in hash_to_id_map:
                        hash_to_id_map[person_hash] = []
                    hash_to_id_map[person_hash].append(record_id)
        
        logger.info(f"Built hash-to-id map with {len(hash_to_id_map)} unique hashes")
        
        # Track statistics
        total_pairs = len(batch)
        matched_pairs = 0
        ground_truth_matches = 0
        
        # Process each pair
        for pair in batch:
            record1_id = pair['record1_id']
            record2_hash = pair['record2_hash']
            
            # Get record2_id from hash mapping
            record2_ids = hash_to_id_map.get(record2_hash, [])
            
            if not record2_ids:
                continue
            
            # Take the first matching record ID
            record2_id = record2_ids[0]
            
            # Skip self-matches
            if record1_id == record2_id:
                continue
            
            matched_pairs += 1
            
            # Check both orderings in ground truth
            pair_id_forward = f"{record1_id}|{record2_id}"
            pair_id_reverse = f"{record2_id}|{record1_id}"
            
            label = None
            if pair_id_forward in self.ground_truth:
                label = 1 if self.ground_truth[pair_id_forward] else 0
                pair_id = pair_id_forward
                ground_truth_matches += 1
            elif pair_id_reverse in self.ground_truth:
                label = 1 if self.ground_truth[pair_id_reverse] else 0
                pair_id = pair_id_reverse
                ground_truth_matches += 1
            else:
                continue
            
            # Get field hashes for both records
            record1_fields = self.record_field_hashes.get(record1_id, {})
            record2_fields = self.record_field_hashes.get(record2_id, {})
            
            # Skip if missing essential fields
            if not record1_fields or not record2_fields:
                continue
            
            # Construct feature vector
            feature_vector = self._construct_feature_vector(
                record1_id, record2_id,
                record1_fields, record2_fields,
                self.unique_strings, None,  # Pass None for collection since we're not using it
                self.feature_names
            )
            
            if feature_vector:
                vectors.append(feature_vector)
                labels.append(label)
                pair_ids.append(pair_id)
        
        logger.info(f"Processed {total_pairs} pairs: {matched_pairs} matched to record2_id, {ground_truth_matches} found in ground truth")
        
        return {
            'vectors': vectors,
            'labels': labels,
            'prefiltered_true': prefiltered_true,
            'prefiltered_false': prefiltered_false,
            'pair_ids': pair_ids
        }

    def _save_batch_results(self, batch_vectors, batch_labels, batch_prefiltered_true, batch_prefiltered_false):
        """
        Save batch results with proper handling of memory-mapped storage.
        """
        if self.use_mmap:
            # For memory-mapped storage, append to memory-mapped files
            
            # Append feature vectors and labels to numpy arrays
            if len(batch_vectors) > 0:
                if self.feature_vectors_count == 0:
                    # First batch, create new arrays
                    np.save(self.feature_vectors_file, batch_vectors)
                    np.save(self.labels_file, batch_labels)
                else:
                    # Append to existing arrays
                    existing_vectors = np.load(self.feature_vectors_file)
                    existing_labels = np.load(self.labels_file)
                    
                    # Concatenate arrays
                    new_vectors = np.vstack([existing_vectors, batch_vectors])
                    new_labels = np.hstack([existing_labels, batch_labels])
                    
                    # Save arrays
                    np.save(self.feature_vectors_file, new_vectors)
                    np.save(self.labels_file, new_labels)
                
                # Update counts
                self.feature_vectors_count += len(batch_vectors)
            
            # Append prefiltered pairs to pickle files
            if batch_prefiltered_true:
                with open(self.prefiltered_true_file, 'ab') as f:
                    for pair in batch_prefiltered_true:
                        pickle.dump(pair, f)
                self.prefiltered_true_count += len(batch_prefiltered_true)
            
            if batch_prefiltered_false:
                with open(self.prefiltered_false_file, 'ab') as f:
                    for pair in batch_prefiltered_false:
                        pickle.dump(pair, f)
                self.prefiltered_false_count += len(batch_prefiltered_false)
        else:
            # For in-memory storage, append to lists
            if len(batch_vectors) > 0:
                # Convert to list if numpy array
                if isinstance(batch_vectors, np.ndarray):
                    batch_vectors = batch_vectors.tolist()
                if isinstance(batch_labels, np.ndarray):
                    batch_labels = batch_labels.tolist()
                
                self.feature_vectors.extend(batch_vectors)
                self.labels.extend(batch_labels)
            
            self.prefiltered_true.extend(batch_prefiltered_true)
            self.prefiltered_false.extend(batch_prefiltered_false)

    def _save_checkpoint(self, processed_pairs, batch_idx):
        """
        Save checkpoint for feature engineering progress.
        """
        try:
            checkpoint_dir = Path(self.config['system']['checkpoint_dir'])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = checkpoint_dir / f"features_{batch_idx}.ckpt"
            
            # Prepare checkpoint data
            if self.use_mmap:
                # For memory-mapped storage, save file paths and counts
                checkpoint_data = {
                    'feature_vectors_file': str(self.feature_vectors_file),
                    'labels_file': str(self.labels_file),
                    'prefiltered_true_file': str(self.prefiltered_true_file),
                    'prefiltered_false_file': str(self.prefiltered_false_file),
                    'feature_vectors_count': self.feature_vectors_count,
                    'prefiltered_true_count': self.prefiltered_true_count,
                    'prefiltered_false_count': self.prefiltered_false_count,
                    'feature_names': self.feature_names,
                    'processed_pairs': list(processed_pairs)
                }
            else:
                # For in-memory storage, save actual data
                checkpoint_data = {
                    'feature_vectors': self.feature_vectors,
                    'labels': self.labels,
                    'prefiltered_true': self.prefiltered_true,
                    'prefiltered_false': self.prefiltered_false,
                    'feature_names': self.feature_names,
                    'processed_pairs': list(processed_pairs)
                }
            
            # Save checkpoint
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            logger.info("Saved feature engineering checkpoint at %s with %d pairs", 
                       checkpoint_path, len(processed_pairs))
        
        except Exception as e:
            logger.error("Error saving checkpoint: %s", str(e))

    # Add to src/parallel_features.py in the _save_results method
    def _save_results(self):
        """
        Save feature engineering results with proper synchronization.
        """
        output_dir = Path(self.config['system']['output_dir'])
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save feature names
        with open(output_dir / "feature_names.json", 'w') as f:
            json.dump(self.feature_names, f)
        
        # Get feature vectors and labels
        if self.use_mmap:
            # Update memory-mapped files with current data
            np.save(self.feature_vectors_file, np.array(self.feature_vectors))
            np.save(self.labels_file, np.array(self.labels))
            
            # Copy to standard location for compatibility
            feature_vectors = np.load(self.feature_vectors_file)
            labels = np.load(self.labels_file)
            
            np.save(output_dir / "feature_vectors.npy", feature_vectors)
            np.save(output_dir / "labels.npy", labels)
            
            # Save reference file
            with open(output_dir / "feature_vectors_info.json", 'w') as f:
                json.dump({
                    'count': len(feature_vectors),
                    'feature_count': len(self.feature_names),
                    'feature_vectors_file': str(self.feature_vectors_file),
                    'labels_file': str(self.labels_file)
                }, f)
            
            logger.info(f"Saved {len(feature_vectors)} feature vectors and {len(labels)} labels")
            
            # Update internal counts
            self.feature_vectors_count = len(feature_vectors)
        else:
            # For in-memory storage, simply save to output directory
            feature_vectors = np.array(self.feature_vectors)
            labels = np.array(self.labels)
            
            np.save(output_dir / "feature_vectors.npy", feature_vectors)
            np.save(output_dir / "labels.npy", labels)
            
            logger.info(f"Saved {len(feature_vectors)} feature vectors and {len(labels)} labels")
    
    def _find_records_by_hash(self, hash_value, field_type, record_field_hashes):
        """
        Find records containing a hash value for a specific field.
        """
        matching_records = []
        
        for record_id, field_hashes in record_field_hashes.items():
            if field_type in field_hashes and field_hashes[field_type] == hash_value:
                matching_records.append(record_id)
        
        return matching_records

    def _diagnostic_feature_check(self, feature_vectors):
        """
        Perform comprehensive diagnostic checks on feature vectors to identify quality issues.
        
        Args:
            feature_vectors (list): List of feature vectors to analyze
        """
        if not feature_vectors or len(feature_vectors) == 0:
            logger.warning("No feature vectors to analyze")
            return
        
        num_vectors = len(feature_vectors)
        num_features = len(feature_vectors[0])
        logger.info(f"Analyzing {num_vectors} feature vectors with {num_features} features")
        
        # 1. Check for constant features
        constant_features = []
        constant_values = []
        for i in range(num_features):
            first_value = feature_vectors[0][i]
            if all(vector[i] == first_value for vector in feature_vectors):
                constant_features.append(i)
                constant_values.append(first_value)
        
        if constant_features:
            logger.warning(f"Found {len(constant_features)} constant features at indices: {constant_features}")
            logger.warning(f"Constant values: {constant_values}")
            
            # If feature names are available, map indices to names
            if hasattr(self, 'feature_names') and len(self.feature_names) == num_features:
                constant_feature_names = [self.feature_names[i] for i in constant_features]
                logger.warning(f"Constant feature names: {constant_feature_names}")
        
        # 2. Check for NaN/inf values
        nan_features = []
        for i in range(num_features):
            nan_count = sum(1 for vector in feature_vectors if not np.isfinite(vector[i]))
            if nan_count > 0:
                nan_features.append((i, nan_count))
        
        if nan_features:
            logger.warning(f"Found {len(nan_features)} features with NaN/inf values:")
            for idx, count in nan_features:
                feature_name = self.feature_names[idx] if hasattr(self, 'feature_names') else f"Feature {idx}"
                logger.warning(f"  {feature_name}: {count} NaN/inf values ({count/num_vectors*100:.1f}%)")
        
        # 3. Check feature value distributions
        try:
            # Convert to numpy array for efficient analysis
            feature_arrays = np.array(feature_vectors)
            
            # Calculate basic statistics for each feature
            min_values = feature_arrays.min(axis=0)
            max_values = feature_arrays.max(axis=0)
            mean_values = feature_arrays.mean(axis=0)
            std_values = feature_arrays.std(axis=0)
            
            # Identify features with suspicious distributions
            suspicious_features = []
            for i in range(num_features):
                if i not in constant_features:  # Skip features already identified as constant
                    feature_min = min_values[i]
                    feature_max = max_values[i]
                    feature_mean = mean_values[i]
                    feature_std = std_values[i]
                    
                    # Check for features with very low variance
                    if feature_std < 0.01 and feature_max - feature_min < 0.1:
                        feature_name = self.feature_names[i] if hasattr(self, 'feature_names') else f"Feature {i}"
                        suspicious_features.append((feature_name, feature_min, feature_max, feature_mean, feature_std))
            
            if suspicious_features:
                logger.warning(f"Found {len(suspicious_features)} features with suspicious distributions:")
                for feature_name, min_val, max_val, mean_val, std_val in suspicious_features:
                    logger.warning(f"  {feature_name}: min={min_val:.4f}, max={max_val:.4f}, "
                                f"mean={mean_val:.4f}, std={std_val:.4f}")
            
            # 4. Check for correlations between features
            if len(feature_vectors) > 10 and num_features > 1:
                try:
                    correlation_matrix = np.corrcoef(feature_arrays.T)
                    
                    # Find highly correlated feature pairs (ignoring self-correlations)
                    highly_correlated = []
                    for i in range(num_features):
                        for j in range(i+1, num_features):
                            if np.isfinite(correlation_matrix[i, j]) and abs(correlation_matrix[i, j]) > 0.95:
                                feature1 = self.feature_names[i] if hasattr(self, 'feature_names') else f"Feature {i}"
                                feature2 = self.feature_names[j] if hasattr(self, 'feature_names') else f"Feature {j}"
                                highly_correlated.append((feature1, feature2, correlation_matrix[i, j]))
                    
                    if highly_correlated:
                        logger.warning(f"Found {len(highly_correlated)} highly correlated feature pairs:")
                        for feature1, feature2, corr in highly_correlated[:10]:  # Limit to 10 examples
                            logger.warning(f"  {feature1} and {feature2}: correlation = {corr:.4f}")
                except Exception as e:
                    logger.warning(f"Could not compute feature correlations: {e}")
        
        except Exception as e:
            logger.warning(f"Error in feature distribution analysis: {e}")
        
        # 5. Check for duplicate vectors
        try:
            # This is an approximate check for duplicate vectors
            vector_strings = [str(v) for v in feature_vectors[:min(1000, len(feature_vectors))]]
            unique_vectors = set(vector_strings)
            duplicate_percent = (1 - len(unique_vectors) / len(vector_strings)) * 100
            
            if duplicate_percent > 10:
                logger.warning(f"Approximately {duplicate_percent:.1f}% of feature vectors are duplicates "
                            f"(based on sample of {len(vector_strings)} vectors)")
        except Exception as e:
            logger.warning(f"Error checking for duplicate vectors: {e}")
        
        # 6. Summary
        logger.info("Feature vector diagnostic summary:")
        logger.info(f"  Total vectors: {num_vectors}")
        logger.info(f"  Features per vector: {num_features}")
        logger.info(f"  Constant features: {len(constant_features)}")
        logger.info(f"  Features with NaN/inf values: {len(nan_features)}")

    def _apply_prefilters(self, record1_id, record2_id, record1_fields, record2_fields, unique_strings):
        """
        Apply prefilters to automatically classify candidate pairs.
        """
        # Check exact name with birth/death years prefilter
        if self.config['features']['prefilters']['exact_name_birth_death_prefilter']:
            if 'person' in record1_fields and 'person' in record2_fields:
                person1_hash = record1_fields['person']
                person2_hash = record2_fields['person']
                
                if person1_hash == person2_hash and person1_hash in unique_strings:
                    person_name = unique_strings[person1_hash]
                    
                    # Check if name contains birth/death years
                    birth1, death1 = self.birth_death_parser.parse(person_name)
                    
                    if birth1 is not None or death1 is not None:
                        return 'true'
        
        # Check composite cosine prefilter (simplified for scalability)
        if self.config['features']['prefilters']['composite_cosine_prefilter']['enabled']:
            threshold = self.config['features']['prefilters']['composite_cosine_prefilter']['threshold']
            
            if 'composite' in record1_fields and 'composite' in record2_fields:
                composite1_hash = record1_fields['composite']
                composite2_hash = record2_fields['composite']
                
                if composite1_hash in unique_strings and composite2_hash in unique_strings:
                    composite1 = unique_strings[composite1_hash]
                    composite2 = unique_strings[composite2_hash]
                    
                    # Simple string comparison for efficiency
                    if composite1 == composite2:
                        return 'true'
        
        # Check person cosine prefilter (simplified for scalability)
        if self.config['features']['prefilters']['person_cosine_prefilter']['enabled']:
            threshold = self.config['features']['prefilters']['person_cosine_prefilter']['threshold']
            
            if 'person' in record1_fields and 'person' in record2_fields:
                person1_hash = record1_fields['person']
                person2_hash = record2_fields['person']
                
                if person1_hash in unique_strings and person2_hash in unique_strings:
                    person1 = unique_strings[person1_hash]
                    person2 = unique_strings[person2_hash]
                    
                    # Simple string comparison for efficiency
                    if not self._has_min_name_similarity(person1, person2):
                        return 'false'
        
        # No prefilter matched
        return None

    def _has_min_name_similarity(self, name1, name2, threshold=0.3):
        """
        Check if names have minimum similarity.
        """
        # Simple check: normalized Levenshtein distance
        max_len = max(len(name1), len(name2))
        if max_len == 0:
            return False
        
        distance = Levenshtein.distance(name1, name2)
        similarity = 1.0 - (distance / max_len)
        
        return similarity >= threshold

    def _construct_feature_vector(self, record1_id, record2_id, record1_fields, record2_fields, 
                                 unique_strings, collection, feature_names):
        """
        Construct feature vector for a record pair.
        """

         # Use print statements instead of logger
        print(f"DIRECT DEBUG: Constructing features for {record1_id} and {record2_id}")
        print(f"DIRECT DEBUG: Person hashes: {record1_fields.get('person', 'NOT FOUND')}, {record2_fields.get('person', 'NOT FOUND')}")
        
        # Check the first 5 pairs only to avoid overwhelming output
        static_counter = getattr(self, '_debug_counter', 0)
        setattr(self, '_debug_counter', static_counter + 1)
        if static_counter < 5:
            # Rest of debug code with print statements
            person1_hash = record1_fields.get('person', 'NOT FOUND')
            person2_hash = record2_fields.get('person', 'NOT FOUND')
            print(f"DIRECT DEBUG: Person1 in unique_strings: {person1_hash in unique_strings}")
            print(f"DIRECT DEBUG: Person2 in unique_strings: {person2_hash in unique_strings}")
            
            if person1_hash != 'NOT FOUND' and person2_hash != 'NOT FOUND':
                if person1_hash in unique_strings and person2_hash in unique_strings:
                    print(f"DIRECT DEBUG: Person1 string: {unique_strings[person1_hash]}")
                    print(f"DIRECT DEBUG: Person2 string: {unique_strings[person2_hash]}")
                else:
                    print(f"DIRECT DEBUG: Person hashes not found in unique_strings")

        # Initialize feature vector with zeros
        feature_vector = [0.0] * len(feature_names)
        feature_computed = [False] * len(feature_names)
        
        
        # Compute string similarity features for person field
        field = 'person'
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
                birth1, death1 = self.birth_death_parser.parse(person1_name)
                birth2, death2 = self.birth_death_parser.parse(person2_name)
                
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
        
        # Estimate cosine similarity for each field type
        field_similarities = {}
        for field in ['person', 'title', 'provision', 'subjects', 'composite']:
            feature_index = feature_names.index(f"{field}_cosine") if f"{field}_cosine" in feature_names else -1
            
            if feature_index >= 0:
                field1_hash = record1_fields.get(field, 'NULL')
                field2_hash = record2_fields.get(field, 'NULL')
                
                if field1_hash != 'NULL' and field2_hash != 'NULL' and field1_hash in unique_strings and field2_hash in unique_strings:
                    # For efficiency, use string comparison for identical strings
                    if field1_hash == field2_hash:
                        similarity = 1.0
                    else:
                        string1 = unique_strings[field1_hash]
                        string2 = unique_strings[field2_hash]
                        
                        # Estimate similarity from string features
                        if field == 'person':
                            # Use Levenshtein for person field
                            max_len = max(len(string1), len(string2))
                            if max_len > 0:
                                distance = Levenshtein.distance(string1, string2)
                                similarity = 1.0 - (distance / max_len)
                            else:
                                similarity = 1.0
                        else:
                            # Use token sort ratio for other fields
                            similarity = fuzz.token_sort_ratio(string1, string2) / 100.0
                    
                    field_similarities[field] = similarity
                    
                    if feature_index >= 0:
                        feature_vector[feature_index] = similarity
                        feature_computed[feature_index] = True
        
        # Compute interaction features using estimated similarities
        # Harmonic means
        for field_pair in [['person', 'title'], ['person', 'provision'], ['person', 'subjects'],
                          ['title', 'subjects'], ['title', 'provision'], ['provision', 'subjects']]:
            field1, field2 = field_pair
            feature_index = feature_names.index(f"{field1}_{field2}_harmonic") if f"{field1}_{field2}_harmonic" in feature_names else -1
            
            if feature_index >= 0 and field1 in field_similarities and field2 in field_similarities:
                sim1 = field_similarities[field1]
                sim2 = field_similarities[field2]
                
                # Compute harmonic mean
                if sim1 > 0 and sim2 > 0:
                    harmonic_mean = 2 * (sim1 * sim2) / (sim1 + sim2)
                else:
                    harmonic_mean = 0.0
                
                feature_vector[feature_index] = harmonic_mean
                feature_computed[feature_index] = True
        
        # Product interactions
        for field_pair in [['person', 'subjects']]:
            field1, field2 = field_pair
            feature_index = feature_names.index(f"{field1}_{field2}_product") if f"{field1}_{field2}_product" in feature_names else -1
            
            if feature_index >= 0 and field1 in field_similarities and field2 in field_similarities:
                sim1 = field_similarities[field1]
                sim2 = field_similarities[field2]
                
                # Compute product
                product = sim1 * sim2
                
                feature_vector[feature_index] = product
                feature_computed[feature_index] = True
        
        # Ratio interactions
        for field_pair in [['composite', 'subjects']]:
            field1, field2 = field_pair
            feature_index = feature_names.index(f"{field1}_{field2}_ratio") if f"{field1}_{field2}_ratio" in feature_names else -1
            
            if feature_index >= 0 and field1 in field_similarities and field2 in field_similarities:
                sim1 = field_similarities[field1]
                sim2 = field_similarities[field2]
                
                # Compute ratio
                if sim2 > 0:
                    ratio = sim1 / sim2
                else:
                    ratio = 0.0
                
                feature_vector[feature_index] = min(ratio, 10.0)  # Cap at 10.0
                feature_computed[feature_index] = True
        
        # Set missing features to 0.0
        for i in range(len(feature_names)):
            if not feature_computed[i]:
                feature_vector[i] = 0.0
        
        return feature_vector

    def _check_feature_files(self):
        """
        Check the state of feature files for diagnostic purposes.
        """
        output_dir = Path(self.config['system']['output_dir'])
        
        # Check feature vectors
        feature_vectors_path = output_dir / "feature_vectors.npy"
        if feature_vectors_path.exists():
            try:
                feature_vectors = np.load(feature_vectors_path)
                logger.info("Feature vectors file exists with shape: %s", feature_vectors.shape)
            except Exception as e:
                logger.error("Error loading feature vectors: %s", str(e))
        else:
            # Check memory-mapped file
            if self.use_mmap and self.feature_vectors_file.exists():
                try:
                    feature_vectors = np.load(self.feature_vectors_file)
                    logger.info("Memory-mapped feature vectors file exists with shape: %s", 
                               feature_vectors.shape)
                except Exception as e:
                    logger.error("Error loading memory-mapped feature vectors: %s", str(e))
            else:
                logger.error("Feature vectors file does not exist")
        
        # Check labels
        labels_path = output_dir / "labels.npy"
        if labels_path.exists():
            try:
                labels = np.load(labels_path)
                logger.info("Labels file exists with shape: %s", labels.shape)
            except Exception as e:
                logger.error("Error loading labels: %s", str(e))
        else:
            # Check memory-mapped file
            if self.use_mmap and self.labels_file.exists():
                try:
                    labels = np.load(self.labels_file)
                    logger.info("Memory-mapped labels file exists with shape: %s", 
                               labels.shape)
                except Exception as e:
                    logger.error("Error loading memory-mapped labels: %s", str(e))
            else:
                logger.error("Labels file does not exist")
        
        # Check feature names
        feature_names_path = output_dir / "feature_names.json"
        if feature_names_path.exists():
            try:
                with open(feature_names_path, 'r') as f:
                    feature_names = json.load(f)
                logger.info("Feature names file exists with %d features", len(feature_names))
            except Exception as e:
                logger.error("Error loading feature names: %s", str(e))
        else:
            logger.error("Feature names file does not exist")


def process_batch_worker(batch, worker_config):
    """
    Worker function for parallel batch processing.
    This must be a top-level function to be picklable.
    """
    try:
        # Extract configuration
        weaviate_host = worker_config['weaviate_host']
        weaviate_port = worker_config['weaviate_port']
        collection_name = worker_config['collection_name']
        feature_names = worker_config['feature_names']
        prefilters = worker_config['prefilters']
        
        # Initialize resources
        client = weaviate.connect_to_local(
            host=weaviate_host,
            port=weaviate_port
        )
        collection = client.collections.get(collection_name)
        birth_death_parser = BirthDeathYearExtractor()
        
        # Load field hashes and unique strings
        # This is a simplified approach for the worker
        # In a production environment, you would use a shared database or cache
        
        # Process batch
        vectors = []
        labels = []
        prefiltered_true = []
        prefiltered_false = []
        pair_ids = []
        
        # Process each pair
        for pair in batch:
            try:
                record1_id = pair['record1_id']
                record1_hash = pair['record1_hash']
                record2_hash = pair['record2_hash']
                
                # For worker, we use Weaviate to get values directly
                # Get vectors for record1
                record1_hash_filter = Filter.by_property("hash").equal(record1_hash)
                record1_field_filter = Filter.by_property("field_type").equal("person")
                record1_filter = Filter.all_of([record1_hash_filter, record1_field_filter])
                
                record1_results = collection.query.fetch_objects(
                    filters=record1_filter,
                    limit=1
                )
                
                if not record1_results.objects:
                    continue
                
                record1_person = record1_results.objects[0].properties.get('value', '')
                
                # Get vectors for record2 (using hash)
                record2_hash_filter = Filter.by_property("hash").equal(record2_hash)
                record2_field_filter = Filter.by_property("field_type").equal("person")
                record2_filter = Filter.all_of([record2_hash_filter, record2_field_filter])
                
                record2_results = collection.query.fetch_objects(
                    filters=record2_filter,
                    limit=1
                )
                
                if not record2_results.objects:
                    continue
                
                record2_person = record2_results.objects[0].properties.get('value', '')
                
                # Get record2_id from query result if available
                record2_id = f"unknown_{record2_hash}"  # Fallback value
                
                # Verify record2_id is not the same as record1_id
                if record2_id == record1_id:
                    continue
                
                # Get pair ID
                record1_id = str(record1_id) if record1_id is not None else ""
                record2_id = str(record2_id) if record2_id is not None else ""

                if record1_id > record2_id:
                    record1_id, record2_id = record2_id, record1_id
                    record1_hash, record2_hash = record2_hash, record1_hash
                    record1_person, record2_person = record2_person, record1_person
                
                pair_id = f"{record1_id}|{record2_id}"
                
                # Apply simplified prefilters
                # Check exact name with birth/death years
                if prefilters['exact_name_birth_death_prefilter']:
                    if record1_hash == record2_hash:
                        # Check if name contains birth/death years
                        birth1, death1 = birth_death_parser.parse(record1_person)
                        
                        if birth1 is not None or death1 is not None:
                            prefiltered_true.append({
                                'record1_id': record1_id,
                                'record2_id': record2_id,
                                'reason': 'prefilter_exact_name_birth_death'
                            })
                            continue
                
                # Check minimum name similarity
                if prefilters['person_cosine_prefilter']['enabled']:
                    threshold = prefilters['person_cosine_prefilter']['threshold']
                    max_len = max(len(record1_person), len(record2_person))
                    
                    if max_len > 0:
                        distance = Levenshtein.distance(record1_person, record2_person)
                        similarity = 1.0 - (distance / max_len)
                        
                        if similarity < threshold:
                            prefiltered_false.append({
                                'record1_id': record1_id,
                                'record2_id': record2_id,
                                'reason': 'prefilter_person_cosine'
                            })
                            continue
                
                # Without ground truth, use a simplified feature vector
                # In a production environment, you would implement more sophisticated
                # feature engineering using the methods shown earlier
                
                # Create simple feature vector with person name similarity
                feature_vector = [0.0] * len(feature_names)
                
                # Person Levenshtein similarity
                person_levenshtein_idx = feature_names.index("person_levenshtein") if "person_levenshtein" in feature_names else -1
                if person_levenshtein_idx >= 0:
                    max_len = max(len(record1_person), len(record2_person))
                    if max_len > 0:
                        distance = Levenshtein.distance(record1_person, record2_person)
                        similarity = 1.0 - (distance / max_len)
                        feature_vector[person_levenshtein_idx] = similarity
                
                # Person Jaro-Winkler similarity
                person_jw_idx = feature_names.index("person_jaro_winkler") if "person_jaro_winkler" in feature_names else -1
                if person_jw_idx >= 0:
                    similarity = fuzz.token_sort_ratio(record1_person, record2_person) / 100.0
                    feature_vector[person_jw_idx] = similarity
                
                # Birth/death year features
                birth_year_match_idx = feature_names.index("birth_year_match") if "birth_year_match" in feature_names else -1
                death_year_match_idx = feature_names.index("death_year_match") if "death_year_match" in feature_names else -1
                has_birth_death_years_idx = feature_names.index("has_birth_death_years") if "has_birth_death_years" in feature_names else -1
                
                birth1, death1 = birth_death_parser.parse(record1_person)
                birth2, death2 = birth_death_parser.parse(record2_person)
                
                if birth_year_match_idx >= 0:
                    if birth1 is not None and birth2 is not None:
                        feature_vector[birth_year_match_idx] = 1.0 if birth1 == birth2 else 0.0
                
                if death_year_match_idx >= 0:
                    if death1 is not None and death2 is not None:
                        feature_vector[death_year_match_idx] = 1.0 if death1 == death2 else 0.0
                
                if has_birth_death_years_idx >= 0:
                    has_years = (birth1 is not None or death1 is not None) and (birth2 is not None or death2 is not None)
                    feature_vector[has_birth_death_years_idx] = 1.0 if has_years else 0.0
                
                # Add to results
                vectors.append(feature_vector)
                labels.append(1)  # In worker, we don't have ground truth, assume match
                pair_ids.append(pair_id)
            
            except Exception as e:
                import traceback
                print(f"Error processing pair: {str(e)}\n{traceback.format_exc()}")
                continue
        
        # Close client
        client.close()
        
        return {
            'vectors': vectors,
            'labels': labels,
            'prefiltered_true': prefiltered_true,
            'prefiltered_false': prefiltered_false,
            'pair_ids': pair_ids
        }
    
    except Exception as e:
        import traceback
        error_str = f"Error in worker: {str(e)}\n{traceback.format_exc()}"
        print(error_str)  # Print for parallel debugging
        return None
