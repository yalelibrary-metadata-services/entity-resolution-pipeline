"""
Redesigned feature engineering module for entity resolution pipeline.

This module provides the FeatureEngineer class, which handles construction of
feature vectors for entity resolution with improved data management.
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from tqdm import tqdm
import time

from src.utils import Timer
from src.data_manager import DataManager
from src.birth_death_regexes import BirthDeathYearExtractor

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Handles construction of feature vectors for entity resolution with improved
    data organization and consistency.
    
    Features:
    - Standardized data management using DataManager
    - Consistent feature construction regardless of data scale
    - Better debugging and validation
    - Simplified feature vector creation
    """
    
    def __init__(self, config):
        """
        Initialize the feature engineer with configuration parameters.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        
        # Initialize birth/death year parser
        self.birth_death_parser = BirthDeathYearExtractor()
        
        # Initialize data manager
        self.data_manager = DataManager(config)
        
        # Initialize tracking variables
        self.feature_names = []
        self.prefiltered_true = []
        self.prefiltered_false = []
        
        # Track statistics
        self.stats = {
            'total_pairs_processed': 0,
            'feature_vectors_created': 0,
            'prefiltered_true': 0,
            'prefiltered_false': 0,
            'missing_data_count': 0
        }
        
        logger.info("FeatureEngineer initialized")
    
    def execute(self, checkpoint=None):
        """
        Execute feature engineering for the entity resolution pipeline.
        
        Args:
            checkpoint (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Feature engineering results
        """
        logger.info("Starting feature engineering")
        
        # Check if feature vectors already exist and are valid
        if self._check_existing_features():
            logger.info("Using existing feature vectors")
            return self._load_feature_results()
        
        # Load ground truth data
        ground_truth = self._load_ground_truth()
        if not ground_truth:
            logger.error("No ground truth data found. Cannot proceed with feature engineering.")
            return {
                'error': 'No ground truth data found',
                'pairs_processed': 0,
                'feature_vectors_created': 0
            }
        
        logger.info(f"Loaded {len(ground_truth)} ground truth pairs")
        
        # Load data required for feature engineering
        unique_strings = self._load_unique_strings()
        record_field_hashes = self._load_record_field_hashes()
        
        if not unique_strings or not record_field_hashes:
            logger.error("Required data not found. Cannot proceed with feature engineering.")
            return {
                'error': 'Required data not found',
                'pairs_processed': 0,
                'feature_vectors_created': 0
            }
        
        # Initialize feature configuration
        self.feature_names = self._initialize_feature_config()
        
        # Process ground truth pairs
        with Timer() as timer:
            feature_vectors, labels, pair_ids = self._process_ground_truth_pairs(
                ground_truth, record_field_hashes, unique_strings
            )
        
        if len(feature_vectors) == 0:
            logger.error("No feature vectors were created. Check ground truth data and record field hashes.")
            return {
                'error': 'No feature vectors created',
                'pairs_processed': self.stats['total_pairs_processed'],
                'feature_vectors_created': 0
            }
        
        # Convert to numpy arrays
        feature_vectors_array = np.array(feature_vectors)
        labels_array = np.array(labels)
        
        # Run diagnostic check
        self._diagnostic_feature_check(feature_vectors_array)
        
        # Save results using data manager
        feature_paths = self.data_manager.save_feature_data(
            feature_vectors_array,
            labels_array,
            self.feature_names,
            stage='features'
        )
        
        # Save additional data
        self.data_manager.save('processed_pairs', pair_ids, stage='features')
        self.data_manager.save('prefiltered_true', self.prefiltered_true, stage='features')
        self.data_manager.save('prefiltered_false', self.prefiltered_false, stage='features')
        
        # Save feature statistics
        feature_stats = {
            'feature_count': len(self.feature_names),
            'sample_count': len(feature_vectors),
            'positive_samples': int(np.sum(labels_array == 1)),
            'negative_samples': int(np.sum(labels_array == 0)),
            'feature_names': self.feature_names,
            'prefiltered_true_count': len(self.prefiltered_true),
            'prefiltered_false_count': len(self.prefiltered_false),
            'processing_time': timer.duration
        }
        
        self.data_manager.save('feature_stats', feature_stats, stage='features')
        
        # Update overall stats
        self.stats.update({
            'feature_vectors_created': len(feature_vectors),
            'prefiltered_true': len(self.prefiltered_true),
            'prefiltered_false': len(self.prefiltered_false),
            'duration': timer.duration
        })
        
        logger.info(f"Feature engineering completed: {len(feature_vectors)} feature vectors created in {timer.duration:.2f} seconds")
        logger.info(f"Class distribution: {feature_stats['positive_samples']} positive, {feature_stats['negative_samples']} negative")
        
        return {
            'pairs_processed': self.stats['total_pairs_processed'],
            'feature_vectors_created': self.stats['feature_vectors_created'],
            'feature_count': len(self.feature_names),
            'duration': timer.duration,
            'paths': feature_paths
        }
    
    def _check_existing_features(self):
        """
        Check if feature vectors already exist and are valid.
        
        Returns:
            bool: True if valid feature vectors exist, False otherwise
        """
        # Check if feature index exists
        if not self.data_manager.exists('feature_index'):
            return False
        
        # Load feature data
        feature_vectors, labels, feature_names = self.data_manager.load_feature_data()
        
        # Check if data is valid
        if feature_vectors is None or labels is None or feature_names is None:
            return False
        
        # Check if data has reasonable shape
        if len(feature_vectors) < 10 or len(labels) < 10:
            logger.warning("Existing feature data has too few samples")
            return False
        
        # Check if both classes are represented
        if len(np.unique(labels)) < 2:
            logger.warning("Existing labels do not contain both classes")
            return False
        
        # Use existing feature names
        self.feature_names = feature_names
        
        return True
    
    def _load_feature_results(self):
        """
        Load existing feature engineering results.
        
        Returns:
            dict: Feature engineering results
        """
        # Load feature data
        feature_vectors, labels, feature_names = self.data_manager.load_feature_data()
        
        # Load additional data
        feature_stats = self.data_manager.load('feature_stats')
        processed_pairs = self.data_manager.load('processed_pairs')
        prefiltered_true = self.data_manager.load('prefiltered_true')
        prefiltered_false = self.data_manager.load('prefiltered_false')
        
        # Set instance variables
        self.feature_names = feature_names
        self.prefiltered_true = prefiltered_true or []
        self.prefiltered_false = prefiltered_false or []
        
        # Get feature index metadata
        feature_index = self.data_manager.get_metadata('feature_index')
        
        return {
            'pairs_processed': len(processed_pairs) if processed_pairs else 0,
            'feature_vectors_created': len(feature_vectors),
            'feature_count': len(feature_names),
            'duration': feature_stats.get('processing_time', 0) if feature_stats else 0,
            'paths': {
                'feature_vectors': feature_index.get('path') if feature_index else None,
                'feature_names': self.data_manager.get_metadata('feature_names', {}).get('path'),
                'labels': self.data_manager.get_metadata('labels', {}).get('path')
            }
        }
    
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
            
            # Check if ground truth data is already cached
            if self.data_manager.exists('ground_truth'):
                logger.info("Loading ground truth from data manager")
                return self.data_manager.load('ground_truth')
            
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
                    
                    # Load with pandas for better parsing
                    try:
                        df = pd.read_csv(ground_truth_file)
                        req_columns = ['left', 'right', 'match']
                        
                        # Check if required columns exist (case insensitive)
                        df_cols_lower = [col.lower() for col in df.columns]
                        col_map = {}
                        for req_col in req_columns:
                            if req_col in df_cols_lower:
                                col_map[req_col] = df.columns[df_cols_lower.index(req_col)]
                            else:
                                logger.error(f"Required column '{req_col}' not found in CSV")
                                return {}
                        
                        # Process rows
                        for _, row in df.iterrows():
                            left_id = str(row[col_map['left']]).strip()
                            right_id = str(row[col_map['right']]).strip()
                            
                            # Ensure consistent ordering of IDs
                            if left_id > right_id:
                                left_id, right_id = right_id, left_id
                            
                            # Normalize match value
                            match_str = str(row[col_map['match']]).strip().lower()
                            is_match = match_str in ['true', 'yes', 't', 'y', '1', 'match']
                            
                            # Create pair key
                            pair_key = f"{left_id}|{right_id}"
                            ground_truth[pair_key] = is_match
                        
                    except Exception as e:
                        logger.error(f"Error parsing CSV with pandas: {e}")
                        
                        # Fall back to manual CSV parsing
                        f.seek(0)  # Reset file pointer
                        import csv
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
                        import json
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
            
            # Cache ground truth for future use
            self.data_manager.save('ground_truth', ground_truth, 
                                  metadata={
                                      'match_count': match_count,
                                      'non_match_count': non_match_count,
                                      'match_percentage': match_percentage
                                  })
            
            return ground_truth
        
        except Exception as e:
            logger.error(f"Error loading ground truth: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def _load_unique_strings(self):
        """
        Load unique strings using data manager.
        
        Returns:
            dict: Dictionary of unique strings
        """
        # Check if already loaded by data manager
        if self.data_manager.exists('unique_strings'):
            return self.data_manager.load('unique_strings')
        
        try:
            output_dir = Path(self.config['system']['output_dir'])
            
            # Try different file formats in order of preference
            for file_name in ['unique_strings.json', 'unique_strings_sample.json']:
                file_path = output_dir / file_name
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        unique_strings = json.load(f)
                    
                    # Cache in data manager
                    self.data_manager.save('unique_strings', unique_strings, 
                                          metadata={'source_file': str(file_path)})
                    
                    logger.info(f"Loaded {len(unique_strings)} unique strings from {file_path}")
                    return unique_strings
            
            logger.error("No unique strings file found")
            return {}
        
        except Exception as e:
            logger.error(f"Error loading unique strings: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def _load_record_field_hashes(self):
        """
        Load record field hashes using data manager.
        
        Returns:
            dict: Dictionary of record field hashes
        """
        # Check if already loaded by data manager
        if self.data_manager.exists('record_field_hashes'):
            return self.data_manager.load('record_field_hashes')
        
        try:
            output_dir = Path(self.config['system']['output_dir'])
            
            # Try different file formats in order of preference
            for file_name in ['record_field_hashes.json', 'record_field_hashes_sample.json']:
                file_path = output_dir / file_name
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        record_field_hashes = json.load(f)
                    
                    # Cache in data manager
                    self.data_manager.save('record_field_hashes', record_field_hashes, 
                                          metadata={'source_file': str(file_path)})
                    
                    logger.info(f"Loaded {len(record_field_hashes)} record field hashes from {file_path}")
                    return record_field_hashes
            
            logger.error("No record field hashes file found")
            return {}
        
        except Exception as e:
            logger.error(f"Error loading record field hashes: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def _initialize_feature_config(self):
        """
        Initialize feature configuration from config settings.
        
        Returns:
            list: Feature names
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
        
        logger.info(f"Initialized {len(feature_names)} feature names")
        return feature_names
    
    def _process_ground_truth_pairs(self, ground_truth, record_field_hashes, unique_strings):
        """
        Process ground truth pairs to create feature vectors.
        
        Args:
            ground_truth (dict): Ground truth pairs
            record_field_hashes (dict): Record field hashes
            unique_strings (dict): Unique strings
            
        Returns:
            tuple: (feature_vectors, labels, pair_ids)
        """
        feature_vectors = []
        labels = []
        pair_ids = []
        
        # Process each pair in the ground truth
        for pair_id, is_match in tqdm(ground_truth.items(), desc="Processing pairs"):
            try:
                # Split pair ID into record IDs
                record_ids = pair_id.split('|')
                if len(record_ids) != 2:
                    logger.warning(f"Invalid pair ID format: {pair_id}")
                    continue
                
                record1_id, record2_id = record_ids
                
                # Get field hashes for records
                record1_fields = record_field_hashes.get(record1_id, {})
                record2_fields = record_field_hashes.get(record2_id, {})
                
                # Skip if missing essential fields
                if not record1_fields or not record2_fields:
                    logger.warning(f"Missing field data for pair {pair_id}")
                    self.stats['missing_data_count'] += 1
                    continue
                
                # Apply prefilters
                prefilter_result = self._apply_prefilters(
                    record1_id, record2_id,
                    record1_fields, record2_fields,
                    unique_strings
                )
                
                # Handle prefiltered pairs
                if prefilter_result == 'true':
                    self.prefiltered_true.append({
                        'record1_id': record1_id,
                        'record2_id': record2_id,
                        'reason': 'prefilter',
                        'actual_match': is_match
                    })
                    self.stats['prefiltered_true'] += 1
                    continue
                
                elif prefilter_result == 'false':
                    self.prefiltered_false.append({
                        'record1_id': record1_id,
                        'record2_id': record2_id,
                        'reason': 'prefilter',
                        'actual_match': is_match
                    })
                    self.stats['prefiltered_false'] += 1
                    continue
                
                # Construct feature vector
                feature_vector = self._construct_feature_vector(
                    record1_id, record2_id,
                    record1_fields, record2_fields,
                    unique_strings
                )
                
                # Add to results if feature vector was created successfully
                if feature_vector:
                    feature_vectors.append(feature_vector)
                    labels.append(1 if is_match else 0)
                    pair_ids.append(pair_id)
                
                self.stats['total_pairs_processed'] += 1
            
            except Exception as e:
                logger.error(f"Error processing pair {pair_id}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        return feature_vectors, labels, pair_ids
    
    def _apply_prefilters(self, record1_id, record2_id, record1_fields, record2_fields, unique_strings):
        """
        Apply prefilters to automatically classify candidate pairs.
        
        Args:
            record1_id (str): ID of first record
            record2_id (str): ID of second record
            record1_fields (dict): Field hashes for first record
            record2_fields (dict): Field hashes for second record
            unique_strings (dict): Dictionary of unique strings
            
        Returns:
            str: 'true' for true match, 'false' for false match, None for no prefilter match
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
        
        # Check composite cosine prefilter
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
        
        # Check person cosine prefilter
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
        
        Args:
            name1 (str): First name
            name2 (str): Second name
            threshold (float): Minimum similarity threshold
            
        Returns:
            bool: True if names have minimum similarity, False otherwise
        """
        try:
            import Levenshtein
            
            # Simple check: normalized Levenshtein distance
            max_len = max(len(name1), len(name2))
            if max_len == 0:
                return False
            
            distance = Levenshtein.distance(name1, name2)
            similarity = 1.0 - (distance / max_len)
            
            return similarity >= threshold
        except ImportError:
            # Fallback if Levenshtein not available
            return name1 == name2
    
    def _construct_feature_vector(self, record1_id, record2_id, record1_fields, record2_fields, unique_strings):
        """
        Construct feature vector for a record pair.
        
        Args:
            record1_id (str): ID of first record
            record2_id (str): ID of second record
            record1_fields (dict): Field hashes for first record
            record2_fields (dict): Field hashes for second record
            unique_strings (dict): Dictionary of unique strings
            
        Returns:
            list: Feature vector
        """
        # Initialize feature vector with zeros
        feature_vector = [0.0] * len(self.feature_names)
        feature_computed = [False] * len(self.feature_names)
        
        # Compute string similarity features for person field
        field = 'person'
        for metric in ['levenshtein', 'jaro_winkler']:
            feature_index = self.feature_names.index(f"{field}_{metric}") if f"{field}_{metric}" in self.feature_names else -1
            
            if feature_index >= 0:
                field1_hash = record1_fields.get(field, 'NULL')
                field2_hash = record2_fields.get(field, 'NULL')
                
                if field1_hash != 'NULL' and field2_hash != 'NULL' and field1_hash in unique_strings and field2_hash in unique_strings:
                    string1 = unique_strings[field1_hash]
                    string2 = unique_strings[field2_hash]
                    
                    if metric == 'levenshtein':
                        try:
                            import Levenshtein
                            
                            # Normalize by maximum length
                            max_len = max(len(string1), len(string2))
                            if max_len > 0:
                                distance = Levenshtein.distance(string1, string2)
                                similarity = 1.0 - (distance / max_len)
                            else:
                                similarity = 1.0
                        except ImportError:
                            # Fallback if Levenshtein not available
                            similarity = 1.0 if string1 == string2 else 0.0
                    
                    elif metric == 'jaro_winkler':
                        try:
                            from rapidfuzz import fuzz
                            similarity = fuzz.token_sort_ratio(string1, string2) / 100.0
                        except ImportError:
                            # Fallback if rapidfuzz not available
                            similarity = 1.0 if string1 == string2 else 0.0
                    
                    feature_vector[feature_index] = similarity
                    feature_computed[feature_index] = True
        
        # Compute birth/death year match features
        birth_year_match_index = self.feature_names.index("birth_year_match") if "birth_year_match" in self.feature_names else -1
        death_year_match_index = self.feature_names.index("death_year_match") if "death_year_match" in self.feature_names else -1
        has_birth_death_years_index = self.feature_names.index("has_birth_death_years") if "has_birth_death_years" in self.feature_names else -1
        
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
            feature_index = self.feature_names.index(f"{field}_cosine") if f"{field}_cosine" in self.feature_names else -1
            
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
                            try:
                                # Use Levenshtein for person field
                                import Levenshtein
                                max_len = max(len(string1), len(string2))
                                if max_len > 0:
                                    distance = Levenshtein.distance(string1, string2)
                                    similarity = 1.0 - (distance / max_len)
                                else:
                                    similarity = 1.0
                            except ImportError:
                                # Fallback if Levenshtein not available
                                similarity = 1.0 if string1 == string2 else 0.0
                        else:
                            try:
                                # Use token sort ratio for other fields
                                from rapidfuzz import fuzz
                                similarity = fuzz.token_sort_ratio(string1, string2) / 100.0
                            except ImportError:
                                # Fallback if rapidfuzz not available
                                similarity = 1.0 if string1 == string2 else 0.0
                    
                    field_similarities[field] = similarity
                    
                    if feature_index >= 0:
                        feature_vector[feature_index] = similarity
                        feature_computed[feature_index] = True
        
        # Compute interaction features using estimated similarities
        # Harmonic means
        for field_pair in [['person', 'title'], ['person', 'provision'], ['person', 'subjects'],
                           ['title', 'subjects'], ['title', 'provision'], ['provision', 'subjects']]:
            field1, field2 = field_pair
            feature_index = self.feature_names.index(f"{field1}_{field2}_harmonic") if f"{field1}_{field2}_harmonic" in self.feature_names else -1
            
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
            feature_index = self.feature_names.index(f"{field1}_{field2}_product") if f"{field1}_{field2}_product" in self.feature_names else -1
            
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
            feature_index = self.feature_names.index(f"{field1}_{field2}_ratio") if f"{field1}_{field2}_ratio" in self.feature_names else -1
            
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
        for i in range(len(self.feature_names)):
            if not feature_computed[i]:
                feature_vector[i] = 0.0
        
        return feature_vector
    
    def _diagnostic_feature_check(self, feature_vectors):
        """
        Perform comprehensive diagnostic checks on feature vectors to identify quality issues.
        
        Args:
            feature_vectors (numpy.ndarray): Feature vectors to analyze
        """
        if not isinstance(feature_vectors, np.ndarray) or len(feature_vectors) == 0:
            logger.warning("No feature vectors to analyze")
            return
        
        num_vectors = feature_vectors.shape[0]
        num_features = feature_vectors.shape[1]
        logger.info(f"Analyzing {num_vectors} feature vectors with {num_features} features")
        
        # 1. Check for constant features
        constant_features = []
        constant_values = []
        for i in range(num_features):
            first_value = feature_vectors[0, i]
            if np.all(feature_vectors[:, i] == first_value):
                constant_features.append(i)
                constant_values.append(first_value)
        
        if constant_features:
            logger.warning(f"Found {len(constant_features)} constant features")
            
            # If feature names are available, map indices to names
            if len(self.feature_names) == num_features:
                constant_feature_names = [self.feature_names[i] for i in constant_features]
                logger.warning(f"Constant features: {', '.join(constant_feature_names)}")
        
        # 2. Check for NaN/inf values
        nan_features = []
        for i in range(num_features):
            nan_count = np.sum(~np.isfinite(feature_vectors[:, i]))
            if nan_count > 0:
                nan_features.append((i, nan_count))
        
        if nan_features:
            logger.warning(f"Found {len(nan_features)} features with NaN/inf values")
            for idx, count in nan_features:
                feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f"Feature {idx}"
                logger.warning(f"  {feature_name}: {count} NaN/inf values ({count/num_vectors*100:.1f}%)")
        
        # 3. Check feature value distributions
        try:
            # Calculate basic statistics for each feature
            min_values = np.min(feature_vectors, axis=0)
            max_values = np.max(feature_vectors, axis=0)
            mean_values = np.mean(feature_vectors, axis=0)
            std_values = np.std(feature_vectors, axis=0)
            
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
                        feature_name = self.feature_names[i] if i < len(self.feature_names) else f"Feature {i}"
                        suspicious_features.append((feature_name, feature_min, feature_max, feature_mean, feature_std))
            
            if suspicious_features:
                logger.warning(f"Found {len(suspicious_features)} features with suspicious distributions")
                for feature_name, min_val, max_val, mean_val, std_val in suspicious_features[:5]:  # Show first 5
                    logger.warning(f"  {feature_name}: min={min_val:.4f}, max={max_val:.4f}, "
                                 f"mean={mean_val:.4f}, std={std_val:.4f}")
        
        except Exception as e:
            logger.error(f"Error in feature distribution analysis: {e}")
        
        # 4. Print overall statistics
        logger.info(f"Feature vector statistics: "
                  f"min={np.min(feature_vectors):.4f}, "
                  f"max={np.max(feature_vectors):.4f}, "
                  f"mean={np.mean(feature_vectors):.4f}, "
                  f"std={np.std(feature_vectors):.4f}")
