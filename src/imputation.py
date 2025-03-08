"""
Vector-based imputation module for entity resolution.

This module provides the Imputator class, which handles the imputation of
missing values using vector-based hot deck approach.
"""

import os
import logging
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import weaviate
from weaviate.classes.query import Filter, MetadataQuery

from src.utils import save_checkpoint, load_checkpoint, Timer

logger = logging.getLogger(__name__)

class Imputator:
    """
    Handles imputation of missing values using vector-based hot deck approach.
    
    Features:
    - Uses vectors to find similar records for imputation
    - Supports multiple imputation methods (average, weighted average, nearest)
    - Configurable similarity threshold and candidate count
    - Checkpointing for resuming interrupted imputation
    """
    
    def __init__(self, config):
        """
        Initialize the imputator with configuration parameters.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        
        # Imputation parameters
        self.fields_to_impute = config['imputation']['fields_to_impute']
        self.similarity_threshold = config['imputation']['vector_similarity_threshold']
        self.max_candidates = config['imputation']['max_candidates']
        self.imputation_method = config['imputation']['imputation_method']
        
        # Connect to Weaviate
        self.client = self._connect_to_weaviate()
        self.collection_name = config['weaviate']['collection_name']
        
        # Initialize imputed values dictionary
        self.imputed_values = {}
        
        logger.info("Imputator initialized with fields to impute: %s", self.fields_to_impute)

    def execute(self, checkpoint=None):
        """
        Execute imputation of missing values.
        
        Args:
            checkpoint (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Imputation results
        """
        # Load checkpoint if provided
        if checkpoint and os.path.exists(checkpoint):
            state = load_checkpoint(checkpoint)
            self.imputed_values = state.get('imputed_values', {})
            processed_records = set(state.get('processed_records', []))
            logger.info("Resumed imputation from checkpoint: %s", checkpoint)
        else:
            processed_records = set()
        
        # Load record field hashes
        record_field_hashes = self._load_record_field_hashes()
        logger.info(f"Loaded {len(record_field_hashes)} records total")
        
        # Identify records with missing values using enhanced detection that handles hash-based nulls
        records_to_impute = self._identify_missing_values(record_field_hashes)
        
        logger.info(f"Imputing values for {len(records_to_impute)}/{len(record_field_hashes)} records")
        
        if self.config['system']['mode'] == 'dev':
            # In dev mode, limit the number of records to impute
            dev_sample_size = min(self.config['system']['dev_sample_size'], len(records_to_impute))
            record_sample = list(records_to_impute.keys())[:dev_sample_size]
            records_to_impute = {r: records_to_impute[r] for r in record_sample}
            logger.info("Dev mode: limited to %d records", len(records_to_impute))
        
        # Process records
        total_imputed = 0
        total_fields_imputed = 0
        
        with Timer() as timer:
            for record_id, missing_fields in tqdm(records_to_impute.items(), desc="Imputing records"):
                try:
                    # Get record field hashes
                    field_hashes = record_field_hashes[record_id]
                    
                    # Check if composite field is available
                    if 'composite' not in field_hashes or field_hashes['composite'] == 'NULL':
                        logger.warning("Record %s missing composite field, skipping", record_id)
                        continue
                    
                    # Get composite vector
                    composite_hash = field_hashes['composite']
                    composite_vector = self._get_vector_by_hash(composite_hash, 'composite')
                    
                    if not composite_vector:
                        logger.warning("Failed to get composite vector for record %s", record_id)
                        continue
                    
                    # Impute each missing field
                    imputed_fields = 0
                    
                    for field in missing_fields:
                        # Impute value using vector similarity
                        imputed_value, imputed_hash, imputed_vector = self._impute_field(
                            field, composite_vector
                        )
                        
                        if imputed_hash:
                            # Store imputed value
                            if record_id not in self.imputed_values:
                                self.imputed_values[record_id] = {}
                            
                            self.imputed_values[record_id][field] = {
                                'hash': imputed_hash,
                                'value': imputed_value,
                                'similarity': None  # Could store similarity score
                            }
                            
                            imputed_fields += 1
                    
                    if imputed_fields > 0:
                        total_imputed += 1
                        total_fields_imputed += imputed_fields
                    
                    # Update processed records
                    processed_records.add(record_id)
                    
                    # Save checkpoint periodically
                    if self.config['data']['checkpoints_enabled'] and len(processed_records) % 100 == 0:
                        checkpoint_path = Path(self.config['system']['checkpoint_dir']) / f"imputation_{len(processed_records)}.ckpt"
                        save_checkpoint({
                            'imputed_values': self.imputed_values,
                            'processed_records': list(processed_records)
                        }, checkpoint_path)
                
                except Exception as e:
                    logger.error("Error imputing values for record %s: %s", record_id, str(e))
                    
                    # Save checkpoint on error
                    error_checkpoint = Path(self.config['system']['checkpoint_dir']) / f"imputation_error.ckpt"
                    save_checkpoint({
                        'imputed_values': self.imputed_values,
                        'processed_records': list(processed_records)
                    }, error_checkpoint)
                    
                    # Continue with next record
                    continue
        
        # Save final results
        self._save_results()
        
        results = {
            'records_processed': len(processed_records),
            'records_imputed': total_imputed,
            'fields_imputed': total_fields_imputed,
            'duration': timer.duration
        }
        
        logger.info("Imputation completed: %d records imputed, %d fields, %.2f seconds",
                total_imputed, total_fields_imputed, timer.duration)
        
        return results

    def _save_results(self):
        """
        Save imputation results.
        """
        output_dir = Path(self.config['system']['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        # Save imputed values
        with open(output_dir / "imputed_values.json", 'w') as f:
            json.dump(self.imputed_values, f, indent=2)
        
        # Save sample of imputed values
        sample_size = min(100, len(self.imputed_values))
        sample_records = list(self.imputed_values.keys())[:sample_size]
        sample_values = {r: self.imputed_values[r] for r in sample_records}
        
        with open(output_dir / "imputed_values_sample.json", 'w') as f:
            json.dump(sample_values, f, indent=2)
        
        # Save statistics
        field_stats = {}
        for record_id, fields in self.imputed_values.items():
            for field, data in fields.items():
                if field not in field_stats:
                    field_stats[field] = 0
                
                field_stats[field] += 1
        
        with open(output_dir / "imputation_statistics.json", 'w') as f:
            json.dump(field_stats, f, indent=2)
        
        # Save final checkpoint
        checkpoint_path = Path(self.config['system']['checkpoint_dir']) / "imputation_final.ckpt"
        save_checkpoint({
            'imputed_values': self.imputed_values,
            'processed_records': list(self.imputed_values.keys())
        }, checkpoint_path)
        
        logger.info("Imputation results saved to %s", output_dir)

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
        Load record field hashes with support for full dataset.
        """
        try:
            output_dir = Path(self.config['system']['output_dir'])
            mmap_dir = Path(self.config['system']['temp_dir']) / "mmap"
            
            # First try memory-mapped full dataset
            mmap_path = mmap_dir / "record_field_hashes.mmap"
            if mmap_path.exists():
                from src.mmap_dict import MMapDict
                record_field_hashes = MMapDict(mmap_path)
                logger.info(f"Loaded FULL dataset: {len(record_field_hashes)} records from memory-mapped file")
                return record_field_hashes
                
            # Next try record_index.json which points to the full dataset
            record_index_path = output_dir / "record_index.json"
            if record_index_path.exists():
                with open(record_index_path, 'r') as f:
                    record_index = json.load(f)
                
                location = record_index.get('location')
                if location and location != "in-memory" and os.path.exists(location):
                    from src.mmap_dict import MMapDict
                    record_field_hashes = MMapDict(location)
                    logger.info(f"Loaded FULL dataset: {len(record_field_hashes)} records from indexed location")
                    return record_field_hashes
                    
            # Fall back to non-sample file if it exists
            full_path = output_dir / "record_field_hashes.json"
            if full_path.exists():
                with open(full_path, 'r') as f:
                    record_field_hashes = json.load(f)
                logger.info(f"Loaded FULL dataset: {len(record_field_hashes)} records from JSON file")
                return record_field_hashes
                
            # Finally, fall back to sample file with warning
            sample_path = output_dir / "record_field_hashes_sample.json"
            if sample_path.exists():
                with open(sample_path, 'r') as f:
                    record_field_hashes = json.load(f)
                logger.warning(f"WARNING: Only found SAMPLE data with {len(record_field_hashes)} records. This is not the full dataset!")
                return record_field_hashes
                
            logger.error("No record data found! Check preprocessing output.")
            return {}
            
        except Exception as e:
            logger.error(f"Error loading record field hashes: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def _get_vector_by_hash(self, hash_value, field_type):
        """
        Get vector for a hash value and field type.
        
        Args:
            hash_value (str): Hash value
            field_type (str): Field type
            
        Returns:
            list: Vector or None if not found
        """
        try:
            collection = self.client.collections.get(self.collection_name)
            
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

    def _impute_field(self, field, query_vector):
        """
        Impute value for a field using vector similarity with better query settings.
        
        Args:
            field (str): Field to impute
            query_vector (list): Query vector (composite field)
            
        Returns:
            tuple: (imputed_value, imputed_hash, imputed_vector)
        """
        try:
            collection = self.client.collections.get(self.collection_name)
            
            # Create filter for field type
            field_filter = Filter.by_property("field_type").equal(field)
            
            # Execute search with improved parameters - remove distance limit
            # and increase limit to get more candidates
            results = collection.query.near_vector(
                near_vector=query_vector,
                filters=field_filter,
                limit=self.max_candidates * 2,  # Double the limit
                return_metadata=MetadataQuery(distance=True),
                # Remove distance parameter
                # distance=0.3,  
                target_vector=[field],  # Ensure we're targeting the right field
                include_vector=True
            )
            
            logger.info(f"Found {len(results.objects)} initial candidates for field {field}")
            
            if not results.objects:
                logger.warning(f"No candidates found for field {field}")
                return None, None, None
            
            # Filter candidates by similarity threshold
            candidates = []
            
            for obj in results.objects:
                # Convert distance to similarity (1 - distance)
                similarity = 1.0 - obj.metadata.distance
                
                if similarity >= self.similarity_threshold:
                    candidates.append({
                        'hash': obj.properties['hash'],
                        'value': obj.properties['value'],
                        'vector': obj.vector.get(field),
                        'similarity': similarity
                    })
            
            logger.info(f"After filtering: {len(candidates)} candidates above threshold {self.similarity_threshold}")
            
            if not candidates:
                logger.warning(f"No candidates above similarity threshold {self.similarity_threshold} for field {field}")
                return None, None, None
            
            # Rest of the method remains the same...
            if self.imputation_method == 'nearest':
                # Use nearest neighbor
                imputed = candidates[0]
                return imputed['value'], imputed['hash'], imputed['vector']
            
            elif self.imputation_method == 'weighted_average':
                # Compute weighted average of vectors
                weights = np.array([c['similarity'] for c in candidates])
                weights = weights / np.sum(weights)  # Normalize
                
                vectors = np.array([c['vector'] for c in candidates])
                imputed_vector = np.average(vectors, axis=0, weights=weights)
                
                # Use value from highest weight candidate
                best_candidate = candidates[np.argmax(weights)]
                
                return best_candidate['value'], best_candidate['hash'], imputed_vector.tolist()
            
            else:  # 'average' (default)
                # Compute average of vectors
                vectors = np.array([c['vector'] for c in candidates])
                imputed_vector = np.mean(vectors, axis=0)
                
                # Use most frequent value
                value_counts = {}
                for c in candidates:
                    value = c['value']
                    value_counts[value] = value_counts.get(value, 0) + 1
                
                most_frequent_value = max(value_counts.items(), key=lambda x: x[1])[0]
                most_frequent_hash = next(c['hash'] for c in candidates if c['value'] == most_frequent_value)
                
                return most_frequent_value, most_frequent_hash, imputed_vector.tolist()
        
        except Exception as e:
            logger.error(f"Error imputing field {field}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None, None

    def get_imputed_value(self, record_id, field):
        """
        Get imputed value for a record field.
        
        Args:
            record_id (str): Record ID
            field (str): Field name
            
        Returns:
            dict: Imputed value information or None if not found
        """
        if record_id in self.imputed_values and field in self.imputed_values[record_id]:
            return self.imputed_values[record_id][field]
        
        return None

    def _identify_missing_values(self, record_field_hashes):
        """
        Identify records with missing values with enhanced null detection
        including known NULL hash values.
        
        Args:
            record_field_hashes (dict): Dictionary of record ID -> field hashes
            
        Returns:
            dict: Dictionary of record ID -> missing fields
        """
        records_to_impute = {}
        
        # List of possible null representations
        null_values = ['NULL', 'null', '', None, 'None', 'NA', 'N/A', 'none']
        
        # Known NULL hash values (add the hash you discovered)
        null_hash_values = ['132172610905071792854514019103556680276', '289559475738986448570450700985395496420']
        
        # Count field hash frequencies to identify potential NULL values
        field_hash_counts = {}
        for field in self.fields_to_impute:
            field_hash_counts[field] = {}
        
        # Count frequencies
        for record_id, fields in record_field_hashes.items():
            for field in self.fields_to_impute:
                if field in fields:
                    hash_value = fields[field]
                    if hash_value not in field_hash_counts[field]:
                        field_hash_counts[field][hash_value] = 0
                    field_hash_counts[field][hash_value] += 1
        
        # Find most common hash values (potential NULL indicators)
        for field, counts in field_hash_counts.items():
            if counts:
                sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                most_common = sorted_counts[0]
                logger.info(f"Most common hash value for {field}: {most_common[0]} (count: {most_common[1]})")
                
                # If a single hash value appears in more than 15% of records, it's likely a NULL value
                if most_common[1] > len(record_field_hashes) * 0.15:
                    logger.info(f"Potential NULL hash value detected for {field}: {most_common[0]}")
                    if most_common[0] not in null_hash_values:
                        null_hash_values.append(most_common[0])
        
        # Sample a few records to show what we're working with
        sample_records = list(record_field_hashes.items())[:3]
        for record_id, fields in sample_records:
            logger.info(f"Sample record structure: {record_id}: {fields}")
        
        # Log the NULL hash values we're using
        logger.info(f"Using the following hash values to identify NULLs: {null_hash_values}")
        
        # Now check each record
        for record_id, field_hashes in record_field_hashes.items():
            # Check if any field to impute is missing or has a null-like value
            missing_fields = []
            
            for field in self.fields_to_impute:
                # Check if field is missing
                if field not in field_hashes:
                    missing_fields.append(field)
                    continue
                    
                # Check if field has a NULL hash value
                if field_hashes[field] in null_hash_values:
                    missing_fields.append(field)
                    continue
                    
                # Check if field has a null-like string value
                if isinstance(field_hashes[field], str) and field_hashes[field].lower() in null_values:
                    missing_fields.append(field)
                    continue
            
            if missing_fields:
                records_to_impute[record_id] = missing_fields
        
        # Count how many records have missing values for each field
        field_missing_counts = {}
        for field in self.fields_to_impute:
            field_missing_counts[field] = sum(1 for record_id, missing in records_to_impute.items() if field in missing)
        
        logger.info(f"Fields missing values: {field_missing_counts}")
        logger.info(f"Found {len(records_to_impute)} records with missing values using enhanced null detection")
        
        return records_to_impute

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