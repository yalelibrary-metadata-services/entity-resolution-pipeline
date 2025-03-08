"""
Enhanced preprocessing module for large-scale entity resolution.
"""
import os
import logging
import json
import hashlib
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import mmh3
from tqdm import tqdm
import numpy as np
from src.mmap_dict import MMapDict
import pickle

from src.utils import save_checkpoint, load_checkpoint, get_memory_usage
from src.birth_death_regexes import BirthDeathYearExtractor

logger = logging.getLogger(__name__)

class Preprocessor:
    """
    Enhanced preprocessor for large-scale entity resolution.
    """
    
    def __init__(self, config):
        """
        Initialize the preprocessor with configuration parameters.
        """
        self.config = config
        self.birth_death_extractor = BirthDeathYearExtractor()
        
        # Determine storage approach based on dataset size and mode
        self.use_mmap = self.config['system']['mode'] == 'prod'
        self.mmap_dir = Path(self.config['system']['temp_dir']) / "mmap"
        self.mmap_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize data structures
        self._initialize_storage()
        
        logger.info("Preprocessor initialized with mode: %s, using %s storage", 
                   self.config['system']['mode'], 
                   "memory-mapped" if self.use_mmap else "in-memory")

    def _initialize_storage(self):
        """
        Initialize storage based on configuration.
        """
        if self.use_mmap:
            # Use memory-mapped dictionaries for large datasets
            self.unique_strings = MMapDict(self.mmap_dir / "unique_strings.mmap")
            self.string_counts = MMapDict(self.mmap_dir / "string_counts.mmap")
            self.record_field_hashes = MMapDict(self.mmap_dir / "record_field_hashes.mmap")
            self.field_hash_mapping = MMapDict(self.mmap_dir / "field_hash_mapping.mmap")
        else:
            # Use in-memory dictionaries for small datasets
            self.unique_strings = {}
            self.string_counts = {}
            self.record_field_hashes = {}
            self.field_hash_mapping = {}

    def execute(self, checkpoint=None):
        """
        Execute preprocessing tasks with scalable approach.
        """
        # Load checkpoint if provided
        if checkpoint and os.path.exists(checkpoint):
            state = load_checkpoint(checkpoint)
            if self.use_mmap:
                # For memory-mapped storage, we need to load the data
                for key, value in state.get('unique_strings', {}).items():
                    self.unique_strings[key] = value
                for key, value in state.get('string_counts', {}).items():
                    self.string_counts[key] = value
                for key, value in state.get('record_field_hashes', {}).items():
                    self.record_field_hashes[key] = value
                for key, value in state.get('field_hash_mapping', {}).items():
                    self.field_hash_mapping[key] = value
            else:
                # For in-memory storage, we can directly assign
                self.unique_strings = state.get('unique_strings', {})
                self.string_counts = state.get('string_counts', {})
                self.record_field_hashes = state.get('record_field_hashes', {})
                self.field_hash_mapping = state.get('field_hash_mapping', {})
            
            processed_files = state.get('processed_files', [])
            logger.info("Resumed preprocessing from checkpoint: %s", checkpoint)
        else:
            processed_files = []
        
        # Get input files
        input_dir = Path(self.config['data']['input_dir'])
        input_files = sorted([f for f in input_dir.glob('*.csv') if f.name not in processed_files])
        
        if self.config['system']['mode'] == 'dev':
            # In dev mode, use a limited number of files
            dev_sample = min(5, len(input_files))
            input_files = input_files[:dev_sample]
        
        logger.info("Processing %d input files", len(input_files))
        
        # Process files in parallel
        total_records = 0
        batch_size = self.config['system']['batch_size']
        max_workers = self.config['system']['max_workers']
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for input_file in tqdm(input_files, desc="Processing files"):
                # Process file in batches
                batch_results = []
                records = []
                
                for batch in self._read_file_in_batches(input_file, batch_size):
                    batch_results.append(
                        executor.submit(self._process_batch, batch)
                    )
                    records.extend(batch)
                
                # Update data structures with batch results
                for future in tqdm(batch_results, desc=f"Processing batches in {input_file.name}"):
                    batch_data = future.result()
                    self._update_data_structures(batch_data)
                
                total_records += len(records)
                processed_files.append(input_file.name)
                
                # Save checkpoint with progress information
                if self.config['data']['checkpoints_enabled'] and (len(processed_files) % 5 == 0 or len(processed_files) == len(input_files)):
                    checkpoint_path = Path(self.config['system']['checkpoint_dir']) / f"preprocess_{len(processed_files)}files.ckpt"
                    
                    # For memory-mapped storage, we create a lightweight checkpoint
                    # that only contains the processed files and statistics
                    if self.use_mmap:
                        # Flush memory-mapped dictionaries
                        self.unique_strings.flush()
                        self.string_counts.flush()
                        self.record_field_hashes.flush()
                        self.field_hash_mapping.flush()
                        
                        # Create lightweight checkpoint
                        lightweight_checkpoint = {
                            'processed_files': processed_files,
                            'stats': {
                                'unique_strings': len(self.unique_strings),
                                'records': len(self.record_field_hashes)
                            }
                        }
                        save_checkpoint(lightweight_checkpoint, checkpoint_path)
                    else:
                        # Create full checkpoint
                        save_checkpoint({
                            'unique_strings': self.unique_strings,
                            'string_counts': self.string_counts,
                            'record_field_hashes': self.record_field_hashes,
                            'field_hash_mapping': self.field_hash_mapping,
                            'processed_files': processed_files
                        }, checkpoint_path)
                
                logger.info("Processed file: %s, records: %d, unique strings: %d", 
                          input_file.name, len(records), len(self.unique_strings))
                logger.info("Memory usage: %.2f GB", get_memory_usage())
        
        # Save final results and create necessary index files
        self._save_results()
        self._create_lookup_tables()
        
        results = {
            'records_processed': total_records,
            'unique_strings': len(self.unique_strings),
            'unique_records': len(self.record_field_hashes)
        }
        
        # Add this before saving results
        logger.info(f"String count before deduplication: {len(self.string_counts)}")
        # Create a set of unique normalized texts to check true uniqueness
        unique_values = set(self.unique_strings.values())
        logger.info(f"True unique string count based on values: {len(unique_values)}")

        logger.info("Preprocessing completed: %d records processed, %d unique strings",
                   total_records, len(self.unique_strings))
        
        return results

    def _save_results(self):
        """
        Save preprocessing results in a scalable way.
        """
        output_dir = Path(self.config['system']['output_dir'])
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Helper function to convert MMapDict to regular dict
        def convert_to_regular_dict(mmap_dict):
            if not isinstance(mmap_dict, dict) and not hasattr(mmap_dict, 'items'):
                return mmap_dict
            
            regular_dict = {}
            for k, v in mmap_dict.items():
                if isinstance(v, dict) or hasattr(v, 'items'):
                    regular_dict[k] = convert_to_regular_dict(v)
                else:
                    regular_dict[k] = v
            return regular_dict
        
        # Create a lightweight index file that contains references
        # to memory-mapped data rather than copying everything
        
        # Save unique strings index
        unique_strings_index = {
            'count': len(self.unique_strings),
            'location': str(self.mmap_dir / "unique_strings.mmap") if self.use_mmap else "in-memory",
            'sample': convert_to_regular_dict({k: self.unique_strings[k] for k in list(self.unique_strings.keys())[:100]})
        }
        with open(output_dir / "unique_strings_index.json", 'w') as f:
            json.dump(unique_strings_index, f)
        
        # Save string counts index
        string_counts_index = {
            'count': len(self.string_counts),
            'location': str(self.mmap_dir / "string_counts.mmap") if self.use_mmap else "in-memory",
            'sample': convert_to_regular_dict({k: self.string_counts[k] for k in list(self.string_counts.keys())[:100]})
        }
        with open(output_dir / "string_counts_index.json", 'w') as f:
            json.dump(string_counts_index, f)
        
        # Save record field hashes index
        record_index = {
            'count': len(self.record_field_hashes),
            'location': str(self.mmap_dir / "record_field_hashes.mmap") if self.use_mmap else "in-memory",
            'sample': convert_to_regular_dict({k: self.record_field_hashes[k] for k in list(self.record_field_hashes.keys())[:100]})
        }
        with open(output_dir / "record_index.json", 'w') as f:
            json.dump(record_index, f)
        
        # Save field hash mapping index
        field_hash_index = {
            'count': len(self.field_hash_mapping),
            'location': str(self.mmap_dir / "field_hash_mapping.mmap") if self.use_mmap else "in-memory",
            'sample': convert_to_regular_dict({k: self.field_hash_mapping[k] for k in list(self.field_hash_mapping.keys())[:100]})
        }
        with open(output_dir / "field_hash_index.json", 'w') as f:
            json.dump(field_hash_index, f)
        
        # Additionally, save FULL data to JSON for direct access by downstream components
        # This is essential for smaller datasets or components that don't support memory-mapped files
        if len(self.record_field_hashes) < 10000:  # Only for reasonably sized datasets
            full_dict = convert_to_regular_dict(self.record_field_hashes)
            with open(output_dir / "record_field_hashes.json", 'w') as f:
                json.dump(full_dict, f)
            logger.info(f"Saved FULL record data ({len(self.record_field_hashes)} records) to JSON for direct access")
        
        # Save samples for reference and compatibility
        unique_strings_sample = convert_to_regular_dict({k: self.unique_strings[k] for k in list(self.unique_strings.keys())[:1000]})
        with open(output_dir / "unique_strings_sample.json", 'w') as f:
            json.dump(unique_strings_sample, f)
        
        string_counts_sample = convert_to_regular_dict({k: self.string_counts[k] for k in list(self.string_counts.keys())[:1000]})
        with open(output_dir / "string_counts_sample.json", 'w') as f:
            json.dump(string_counts_sample, f)
        
        field_hash_mapping_sample = convert_to_regular_dict({k: self.field_hash_mapping[k] for k in list(self.field_hash_mapping.keys())[:1000]})
        with open(output_dir / "field_hash_mapping_sample.json", 'w') as f:
            json.dump(field_hash_mapping_sample, f)
        
        record_field_hashes_sample = convert_to_regular_dict({k: self.record_field_hashes[k] for k in list(self.record_field_hashes.keys())[:1000]})
        with open(output_dir / "record_field_hashes_sample.json", 'w') as f:
            json.dump(record_field_hashes_sample, f)
        
        # For large datasets, create a memory-mapped numpy array to store the hashes
        # This will allow efficient access to the hashes for embedding
        if self.use_mmap and len(self.unique_strings) > 10000:
            self._create_hash_array()
        
        # Flush memory-mapped dictionaries
        if self.use_mmap:
            self.unique_strings.flush()
            self.string_counts.flush()
            self.record_field_hashes.flush()
            self.field_hash_mapping.flush()
        
        logger.info(f"Preprocessing results saved to {output_dir}")

    def _read_file_in_batches(self, file_path, batch_size):
        """
        Read CSV file in efficient batches using pandas.
        """
        try:
            # Use pandas with chunking for more efficient reading
            for chunk in pd.read_csv(file_path, chunksize=batch_size):
                # Convert chunk to list of dictionaries
                records = chunk.to_dict('records')
                yield records
        
        except Exception as e:
            logger.error("Error reading file %s: %s", file_path, str(e))
            yield []

    def _process_batch(self, batch):
        """
        Process a batch of records.
        """
        batch_unique_strings = {}
        batch_string_counts = {}
        batch_record_field_hashes = {}
        batch_field_hash_mapping = {}
        
        for record in batch:
            person_id = record.get('personId', 'unknown')
            
            if not person_id or person_id == 'unknown':
                continue
            
            # Process each field
            field_hashes = {}
            
            for field in ['composite', 'person', 'roles', 'title', 'provision', 'subjects']:
                value = record.get(field, '')
                
                # Handle null values - make sure empty strings are marked as NULL
                if value is None or value == '' or value == 0.0:
                    field_hashes[field] = 'NULL'
                    continue
                
                # Normalize value
                normalized_value = self._normalize_text(value)
                
                # Generate hash - use MurmurHash for better performance
                value_hash = self._hash_string(normalized_value)
                field_hashes[field] = value_hash
                
                # Update unique strings and counts
                if value_hash not in batch_unique_strings:
                    batch_unique_strings[value_hash] = normalized_value
                
                batch_string_counts[value_hash] = batch_string_counts.get(value_hash, 0) + 1
                
                # Update field hash mapping
                if value_hash not in batch_field_hash_mapping:
                    batch_field_hash_mapping[value_hash] = {}
                
                if field not in batch_field_hash_mapping[value_hash]:
                    batch_field_hash_mapping[value_hash][field] = 0
                
                batch_field_hash_mapping[value_hash][field] += 1
            
            # Store record field hashes
            batch_record_field_hashes[person_id] = field_hashes
        
        return {
            'unique_strings': batch_unique_strings,
            'string_counts': batch_string_counts,
            'record_field_hashes': batch_record_field_hashes,
            'field_hash_mapping': batch_field_hash_mapping
        }


    def _update_data_structures(self, batch_data):
        """
        Update data structures with batch results with improved deduplication.
        """
        # Create value-to-hash mapping for consistent hashing
        value_to_hash = {}
        
        # First pass: build value_to_hash mapping 
        for value_hash, value in batch_data['unique_strings'].items():
            normalized_value = self._normalize_text(value)
            if normalized_value not in value_to_hash:
                value_to_hash[normalized_value] = value_hash
            else:
                # We found a hash collision - use the existing hash for this value
                # This is the key part that ensures consistent hashing
                logger.debug(f"Hash collision detected: '{normalized_value}' has multiple hashes")
        
        # Update unique strings using consistent hashing
        for normalized_value, value_hash in value_to_hash.items():
            if value_hash not in self.unique_strings:
                # Find the original value for this normalized value
                original_value = next((val for h, val in batch_data['unique_strings'].items() 
                                    if self._normalize_text(val) == normalized_value), normalized_value)
                self.unique_strings[value_hash] = original_value
        
        # Update string counts
        for value_hash, count in batch_data['string_counts'].items():
            normalized_value = self._normalize_text(batch_data['unique_strings'].get(value_hash, ''))
            consistent_hash = value_to_hash.get(normalized_value, value_hash)
            self.string_counts[consistent_hash] = self.string_counts.get(consistent_hash, 0) + count
        
        # Update record field hashes with consistent hashing
        for record_id, field_hashes in batch_data['record_field_hashes'].items():
            consistent_field_hashes = {}
            for field, value_hash in field_hashes.items():
                if value_hash == 'NULL':
                    consistent_field_hashes[field] = 'NULL'
                    continue
                    
                if value_hash in batch_data['unique_strings']:
                    normalized_value = self._normalize_text(batch_data['unique_strings'][value_hash])
                    consistent_hash = value_to_hash.get(normalized_value, value_hash)
                    consistent_field_hashes[field] = consistent_hash
                else:
                    consistent_field_hashes[field] = value_hash
            
            self.record_field_hashes[record_id] = consistent_field_hashes
        
        # Update field hash mapping with consistent hashing
        for value_hash, field_counts in batch_data['field_hash_mapping'].items():
            if value_hash in batch_data['unique_strings']:
                normalized_value = self._normalize_text(batch_data['unique_strings'][value_hash])
                consistent_hash = value_to_hash.get(normalized_value, value_hash)
                
                if consistent_hash not in self.field_hash_mapping:
                    self.field_hash_mapping[consistent_hash] = {}
                
                for field, count in field_counts.items():
                    if field not in self.field_hash_mapping[consistent_hash]:
                        self.field_hash_mapping[consistent_hash][field] = 0
                    
                    self.field_hash_mapping[consistent_hash][field] += count
            else:
                # Fall back to original behavior for hashes not in unique_strings
                if value_hash not in self.field_hash_mapping:
                    self.field_hash_mapping[value_hash] = {}
                
                for field, count in field_counts.items():
                    if field not in self.field_hash_mapping[value_hash]:
                        self.field_hash_mapping[value_hash][field] = 0
                    
                    self.field_hash_mapping[value_hash][field] += count

    def _normalize_text(self, text):
        """
        Normalize text value.
        """
        if text is None:
            return ""
        
        # Convert to string if not already
        if not isinstance(text, str):
            text = str(text)
        
        # Basic normalization
        normalized = text.strip()
        
        # Remove excessive whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized

    def _hash_string(self, text):
        """
        Generate consistent hash for a string using MurmurHash.
        """
        # Normalize the text thoroughly for consistent hashing
        if text is None:
            return "NULL"
            
        # Convert to string if not already
        if not isinstance(text, str):
            text = str(text)
        
        # Apply thorough normalization
        normalized = text.strip().lower()
        
        # Remove all whitespace variations 
        import re
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Final trim
        normalized = normalized.strip()
        
        # Use MurmurHash for better performance and consistency
        # Use fixed seed for reproducibility
        return str(mmh3.hash128(normalized, seed=42))

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

    def _create_hash_array(self):
        """
        Create a memory-mapped array of hashes for efficient access.
        """
        hash_array_path = self.mmap_dir / "hash_array.npy"
        
        # Get list of hashes
        hashes = list(self.unique_strings.keys())
        
        # Create memory-mapped array
        hash_array = np.memmap(
            hash_array_path, 
            dtype='U64',  # Unicode string of max length 64
            mode='w+',
            shape=(len(hashes),)
        )
        
        # Fill array with hashes
        for i, hash_value in enumerate(hashes):
            hash_array[i] = hash_value
        
        # Flush to disk
        hash_array.flush()
        
        # Save index mapping
        hash_to_index = {hash_value: i for i, hash_value in enumerate(hashes)}
        
        with open(self.mmap_dir / "hash_to_index.pkl", 'wb') as f:
            pickle.dump(hash_to_index, f)
        
        logger.info("Created hash array with %d hashes", len(hashes))

    def _create_lookup_tables(self):
        """
        Create lookup tables for efficient access with improved field-hash mapping.
        """
        output_dir = Path(self.config['system']['output_dir'])
        
        # Create personId -> field hash lookup table
        person_id_lookup = {}
        
        for record_id, field_hashes in self.record_field_hashes.items():
            person_hash = field_hashes.get('person', 'NULL')
            if person_hash != 'NULL':
                person_id_lookup[record_id] = person_hash
        
        with open(output_dir / "person_id_lookup.json", 'w') as f:
            # For large datasets, save only a sample
            if len(person_id_lookup) > 10000 and self.use_mmap:
                sample = {k: person_id_lookup[k] for k in list(person_id_lookup.keys())[:10000]}
                json.dump(sample, f)
                
                # Save full lookup to memory-mapped file
                person_id_mmap = MMapDict(self.mmap_dir / "person_id_lookup.mmap")
                for k, v in person_id_lookup.items():
                    person_id_mmap[k] = v
                person_id_mmap.flush()
            else:
                json.dump(person_id_lookup, f)
        
        # IMPORTANT FIX: Rebuild field_hash_mapping from record_field_hashes to ensure consistency
        logger.info("Rebuilding field_hash_mapping from record_field_hashes for consistency")
        rebuilt_field_hash_mapping = {}
        
        # Iterate through all record field hashes to build complete field mapping
        for record_id, field_hashes in self.record_field_hashes.items():
            for field, hash_value in field_hashes.items():
                if hash_value == 'NULL':
                    continue
                    
                if hash_value not in rebuilt_field_hash_mapping:
                    rebuilt_field_hash_mapping[hash_value] = {}
                
                if field not in rebuilt_field_hash_mapping[hash_value]:
                    rebuilt_field_hash_mapping[hash_value][field] = 0
                
                rebuilt_field_hash_mapping[hash_value][field] += 1
        
        # Compare original and rebuilt mappings
        original_hash_count = len(self.field_hash_mapping)
        rebuilt_hash_count = len(rebuilt_field_hash_mapping)
        
        added_hashes = set(rebuilt_field_hash_mapping.keys()) - set(self.field_hash_mapping.keys())
        missing_hashes = set(self.field_hash_mapping.keys()) - set(rebuilt_field_hash_mapping.keys())
        
        logger.info(f"Original field_hash_mapping had {original_hash_count} hashes")
        logger.info(f"Rebuilt field_hash_mapping has {rebuilt_hash_count} hashes")
        logger.info(f"Added {len(added_hashes)} missing hashes")
        logger.info(f"Removed {len(missing_hashes)} invalid hashes")
        
        # Log a few examples of fixed hashes
        for hash_value in list(added_hashes)[:5]:
            logger.info(f"Fixed missing hash: {hash_value} with fields {rebuilt_field_hash_mapping[hash_value]}")
        
        # Update the field_hash_mapping with the rebuilt version
        self.field_hash_mapping = rebuilt_field_hash_mapping
        
        # Create field statistics for analysis
        field_stats = {}
        for value_hash, field_counts in self.field_hash_mapping.items():
            for field, count in field_counts.items():
                if field not in field_stats:
                    field_stats[field] = {
                        'total_occurrences': 0,
                        'unique_values': 0
                    }
                
                field_stats[field]['total_occurrences'] += count
                field_stats[field]['unique_values'] += 1
        
        with open(output_dir / "field_statistics.json", 'w') as f:
            json.dump(field_stats, f, indent=2)
        
        logger.info("Created lookup tables and statistics")
        logger.info(f"Field statistics: {field_stats}")

    def get_hash_array(self):
        """
        Get memory-mapped hash array for efficient access.
        """
        if self.use_mmap:
            hash_array_path = self.mmap_dir / "hash_array.npy"
            if hash_array_path.exists():
                return np.memmap(
                    hash_array_path,
                    dtype='U64',
                    mode='r',
                    shape=(len(self.unique_strings),)
                )
        
        # Fallback to list for in-memory mode
        return list(self.unique_strings.keys())

    def get_birth_death_years(self, person_name):
        """
        Extract birth and death years from a person name string.
        """
        return self.birth_death_extractor.parse(person_name)
