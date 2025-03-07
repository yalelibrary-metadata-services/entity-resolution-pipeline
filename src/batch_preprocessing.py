"""
Batch preprocessing module for entity resolution.

This module provides the Preprocessor class, which handles data preprocessing
tasks such as normalization, extraction, and deduplication of text fields.
"""

import os
import logging
import csv
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

from src.utils import save_checkpoint, load_checkpoint, get_memory_usage
from src.birth_death_regexes import BirthDeathYearExtractor

logger = logging.getLogger(__name__)

class Preprocessor:
    """
    Handles preprocessing of data for entity resolution.
    
    Preprocessing tasks include:
    - Reading CSV files
    - Normalizing text fields
    - Extracting and deduplicating fields
    - Maintaining hash-based data structures
    - Tracking frequency of duplicate strings
    """
    
    def __init__(self, config):
        """
        Initialize the preprocessor with configuration parameters.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        self.birth_death_extractor = BirthDeathYearExtractor()
        
        # Initialize data structures
        self.unique_strings = {}  # Hash -> string value
        self.string_counts = {}   # Hash -> count of occurrences
        self.record_field_hashes = {}  # Record ID -> {field -> hash}
        self.field_hash_mapping = {}   # Hash -> {field1: count1, field2: count2, ...}
        
        # Use memory-mapped dictionaries for large datasets in production mode
        if self.config['system']['mode'] == 'prod':
            self.use_mmap = True
            self.mmap_dir = Path(self.config['system']['temp_dir']) / "mmap"
            self.mmap_dir.mkdir(exist_ok=True, parents=True)
            
            # Initialize memory-mapped dictionaries
            self.unique_strings = MMapDict(self.mmap_dir / "unique_strings.mmap")
            self.string_counts = MMapDict(self.mmap_dir / "string_counts.mmap")
        else:
            self.use_mmap = False
        
        logger.info("Preprocessor initialized with mode: %s", self.config['system']['mode'])

    def execute(self, checkpoint=None):
        """
        Execute preprocessing tasks.
        
        Args:
            checkpoint (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Preprocessing results
        """
        # Load checkpoint if provided
        if checkpoint and os.path.exists(checkpoint):
            state = load_checkpoint(checkpoint)
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
                
                # Save checkpoint
                if self.config['data']['checkpoints_enabled']:
                    checkpoint_path = Path(self.config['system']['checkpoint_dir']) / f"preprocess_{input_file.stem}.ckpt"
                    save_checkpoint({
                        'unique_strings': self.unique_strings,
                        'string_counts': self.string_counts,
                        'record_field_hashes': self.record_field_hashes,
                        'field_hash_mapping': self.field_hash_mapping,
                        'processed_files': processed_files
                    }, checkpoint_path)
                
                logger.info("Processed file: %s, records: %d", input_file.name, len(records))
                logger.info("Memory usage: %.2f GB", get_memory_usage())
        
        # Save final results
        self._save_results()
        
        results = {
            'records_processed': total_records,
            'unique_strings': len(self.unique_strings),
            'unique_records': len(self.record_field_hashes)
        }
        
        logger.info("Preprocessing completed: %d records processed, %d unique strings",
                   total_records, len(self.unique_strings))
        
        return results

    def _read_file_in_batches(self, file_path, batch_size):
        """
        Read CSV file in efficient batches using pandas.
        
        Args:
            file_path (Path): Path to CSV file
            batch_size (int): Number of records per batch
            
        Yields:
            list: Batch of records
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
        
        Args:
            batch (list): List of record dictionaries
            
        Returns:
            dict: Processed batch data
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
                
                # Handle null values
                if not value or value == 0.0:
                    field_hashes[field] = 'NULL'
                    continue
                
                # Normalize value
                normalized_value = self._normalize_text(value)
                
                # Generate hash
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
        Update data structures with batch results.
        
        Args:
            batch_data (dict): Processed batch data
        """
        # Update unique strings
        for value_hash, value in batch_data['unique_strings'].items():
            if value_hash not in self.unique_strings:
                self.unique_strings[value_hash] = value
        
        # Update string counts
        for value_hash, count in batch_data['string_counts'].items():
            self.string_counts[value_hash] = self.string_counts.get(value_hash, 0) + count
        
        # Update record field hashes
        self.record_field_hashes.update(batch_data['record_field_hashes'])
        
        # Update field hash mapping
        for value_hash, field_counts in batch_data['field_hash_mapping'].items():
            if value_hash not in self.field_hash_mapping:
                self.field_hash_mapping[value_hash] = {}
            
            for field, count in field_counts.items():
                if field not in self.field_hash_mapping[value_hash]:
                    self.field_hash_mapping[value_hash][field] = 0
                
                self.field_hash_mapping[value_hash][field] += count

    def _normalize_text(self, text):
        """
        Normalize text value.
        
        Args:
            text: Value to normalize (could be string, float, int, etc.)
            
        Returns:
            str: Normalized text
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
        Generate hash for a string.
        
        Args:
            text (str): String to hash
            
        Returns:
            str: Hash value
        """
        # Using MD5 for consistency and to avoid collisions
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _save_results(self):
        """
        Save preprocessing results to disk.
        """
        output_dir = Path(self.config['system']['output_dir'])
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save unique strings sample (first 1000)
        unique_strings_sample = {k: self.unique_strings[k] for k in list(self.unique_strings)[:1000]}
        with open(output_dir / "unique_strings_sample.json", 'w') as f:
            json.dump(unique_strings_sample, f, indent=2)
        
        # Save string counts sample
        string_counts_sample = {k: self.string_counts[k] for k in list(self.string_counts)[:1000]}
        with open(output_dir / "string_counts_sample.json", 'w') as f:
            json.dump(string_counts_sample, f, indent=2)
        
        # Save field hash mapping sample
        field_hash_mapping_sample = {k: self.field_hash_mapping[k] for k in list(self.field_hash_mapping)[:1000]}
        with open(output_dir / "field_hash_mapping_sample.json", 'w') as f:
            json.dump(field_hash_mapping_sample, f, indent=2)
        
        # Save record field hashes sample
        record_field_hashes_sample = {k: self.record_field_hashes[k] for k in list(self.record_field_hashes)[:1000]}
        with open(output_dir / "record_field_hashes_sample.json", 'w') as f:
            json.dump(record_field_hashes_sample, f, indent=2)
        
        # Save statistics
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
        
        # If using memory-mapped dictionaries, make sure they're fully flushed
        if self.use_mmap:
            self.unique_strings.flush()
            self.string_counts.flush()
        
        logger.info("Preprocessing results saved to %s", output_dir)

    def get_birth_death_years(self, person_name):
        """
        Extract birth and death years from a person name string.
        
        Args:
            person_name (str): Person name string
            
        Returns:
            tuple: (birth_year, death_year) or (None, None) if not found
        """
        return self.birth_death_extractor.parse(person_name)
