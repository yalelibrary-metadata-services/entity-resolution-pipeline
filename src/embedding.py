"""
Scalable vector embedding module for large-scale entity resolution.
"""
import os
import logging
import json
import time
import pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import openai
import weaviate
from tenacity import retry, stop_after_attempt, wait_exponential
import mmh3

from src.utils import save_checkpoint, load_checkpoint, get_memory_usage, Timer
from src.mmap_dict import MMapDict

logger = logging.getLogger(__name__)

class Embedder:
    """
    Handles generation of vector embeddings for large datasets.
    
    Features:
    - Scalable approach for millions of strings
    - Memory-mapped storage for embeddings
    - Efficient batching and rate limiting
    - Robust error handling and retry logic
    """
    
    def __init__(self, config):
        """
        Initialize the embedder with configuration parameters.
        """
        self.config = config
        
        # Get OpenAI API key
        self.api_key = os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            logger.warning("OPENAI_API_KEY environment variable not set")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Configure embedding parameters
        self.model = config['embedding']['model']
        self.batch_size = config['embedding']['batch_size']
        self.timeout = config['embedding']['request_timeout']
        
        # Rate limiting parameters
        self.rpm_limit = config['embedding']['rpm_limit']
        self.tpm_limit = config['embedding']['tpm_limit']
        
        # Determine storage approach based on dataset size and mode
        self.use_mmap = self.config['system']['mode'] == 'prod'
        self.mmap_dir = Path(self.config['system']['temp_dir']) / "mmap"
        self.mmap_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize embeddings cache
        self._initialize_storage()
        
        # Rate limiting state
        self.request_timestamps = []
        self.token_counts = []
        
        logger.info("Embedder initialized with model: %s, using %s storage", 
                   self.model,
                   "memory-mapped" if self.use_mmap else "in-memory")

    def _initialize_storage(self):
        """
        Initialize storage based on configuration.
        """
        if self.use_mmap:
            # Use memory-mapped dictionary for embeddings
            self.embeddings = MMapDict(self.mmap_dir / "embeddings.mmap")
        else:
            # Use in-memory dictionary for small datasets
            self.embeddings = {}

    def execute(self, checkpoint=None):
        """Execute embedding generation for large datasets."""
        # Load checkpoint if provided
        if checkpoint and os.path.exists(checkpoint):
            state = load_checkpoint(checkpoint)
            
            # For memory-mapped storage, we need to merge the embeddings
            if self.use_mmap:
                for hash_value, embedding in state.get('embeddings', {}).items():
                    self.embeddings[hash_value] = embedding
            else:
                # For in-memory storage, we can directly assign
                self.embeddings = state.get('embeddings', {})
            
            processed_hashes = set(state.get('processed_hashes', []))
            
            # Load rate limiting state
            self.request_timestamps = state.get('request_timestamps', [])
            self.token_counts = state.get('token_counts', [])
            
            logger.info(f"Resumed embedding from checkpoint: {checkpoint} with {len(processed_hashes)} processed hashes")
        else:
            # Look for existing checkpoint
            checkpoint_dir = Path(self.config['system']['checkpoint_dir'])
            embedding_checkpoint = checkpoint_dir / "embedding_final.ckpt"
            
            if embedding_checkpoint.exists():
                state = load_checkpoint(embedding_checkpoint)
                # Load processed hashes from checkpoint
                processed_hashes = set(state.get('processed_hashes', []))
                logger.info(f"Found existing embedding checkpoint with {len(processed_hashes)} processed hashes")
                
                # For memory-mapped storage, load embeddings
                if self.use_mmap:
                    embeddings_location = state.get('embeddings_location')
                    if embeddings_location and os.path.exists(embeddings_location):
                        logger.info(f"Using existing embeddings from {embeddings_location}")
                        self.embeddings = MMapDict(embeddings_location)
                else:
                    # For in-memory storage, we can directly assign
                    self.embeddings = state.get('embeddings', {})
                    
                # Load rate limiting state
                self.request_timestamps = state.get('request_timestamps', [])
                self.token_counts = state.get('token_counts', [])
            else:
                processed_hashes = set()
        
        # Check if embeddings are already in Weaviate
        if self._check_embeddings_in_weaviate():
            logger.info("Embeddings already indexed in Weaviate, skipping embedding generation")
            return {
                'strings_embedded': len(processed_hashes),
                'total_strings': len(processed_hashes),
                'completion_percentage': 100.0,
                'total_tokens': 0,
                'total_requests': 0,
                'duration': 0.0,
                'skipped': True
            }
        
        # Load hash array or unique strings
        # This approach supports both the enhanced preprocessor and the original one
        hash_array_path = self.mmap_dir / "hash_array.npy"
        if self.use_mmap and hash_array_path.exists():
            # Use memory-mapped hash array for efficient access
            hash_array = np.memmap(
                hash_array_path,
                dtype='U64',
                mode='r'
            )
            all_hashes = [hash_array[i] for i in range(len(hash_array))]
            logger.info("Loaded %d hashes from memory-mapped array", len(all_hashes))
        else:
            # Fall back to loading unique strings from file or index
            unique_strings = self._load_unique_strings()
            all_hashes = list(unique_strings.keys())
            logger.info("Loaded %d hashes from unique strings", len(all_hashes))
        
        # Filter hashes that haven't been processed yet
        hashes_to_process = [h for h in all_hashes if h not in processed_hashes]
        
        # Load field hash mapping to identify field types
        field_hash_mapping = self._load_field_hash_mapping()
        
        # Filter strings to embed
        fields_to_embed = self.config['embedding']['fields_to_embed']
        strings_to_embed = {}
        
        logger.info("Filtering %d hashes to find strings to embed", len(hashes_to_process))
        
        for hash_value in tqdm(hashes_to_process, desc="Filtering strings"):
            # Skip if already processed
            if hash_value in processed_hashes:
                continue
            
            # Look up in field hash mapping
            if hash_value in field_hash_mapping:
                field_types = field_hash_mapping[hash_value].keys()
                if any(field in fields_to_embed for field in field_types):
                    # Get the string value from unique strings
                    if self.use_mmap:
                        unique_strings_mmap = MMapDict(self.mmap_dir / "unique_strings.mmap")
                        if hash_value in unique_strings_mmap:
                            strings_to_embed[hash_value] = unique_strings_mmap[hash_value]
                    else:
                        # For small datasets, we've already loaded unique_strings
                        if hash_value in unique_strings:
                            strings_to_embed[hash_value] = unique_strings[hash_value]
        
        # Log detailed filtering results
        skipped_count = len(hashes_to_process) - len(strings_to_embed)
        logger.info("Filtered %d/%d strings for embedding", len(strings_to_embed), len(hashes_to_process))
        logger.info("Skipped %d strings during filtering:", skipped_count)
        logger.info("  - Already processed: %d", len(all_hashes) - len(hashes_to_process))
        logger.info("  - Not in field mapping or no relevant field types: %d", skipped_count)
        
        if self.config['system']['mode'] == 'dev':
            # In dev mode, limit the number of strings to embed
            dev_sample_size = min(self.config['system']['dev_sample_size'], len(strings_to_embed))
            sample_hashes = list(strings_to_embed.keys())[:dev_sample_size]
            strings_to_embed = {h: strings_to_embed[h] for h in sample_hashes}
            logger.info("Dev mode: limited to %d strings", len(strings_to_embed))
        
        # Organize strings into batches
        batches = self._create_batches(strings_to_embed)
        
        # Process batches
        total_tokens = 0
        total_requests = 0
        
        with Timer() as timer:
            for batch_idx, batch in enumerate(tqdm(batches, desc="Embedding batches")):
                # Check rate limits
                self._wait_for_rate_limit()
                
                # Get batch of strings and their hashes
                batch_strings = [strings_to_embed[h] for h in batch]
                batch_hashes = batch
                
                try:
                    # Generate embeddings for batch
                    batch_embeddings, tokens_used = self._embed_batch(batch_strings)
                    
                    # Update embeddings cache
                    for i, hash_value in enumerate(batch_hashes):
                        self.embeddings[hash_value] = batch_embeddings[i]
                    
                    # Update rate limiting state
                    self.request_timestamps.append(time.time())
                    self.token_counts.append(tokens_used)
                    
                    # Update processed hashes
                    processed_hashes.update(batch_hashes)
                    
                    # Update counters
                    total_tokens += tokens_used
                    total_requests += 1
                    
                    # Log progress
                    if (batch_idx + 1) % 10 == 0:
                        logger.info("Processed %d/%d batches, %d tokens, %d strings", 
                                   batch_idx + 1, len(batches), total_tokens, len(processed_hashes))
                        logger.info("Memory usage: %.2f GB", get_memory_usage())
                    
                    # Save checkpoint periodically
                    if self.config['data']['checkpoints_enabled'] and ((batch_idx + 1) % 50 == 0 or batch_idx == len(batches) - 1):
                        self._save_checkpoint(processed_hashes, batch_idx)
                
                except Exception as e:
                    logger.error("Error embedding batch %d: %s", batch_idx, str(e))
                    
                    # Save checkpoint on error
                    self._save_checkpoint(processed_hashes, f"error_{batch_idx}")
                    
                    # Continue with next batch
                    continue
        
        # Save final results
        self._save_results(processed_hashes)
        
        # Calculate percentage completion
        completion_pct = (len(processed_hashes) / len(all_hashes)) * 100 if all_hashes else 0
        
        results = {
            'strings_embedded': len(processed_hashes),
            'total_strings': len(all_hashes),
            'completion_percentage': completion_pct,
            'total_tokens': total_tokens,
            'total_requests': total_requests,
            'duration': timer.duration
        }
        
        logger.info("Embedding completed: %d strings embedded (%.2f%%), %d tokens, %.2f seconds",
                   len(processed_hashes), completion_pct, total_tokens, timer.duration)
        
        return results

    def _check_embeddings_in_weaviate(self):
        """
        Check if embeddings are already indexed in Weaviate.
        
        Returns:
            bool: True if embeddings are already indexed, False otherwise
        """
        try:
            # Connect to Weaviate
            client = openai.OpenAI(api_key=self.api_key)
            weaviate_client = weaviate.connect_to_local(
                # host=self.config['weaviate']['host'],
                # port=self.config['weaviate']['port']
            )
            
            # Check if collection exists
            collection_name = self.config['weaviate']['collection_name']
            collections = weaviate_client.collections.list_all()
            
            if collection_name not in collections:
                logger.info(f"Collection {collection_name} does not exist in Weaviate")
                return False
            
            # Check if collection has objects
            collection = weaviate_client.collections.get(collection_name)
            count_result = collection.aggregate.over_all(total_count=True)
            
            # If collection exists and has objects, check if it's sufficient
            if count_result.total_count > 0:
                # Get field hash mapping to estimate expected object count
                field_hash_mapping = self._load_field_hash_mapping()
                expected_count = 0
                
                # Estimate expected object count (one object per field type per hash)
                for hash_value, field_types in field_hash_mapping.items():
                    expected_count += len(field_types)
                
                # If we have at least 90% of expected objects, consider embeddings as indexed
                completion_percentage = (count_result.total_count / max(1, expected_count)) * 100
                logger.info(f"Found {count_result.total_count} objects in Weaviate ({completion_percentage:.2f}% of expected)")
                
                return completion_percentage >= 90.0
            
            return False
        
        except Exception as e:
            logger.error(f"Error checking embeddings in Weaviate: {str(e)}")
            return False

    def _load_unique_strings(self):
        """
        Load unique strings with support for both original and enhanced preprocessor.
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
                    # Load from memory-mapped file
                    unique_strings_mmap = MMapDict(location)
                    
                    # Convert to regular dictionary for processing
                    # This is memory-intensive but necessary for large datasets
                    # We'll process in batches in execute() to mitigate this
                    count = index.get('count', 0)
                    logger.info("Loading %d strings from memory-mapped file", count)
                    
                    if count > 1000000:  # If more than 1M strings, we don't fully materialize
                        # Return a proxy that will be used for key checking only
                        return UniqueStringsProxy(unique_strings_mmap)
                    else:
                        return unique_strings_mmap.to_dict()
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

    def _load_field_hash_mapping(self):
        """
        Load field hash mapping with support for both original and enhanced preprocessor.
        """
        try:
            output_dir = Path(self.config['system']['output_dir'])
            
            # Try to load from index first (enhanced preprocessor)
            index_path = output_dir / "field_hash_index.json"
            if index_path.exists():
                with open(index_path, 'r') as f:
                    index = json.load(f)
                
                location = index.get('location')
                
                if location != "in-memory" and os.path.exists(location):
                    # Load from memory-mapped file
                    field_hash_mmap = MMapDict(location)
                    
                    # For field hash mapping, we can afford to materialize
                    # since it's much smaller than the unique strings
                    return field_hash_mmap.to_dict()
                else:
                    # Fall back to sample file
                    with open(output_dir / "field_hash_mapping_sample.json", 'r') as f:
                        return json.load(f)
            else:
                # Fall back to original approach
                with open(output_dir / "field_hash_mapping_sample.json", 'r') as f:
                    return json.load(f)
        
        except Exception as e:
            logger.error("Error loading field hash mapping: %s", str(e))
            return {}

    def _create_batches(self, strings_to_embed):
        """
        Organize strings into batches for embedding.
        """
        batch_size = self.batch_size
        hashes = list(strings_to_embed.keys())
        
        # Create batches
        return [hashes[i:i + batch_size] for i in range(0, len(hashes), batch_size)]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _embed_batch(self, batch_strings):
        """
        Generate embeddings for a batch of strings with retry logic.
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=batch_strings,
                encoding_format="float"
            )
            
            # Extract embeddings
            embeddings = [data.embedding for data in response.data]
            
            # Get token usage
            tokens_used = response.usage.total_tokens
            
            return embeddings, tokens_used
        
        except Exception as e:
            logger.error("Error generating embeddings: %s", str(e))
            raise

    def _wait_for_rate_limit(self):
        """
        Wait if necessary to respect rate limits.
        """
        current_time = time.time()
        
        # Remove old timestamps (older than 1 minute)
        self.request_timestamps = [t for t in self.request_timestamps if current_time - t < 60]
        self.token_counts = self.token_counts[-len(self.request_timestamps):]
        
        # Check requests per minute
        if len(self.request_timestamps) >= self.rpm_limit:
            # Calculate wait time
            oldest_timestamp = min(self.request_timestamps)
            wait_time = 60 - (current_time - oldest_timestamp) + 1  # Add 1 second buffer
            
            logger.info("Rate limit reached, waiting %.2f seconds", wait_time)
            time.sleep(max(0, wait_time))
        
        # Check tokens per minute
        total_tokens = sum(self.token_counts)
        if total_tokens >= self.tpm_limit:
            # Calculate wait time
            wait_time = 60  # Wait a full minute
            
            logger.info("Token limit reached, waiting %.2f seconds", wait_time)
            time.sleep(wait_time)

    def _save_checkpoint(self, processed_hashes, batch_idx):
        """
        Save checkpoint for embedding progress.
        """
        try:
            checkpoint_dir = Path(self.config['system']['checkpoint_dir'])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = checkpoint_dir / f"embedding_{batch_idx}.ckpt"
            
            # For memory-mapped storage, create a lightweight checkpoint
            if self.use_mmap:
                # Flush memory-mapped dictionary
                self.embeddings.flush()
                
                # Create lightweight checkpoint with only processed hashes and state
                lightweight_checkpoint = {
                    'processed_hashes': list(processed_hashes),
                    'request_timestamps': self.request_timestamps,
                    'token_counts': self.token_counts,
                    'embeddings_location': str(self.mmap_dir / "embeddings.mmap"),
                    'embeddings_count': len(processed_hashes)
                }
                
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(lightweight_checkpoint, f)
            else:
                # For in-memory storage, save complete checkpoint
                # but with only the processed embeddings to save space
                processed_embeddings = {h: self.embeddings[h] for h in processed_hashes if h in self.embeddings}
                
                save_checkpoint({
                    'embeddings': processed_embeddings,
                    'processed_hashes': list(processed_hashes),
                    'request_timestamps': self.request_timestamps,
                    'token_counts': self.token_counts
                }, checkpoint_path)
            
            logger.info("Saved embedding checkpoint at %s with %d processed hashes", 
                       checkpoint_path, len(processed_hashes))
        
        except Exception as e:
            logger.error("Error saving checkpoint: %s", str(e))

    def _save_results(self, processed_hashes):
        """
        Save embedding results in a scalable way.
        """
        output_dir = Path(self.config['system']['output_dir'])
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save embedding index
        embedding_index = {
            'count': len(processed_hashes),
            'location': str(self.mmap_dir / "embeddings.mmap") if self.use_mmap else "in-memory",
            'dimensions': self.config['embedding']['dimensions'],
            'model': self.model
        }
        
        with open(output_dir / "embedding_index.json", 'w') as f:
            json.dump(embedding_index, f, indent=2)
        
        # Save list of embedded hashes
        with open(output_dir / "embedded_hashes.json", 'w') as f:
            json.dump(list(processed_hashes)[:10000], f)  # Save first 10K for reference
        
        # For memory-mapped storage, ensure everything is flushed
        if self.use_mmap:
            self.embeddings.flush()
        
        # Create a separate file with embedding info for Weaviate indexing
        embedding_info = {
            'count': len(processed_hashes),
            'dimensions': self.config['embedding']['dimensions'],
            'model': self.model,
            'processed_hashes': list(processed_hashes)[:1000]  # Sample for reference
        }
        
        with open(output_dir / "embedding_info.json", 'w') as f:
            json.dump(embedding_info, f, indent=2)
        
        logger.info("Embedding results saved to %s", output_dir)

    def get_embedding(self, text):
        """
        Get embedding for a single text string.
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
            encoding_format="float"
        )
        
        return response.data[0].embedding

    def get_embedding_by_hash(self, hash_value):
        """
        Get embedding for a hash value.
        """
        if hash_value in self.embeddings:
            return self.embeddings[hash_value]
        
        return None


class UniqueStringsProxy:
    """
    Proxy class for handling large unique strings collections without
    fully materializing them in memory.
    """
    
    def __init__(self, mmap_dict):
        """
        Initialize with a memory-mapped dictionary.
        """
        self.mmap_dict = mmap_dict
    
    def __contains__(self, key):
        """
        Check if key exists in the underlying dictionary.
        """
        return key in self.mmap_dict
    
    def keys(self):
        """
        Return iterator over keys.
        """
        return self.mmap_dict.keys()
    
    def __getitem__(self, key):
        """
        Get value for key.
        """
        return self.mmap_dict[key]
