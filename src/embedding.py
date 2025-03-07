"""
Vector embedding module for entity resolution.

This module provides the Embedder class, which handles the generation of
vector embeddings for unique strings using OpenAI's text-embedding model.
"""

import os
import logging
import json
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils import save_checkpoint, load_checkpoint, get_memory_usage, Timer

logger = logging.getLogger(__name__)

class Embedder:
    """
    Handles generation of vector embeddings for unique strings.
    
    Features:
    - Batched embedding requests for efficiency
    - Rate limiting to respect API constraints
    - Checkpointing for resuming interrupted embedding
    - Caching of embeddings to avoid redundant API calls
    """
    
    def __init__(self, config):
        """
        Initialize the embedder with configuration parameters.
        
        Args:
            config (dict): Configuration parameters
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
        
        # Initialize embeddings cache
        self.embeddings = {}
        
        # Rate limiting state
        self.request_timestamps = []
        self.token_counts = []
        
        logger.info("Embedder initialized with model: %s", self.model)

    def execute(self, checkpoint=None):
        """
        Execute embedding generation for unique strings.
        
        Args:
            checkpoint (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Embedding results
        """
        # Load checkpoint if provided
        if checkpoint and os.path.exists(checkpoint):
            state = load_checkpoint(checkpoint)
            self.embeddings = state.get('embeddings', {})
            processed_hashes = set(state.get('processed_hashes', []))
            logger.info("Resumed embedding from checkpoint: %s", checkpoint)
        else:
            processed_hashes = set()
        
        # Load unique strings from preprocessing results
        unique_strings = self._load_unique_strings()
        
        # Filter fields to embed
        fields_to_embed = self.config['embedding']['fields_to_embed']
        
        # Load field hash mapping to identify field types
        field_hash_mapping = self._load_field_hash_mapping()
        
        # Filter strings to embed
        strings_to_embed = {}
        for hash_value, string_value in unique_strings.items():
            # Skip if already processed
            if hash_value in processed_hashes:
                continue
            
            # Skip if not in field hash mapping (shouldn't happen)
            if hash_value not in field_hash_mapping:
                logger.warning("Hash %s not found in field hash mapping", hash_value)
                continue
            
            # Check if any field type is in fields_to_embed
            field_types = field_hash_mapping[hash_value].keys()
            if any(field in fields_to_embed for field in field_types):
                strings_to_embed[hash_value] = string_value
        
        logger.info("Embedding %d/%d unique strings", len(strings_to_embed), len(unique_strings))
        
        if self.config['system']['mode'] == 'dev':
            # In dev mode, limit the number of strings to embed
            dev_sample_size = min(self.config['system']['dev_sample_size'], len(strings_to_embed))
            hash_sample = list(strings_to_embed.keys())[:dev_sample_size]
            strings_to_embed = {h: strings_to_embed[h] for h in hash_sample}
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
                        logger.info("Processed %d/%d batches, %d tokens", 
                                   batch_idx + 1, len(batches), total_tokens)
                        logger.info("Memory usage: %.2f GB", get_memory_usage())
                    
                    # Save checkpoint
                    if self.config['data']['checkpoints_enabled'] and (batch_idx + 1) % 100 == 0:
                        checkpoint_path = Path(self.config['system']['checkpoint_dir']) / f"embedding_{batch_idx}.ckpt"
                        save_checkpoint({
                            'embeddings': self.embeddings,
                            'processed_hashes': list(processed_hashes)
                        }, checkpoint_path)
                
                except Exception as e:
                    logger.error("Error embedding batch %d: %s", batch_idx, str(e))
                    
                    # Save checkpoint on error
                    error_checkpoint = Path(self.config['system']['checkpoint_dir']) / f"embedding_error_{batch_idx}.ckpt"
                    save_checkpoint({
                        'embeddings': self.embeddings,
                        'processed_hashes': list(processed_hashes)
                    }, error_checkpoint)
                    
                    # Continue with next batch
                    continue
        
        # Save final results
        self._save_results(processed_hashes)
        
        results = {
            'strings_embedded': len(processed_hashes),
            'total_tokens': total_tokens,
            'total_requests': total_requests,
            'duration': timer.duration
        }
        
        logger.info("Embedding completed: %d strings embedded, %d tokens, %.2f seconds",
                   len(processed_hashes), total_tokens, timer.duration)
        
        return results

    def _load_unique_strings(self):
        """
        Load unique strings from preprocessing results.
        
        Returns:
            dict: Dictionary of hash -> string value
        """
        try:
            # In a real implementation, this would load from the preprocessor's output
            # For now, just load from a sample file
            output_dir = Path(self.config['system']['output_dir'])
            with open(output_dir / "unique_strings_sample.json", 'r') as f:
                unique_strings_sample = json.load(f)
            
            return unique_strings_sample
        
        except Exception as e:
            logger.error("Error loading unique strings: %s", str(e))
            return {}

    def _load_field_hash_mapping(self):
        """
        Load field hash mapping from preprocessing results.
        
        Returns:
            dict: Dictionary of hash -> {field -> count}
        """
        try:
            output_dir = Path(self.config['system']['output_dir'])
            with open(output_dir / "field_hash_mapping_sample.json", 'r') as f:
                field_hash_mapping_sample = json.load(f)
            
            return field_hash_mapping_sample
        
        except Exception as e:
            logger.error("Error loading field hash mapping: %s", str(e))
            return {}

    def _create_batches(self, strings_to_embed):
        """
        Organize strings into batches for embedding.
        
        Args:
            strings_to_embed (dict): Dictionary of hash -> string value
            
        Returns:
            list: List of batches, where each batch is a list of hashes
        """
        batch_size = self.batch_size
        hashes = list(strings_to_embed.keys())
        
        # Create batches
        return [hashes[i:i + batch_size] for i in range(0, len(hashes), batch_size)]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _embed_batch(self, batch_strings):
        """
        Generate embeddings for a batch of strings.
        
        Args:
            batch_strings (list): List of strings to embed
            
        Returns:
            tuple: (list of embeddings, tokens used)
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

    def _save_results(self, processed_hashes):
        """
        Save embedding results to disk.
        
        Args:
            processed_hashes (set): Set of processed hash values
        """
        output_dir = Path(self.config['system']['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        # Save list of embedded hashes
        with open(output_dir / "embedded_hashes.json", 'w') as f:
            json.dump(list(processed_hashes), f)
        
        # Save sample of embeddings
        sample_size = min(10, len(self.embeddings))
        sample_hashes = list(self.embeddings.keys())[:sample_size]
        sample_embeddings = {h: self.embeddings[h] for h in sample_hashes}
        
        with open(output_dir / "embeddings_sample.json", 'w') as f:
            json.dump(sample_embeddings, f, indent=2)
        
        # Save final checkpoint
        checkpoint_path = Path(self.config['system']['checkpoint_dir']) / "embedding_final.ckpt"
        save_checkpoint({
            'embeddings': self.embeddings,
            'processed_hashes': list(processed_hashes)
        }, checkpoint_path)
        
        logger.info("Embedding results saved to %s", output_dir)

    def get_embedding(self, text):
        """
        Get embedding for a single text string.
        
        Args:
            text (str): Text to embed
            
        Returns:
            list: Embedding vector
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
            encoding_format="float"
        )
        
        return response.data[0].embedding
