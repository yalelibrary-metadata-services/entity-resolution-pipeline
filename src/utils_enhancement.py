"""
Enhanced utility functions for entity resolution
Additional utility functions to support optimized processing
"""

import os
import time
import pickle
import logging
import mmap
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Iterable, TypeVar, Iterator
import psutil
import mmh3
import contextlib
import tempfile
import threading
from functools import lru_cache
from weaviate.classes.query import Filter, MetadataQuery

# Define type variable for generics
T = TypeVar('T')

# Configure logger
logger = logging.getLogger(__name__)

def chunk_iterable(iterable: Iterable[T], chunk_size: int) -> Iterator[List[T]]:
    """Split an iterable into chunks of specified size
    
    Args:
        iterable: Any iterable to chunk
        chunk_size: Size of each chunk
        
    Yields:
        Lists of items of size chunk_size (last chunk may be smaller)
    """
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    
    if chunk:
        yield chunk

def compute_hash(value: str, algorithm: str = 'mmh3') -> str:
    """Compute hash for a string value with faster algorithms
    
    Args:
        value: String to hash
        algorithm: Hash algorithm to use ('mmh3', 'md5', 'sha1')
        
    Returns:
        Computed hash as a string
    """
    if algorithm == 'mmh3':
        # MurmurHash is much faster than cryptographic hashes
        return str(mmh3.hash128(value))
    
    elif algorithm == 'md5':
        import hashlib
        return hashlib.md5(value.encode('utf-8')).hexdigest()
    
    elif algorithm == 'sha1':
        import hashlib
        return hashlib.sha1(value.encode('utf-8')).hexdigest()
    
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

@contextlib.contextmanager
def numpy_memmap(filename: str, shape: Tuple[int, ...], dtype=np.float32, mode: str = 'w+') -> np.ndarray:
    """Context manager for memory-mapped numpy arrays
    
    Args:
        filename: Path to the file to memory-map
        shape: Shape of the array
        dtype: Data type of the array
        mode: File mode ('r', 'r+', 'w+')
        
    Yields:
        Memory-mapped numpy array
    """
    # Calculate file size
    element_size = np.dtype(dtype).itemsize
    file_size = int(np.prod(shape)) * element_size
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Create the file if it doesn't exist or mode is 'w+'
        if mode == 'w+' or not os.path.exists(filename):
            with open(filename, 'wb') as f:
                f.seek(file_size - 1)
                f.write(b'\0')
        
        # Create memory-mapped array
        memmap_array = np.memmap(filename, dtype=dtype, mode=mode, shape=shape)
        
        # Yield array to the caller
        yield memmap_array
        
        # Flush to disk
        if mode != 'r':
            memmap_array.flush()
    
    except Exception as e:
        logger.error(f"Error with memory-mapped file {filename}: {e}")
        raise
    
    finally:
        # Delete memmap object to close file
        if 'memmap_array' in locals():
            del memmap_array

def optimize_batch_size(num_items: int, item_size_bytes: int, target_batch_mb: int = 100) -> int:
    """Optimize batch size based on item size and target batch size
    
    Args:
        num_items: Total number of items
        item_size_bytes: Estimated size of each item in bytes
        target_batch_mb: Target batch size in MB
        
    Returns:
        Optimal batch size
    """
    # Convert target_batch_mb to bytes
    target_batch_bytes = target_batch_mb * 1024 * 1024
    
    # Calculate optimal batch size
    batch_size = max(1, min(num_items, target_batch_bytes // item_size_bytes))
    
    return batch_size

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage information
    
    Returns:
        Dictionary with memory usage information
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / (1024 * 1024),  # Resident Set Size
        'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual Memory Size
        'percent': process.memory_percent(),
        'system_total_gb': psutil.virtual_memory().total / (1024**3),
        'system_available_gb': psutil.virtual_memory().available / (1024**3),
        'system_percent': psutil.virtual_memory().percent
    }

def display_stage_summary(stage_name, stats, elapsed_time):
    """Display a rich summary of pipeline stage results
    
    Args:
        stage_name: Name of the pipeline stage
        stats: Dictionary containing stage statistics
        elapsed_time: Time taken to complete the stage in seconds
    """
    # Convert elapsed time to a readable format
    if elapsed_time < 60:
        time_str = f"{elapsed_time:.2f} seconds"
    elif elapsed_time < 3600:
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        time_str = f"{minutes} minutes, {seconds:.2f} seconds"
    else:
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60
        time_str = f"{hours} hours, {minutes} minutes, {seconds:.2f} seconds"
    
    # Create a formatted header
    header = f" {stage_name.upper()} STAGE SUMMARY "
    header_padding = "=" * 20
    full_header = f"\n{header_padding}{header}{header_padding}"
    
    # Print the header
    print(full_header)
    
    # Print time information
    print(f"Time elapsed: {time_str}")
    
    # Print statistics based on stage type
    if stage_name.lower() == "preprocessing":
        if "unique_strings" in stats:
            print(f"Unique strings processed: {stats['unique_strings']:,}")
        if "records" in stats:
            print(f"Total records processed: {stats['records']:,}")
        if "person_ids" in stats:
            print(f"Unique person IDs: {stats['person_ids']:,}")
            
    elif stage_name.lower() == "embedding":
        if "embeddings" in stats:
            print(f"Total embeddings generated: {stats['embeddings']:,}")
        if "api_calls" in stats:
            print(f"API calls made: {stats['api_calls']:,}")
        if "tokens" in stats:
            print(f"Total tokens used: {stats['tokens']:,}")
        if "cache_hits" in stats:
            print(f"Cache hits: {stats['cache_hits']:,}")
            
    elif stage_name.lower() == "feature_extraction":
        if "total_pairs" in stats:
            print(f"Total record pairs: {stats['total_pairs']:,}")
        if "successful" in stats:
            print(f"Successful extractions: {stats['successful']:,}")
            success_rate = (stats['successful'] / stats['total_pairs']) * 100 if stats['total_pairs'] > 0 else 0
            print(f"Success rate: {success_rate:.2f}%")
        if "pairs_per_second" in stats:
            print(f"Processing speed: {stats['pairs_per_second']:.2f} pairs/second")
            
    elif stage_name.lower() == "training":
        if "train_size" in stats:
            print(f"Training samples: {stats['train_size']:,}")
        if "test_size" in stats:
            print(f"Test samples: {stats['test_size']:,}")
        if "precision" in stats:
            print(f"Precision: {stats['precision']:.4f}")
        if "recall" in stats:
            print(f"Recall: {stats['recall']:.4f}")
        if "f1" in stats:
            print(f"F1 Score: {stats['f1']:.4f}")
            
    elif stage_name.lower() == "classification":
        if "total_entities" in stats:
            print(f"Total entities: {stats['total_entities']:,}")
        if "match_pairs" in stats:
            print(f"Match pairs found: {stats['match_pairs']:,}")
        if "match_ratio" in stats:
            print(f"Match ratio: {stats['match_ratio']:.4f} pairs per entity")
            
    elif stage_name.lower() == "clustering":
        if "clusters" in stats:
            print(f"Total clusters: {stats['clusters']:,}")
        if "avg_cluster_size" in stats:
            print(f"Average cluster size: {stats['avg_cluster_size']:.2f} entities")
        if "max_cluster_size" in stats:
            print(f"Largest cluster: {stats['max_cluster_size']:,} entities")
    
    # Print memory usage if available
    if "peak_memory_mb" in stats:
        print(f"Peak memory usage: {stats['peak_memory_mb']:.2f} MB")
    
    # Print footer
    print("=" * (len(full_header) - 1))
    print()  # Add a blank line after the summary

@contextlib.contextmanager
def monitor_memory_usage(interval: float = 5.0, log_level: int = logging.INFO) -> None:
    """Context manager to monitor memory usage during a block of code
    
    Args:
        interval: Monitoring interval in seconds
        log_level: Logging level for memory usage information
        
    Yields:
        None
    """
    stop_event = threading.Event()
    
    def monitor_thread():
        peak_memory = 0
        while not stop_event.is_set():
            memory_usage = get_memory_usage()
            peak_memory = max(peak_memory, memory_usage['rss_mb'])
            
            logger.log(log_level, f"Memory usage: {memory_usage['rss_mb']:.1f} MB, "
                                 f"System: {memory_usage['system_percent']:.1f}% used")
            
            stop_event.wait(interval)
        
        logger.log(log_level, f"Peak memory usage: {peak_memory:.1f} MB")
    
    # Start monitoring thread
    thread = threading.Thread(target=monitor_thread, daemon=True)
    thread.start()
    
    try:
        yield
    finally:
        # Stop monitoring thread
        stop_event.set()
        thread.join()

def persistent_lru_cache(maxsize: int = 128, filename: str = None) -> callable:
    """Decorator for an LRU cache that can be saved to and loaded from disk
    
    Args:
        maxsize: Maximum size of the LRU cache
        filename: File to save/load cache (if None, cache is not persistent)
        
    Returns:
        Decorator function
    """
    def decorator(func):
        # Create LRU cache
        cache = lru_cache(maxsize=maxsize)(func)
        
        # Add cache persistence methods
        def save_cache():
            if filename is None:
                return
            
            # Get cache dictionary from LRU cache
            cache_info = cache.cache_info()
            cache_dict = {k: v for k, v in zip(cache.cache_info(), cache.cache_parameters())}
            
            # Save to file
            with open(filename, 'wb') as f:
                pickle.dump(cache_dict, f)
        
        def load_cache():
            if filename is None or not os.path.exists(filename):
                return
            
            try:
                # Load from file
                with open(filename, 'rb') as f:
                    cache_dict = pickle.load(f)
                
                # Populate cache
                for k, v in cache_dict.items():
                    cache(*k)
            except Exception as e:
                logger.error(f"Error loading cache from {filename}: {e}")
        
        # Attach methods to function
        cache.save_cache = save_cache
        cache.load_cache = load_cache
        
        return cache
    
    return decorator

def estimate_vector_storage(num_vectors: int, dimensions: int = 1536, dtype=np.float32) -> Dict[str, float]:
    """Estimate storage requirements for vectors
    
    Args:
        num_vectors: Number of vectors
        dimensions: Vector dimensions
        dtype: Data type of vector elements
        
    Returns:
        Dictionary with storage requirements in different units
    """
    element_size = np.dtype(dtype).itemsize
    bytes_per_vector = dimensions * element_size
    total_bytes = num_vectors * bytes_per_vector
    
    # Add overhead for indexing and metadata (approximately 20%)
    total_bytes_with_overhead = total_bytes * 1.2
    
    return {
        'bytes_per_vector': bytes_per_vector,
        'total_bytes': total_bytes,
        'total_bytes_with_overhead': total_bytes_with_overhead,
        'total_mb': total_bytes_with_overhead / (1024 * 1024),
        'total_gb': total_bytes_with_overhead / (1024**3),
        'total_tb': total_bytes_with_overhead / (1024**4)
    }

def create_disk_backed_queue(maxsize: int = -1) -> 'DiskBackedQueue':
    """Create a disk-backed queue for large datasets
    
    Args:
        maxsize: Maximum size of in-memory queue (-1 for unlimited)
        
    Returns:
        DiskBackedQueue instance
    """
    return DiskBackedQueue(maxsize)

class DiskBackedQueue:
    """Queue that automatically spills to disk when it gets too large"""
    
    def __init__(self, maxsize: int = -1):
        """Initialize the disk-backed queue
        
        Args:
            maxsize: Maximum number of items to keep in memory (-1 for unlimited)
        """
        self.maxsize = maxsize
        self.memory_queue = []
        self.disk_files = []
        self._size = 0
        self._lock = threading.RLock()
    
    def put(self, item: Any) -> None:
        """Add an item to the queue
        
        Args:
            item: Item to add
        """
        with self._lock:
            # If we've reached max memory size, spill to disk
            if self.maxsize >= 0 and len(self.memory_queue) >= self.maxsize:
                self._spill_to_disk()
            
            # Add to memory queue
            self.memory_queue.append(item)
            self._size += 1
    
    def get(self) -> Any:
        """Get an item from the queue
        
        Returns:
            Next item from the queue
            
        Raises:
            IndexError: If queue is empty
        """
        with self._lock:
            if self._size == 0:
                raise IndexError("Queue is empty")
            
            # If memory queue is empty, load from disk
            if not self.memory_queue and self.disk_files:
                self._load_from_disk()
            
            # Get from memory queue
            item = self.memory_queue.pop(0)
            self._size -= 1
            return item
    
    def __len__(self) -> int:
        """Get the size of the queue
        
        Returns:
            Number of items in the queue
        """
        return self._size
    
    def _spill_to_disk(self) -> None:
        """Spill memory queue to disk"""
        if not self.memory_queue:
            return
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        
        try:
            # Save memory queue to disk
            pickle.dump(self.memory_queue, temp_file)
            temp_file.close()
            
            # Add to disk files and clear memory queue
            self.disk_files.append(temp_file.name)
            self.memory_queue = []
        
        except Exception as e:
            logger.error(f"Error spilling queue to disk: {e}")
            # In case of error, keep items in memory
            temp_file.close()
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
    
    def _load_from_disk(self) -> None:
        """Load items from disk to memory queue"""
        if not self.disk_files:
            return
        
        # Get oldest file
        file_name = self.disk_files.pop(0)
        
        try:
            # Load items from disk
            with open(file_name, 'rb') as f:
                items = pickle.load(f)
            
            # Add to memory queue
            self.memory_queue.extend(items)
            
            # Delete file
            os.unlink(file_name)
        
        except Exception as e:
            logger.error(f"Error loading queue from disk: {e}")
            # If we can't load, we've lost these items
            if os.path.exists(file_name):
                os.unlink(file_name)
    
    def clear(self) -> None:
        """Clear the queue"""
        with self._lock:
            # Clear memory queue
            self.memory_queue = []
            
            # Delete all disk files
            for file_name in self.disk_files:
                if os.path.exists(file_name):
                    try:
                        os.unlink(file_name)
                    except Exception as e:
                        logger.error(f"Error deleting queue file {file_name}: {e}")
            
            self.disk_files = []
            self._size = 0
    
    def __del__(self) -> None:
        """Clean up disk files when object is deleted"""
        self.clear()

def diagnose_imputation_issues(query_engine, field_to_diagnose="subjects"):
        """Diagnose why imputation is failing for a specific field"""
        import pandas as pd
        import time
        
        logger.info(f"========== IMPUTATION DIAGNOSTIC FOR {field_to_diagnose} ==========")
        
        # Step 1: Check if field exists in database
        logger.info("Step 1: Checking field existence in database...")
        try:
            from weaviate.classes.aggregate import GroupByAggregate
            from weaviate.classes.query import Filter
            
            
            # Get total count first
            total_count_result = query_engine.collection.aggregate.over_all(
                total_count=True
            )
            total_count = total_count_result.total_count
            logger.info(f"Total objects in collection: {total_count}")
            
            # Count objects by field_type
            field_type_result = query_engine.collection.aggregate.over_all(
                group_by=GroupByAggregate(prop="field_type"),
                total_count=True
            )
            
            field_counts = {}
            for group in field_type_result.groups:
                field_type = group.grouped_by.value
                count = group.total_count
                field_counts[field_type] = count
                
            # Print results
            df = pd.DataFrame(list(field_counts.items()), columns=['field_type', 'count'])
            df = df.sort_values('count', ascending=False)
            logger.info(f"Field type distribution:\n{df.to_string(index=False)}")
            
            # Check specific field
            if field_to_diagnose in field_counts:
                logger.info(f"Field '{field_to_diagnose}' exists with {field_counts[field_to_diagnose]} objects")
            else:
                logger.error(f"Field '{field_to_diagnose}' does not exist in database!")
                logger.error("This is why imputation is failing - no data to draw from!")
                return False
                
            # Step 2: Check if field can be queried at all (with lower threshold)
            logger.info("\nStep 2: Testing field query with minimal filters...")
            
            # Get a sample vector to query with
            sample_field = next((f for f in field_counts.keys() if field_counts[f] > 0), None)
            if not sample_field:
                logger.error("No fields with data found!")
                return False
                
            # Get a sample vector
            sample_result = query_engine.collection.query.fetch_objects(
                filters=Filter.by_property("field_type").equal(sample_field),
                limit=1,
                include_vector=True
            )
            
            if not sample_result.objects:
                logger.error("No objects found for sample query!")
                return False
                
            sample_vector = sample_result.objects[0].vector.get("text_vector")
            if not sample_vector:
                logger.error("No vector found in sample object!")
                return False
                
            # Try to query the target field with very low threshold
            logger.info(f"Querying '{field_to_diagnose}' with threshold 0.1...")
            
            # Direct query to see raw results before applying threshold
            field_filter = Filter.by_property("field_type").equal(field_to_diagnose)
            raw_results = query_engine.collection.query.near_vector(
                near_vector=sample_vector,
                limit=5,
                return_metadata=MetadataQuery(distance=True),
                include_vector=True,
                filters=field_filter
            )
            
            if not raw_results.objects:
                logger.error(f"No raw results found for {field_to_diagnose} - field exists but can't be queried!")
                return False
                
            # Print distances to see what's happening
            logger.info("Raw query results:")
            for i, obj in enumerate(raw_results.objects):
                similarity = 1.0 - obj.metadata.distance
                logger.info(f"  Result {i+1}: Similarity = {similarity:.4f}, Distance = {obj.metadata.distance:.4f}")
                
            # Step 3: Test different thresholds
            logger.info("\nStep 3: Testing different similarity thresholds...")
            
            for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
                filtered_results = [obj for obj in raw_results.objects if 1.0 - obj.metadata.distance >= threshold]
                logger.info(f"Threshold {threshold}: {len(filtered_results)} results pass")
                
            # Determine optimal threshold
            all_similarities = [1.0 - obj.metadata.distance for obj in raw_results.objects]
            if all_similarities:
                avg_similarity = sum(all_similarities) / len(all_similarities)
                suggested_threshold = max(0.1, avg_similarity - 0.2)  # Set threshold 0.2 below average
                logger.info(f"Suggested similarity threshold for {field_to_diagnose}: {suggested_threshold:.2f}")
            
            # Step 4: Verify specific example
            logger.info("\nStep 4: Testing imputation query directly...")
            
            # Sample 5 composite vectors to test
            composite_results = query_engine.collection.query.fetch_objects(
                filters=Filter.by_property("field_type").equal("composite"),
                limit=5,
                include_vector=True
            )
            
            if composite_results.objects:
                for i, obj in enumerate(composite_results.objects[:3]):  # Test first 3
                    composite_vector = obj.vector.get("text_vector")
                    composite_text = obj.properties.get("text", "")[:100]  # First 100 chars
                    
                    logger.info(f"\nTesting composite {i+1}: {composite_text}...")
                    
                    # Try query with this vector
                    test_results = query_engine.collection.query.near_vector(
                        near_vector=composite_vector,
                        limit=3,
                        return_metadata=MetadataQuery(distance=True),
                        include_vector=True,
                        filters=field_filter
                    )
                    
                    if test_results.objects:
                        best_similarity = 1.0 - test_results.objects[0].metadata.distance
                        logger.info(f"Found {len(test_results.objects)} results, best similarity: {best_similarity:.4f}")
                        
                        # Is it good enough for current threshold?
                        current_threshold = query_engine.min_similarity
                        logger.info(f"Would pass current threshold ({current_threshold})? {best_similarity >= current_threshold}")
                    else:
                        logger.info("No results found for this composite")
            
            logger.info("\n===== DIAGNOSIS COMPLETE =====")
            return True
            
        except Exception as e:
            logger.error(f"Error during diagnosis: {e}", exc_info=True)
            return False