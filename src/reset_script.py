#!/usr/bin/env python3
"""
Reset script for entity resolution pipeline.

This script resets the pipeline by:
1. Removing checkpoint files
2. Clearing output directories
3. Resetting Weaviate collections
4. Cleaning temporary files
"""

import os
import shutil
import argparse
import logging
from pathlib import Path
import yaml
import weaviate

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("reset_pipeline")

def load_config(config_path="config.yml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return None

def reset_weaviate_collections(config):
    """Reset Weaviate collections."""
    try:
        # Connect to Weaviate
        client = weaviate.connect_to_local(
            host=config['weaviate']['host'],
            port=config['weaviate']['port']
        )

        # Get list of collections
        collections = client.collections.list_all()
        
        # Delete specified collection
        collection_name = config['weaviate']['collection_name']
        if collection_name in collections:
            logger.info(f"Deleting Weaviate collection: {collection_name}")
            client.collections.delete(collection_name)
            logger.info(f"Collection {collection_name} deleted successfully")
        else:
            logger.info(f"Collection {collection_name} not found, nothing to delete")
        
        # Close client
        client.close()
        return True
    
    except Exception as e:
        logger.error(f"Error resetting Weaviate collections: {e}")
        return False

def clear_directory(directory_path, keep_dir=True):
    """
    Clear contents of a directory.
    
    Args:
        directory_path (Path): Directory path
        keep_dir (bool): If True, keep directory but remove contents, 
                        if False, remove directory completely
    """
    path = Path(directory_path)
    
    if not path.exists():
        logger.info(f"Directory {path} doesn't exist, nothing to clear")
        return
    
    try:
        if keep_dir:
            # Remove all contents but keep directory
            for item in path.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            logger.info(f"Cleared contents of {path}")
        else:
            # Remove directory completely
            shutil.rmtree(path)
            logger.info(f"Removed directory {path}")
    
    except Exception as e:
        logger.error(f"Error clearing directory {path}: {e}")

def reset_pipeline(config, keep_directories=True):
    """
    Reset pipeline based on configuration.
    
    Args:
        config (dict): Configuration parameters
        keep_directories (bool): If True, keep directories but remove contents
    """
    # Clear checkpoint directory
    checkpoint_dir = Path(config['system']['checkpoint_dir'])
    clear_directory(checkpoint_dir, keep_directories)
    
    # Clear output directory
    output_dir = Path(config['system']['output_dir'])
    clear_directory(output_dir, keep_directories)
    
    # Clear temporary directory
    temp_dir = Path(config['system']['temp_dir'])
    clear_directory(temp_dir, keep_directories)
    
    # Reset Weaviate collections
    reset_weaviate_collections(config)
    
    logger.info("Pipeline reset completed")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Reset entity resolution pipeline")
    parser.add_argument("--config", default="config.yml", help="Path to configuration file")
    parser.add_argument("--remove-dirs", action="store_true", help="Remove directories completely instead of just clearing them")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        return
    
    # Confirm reset
    confirm = input("This will reset the pipeline and delete all data. Are you sure? (y/n): ")
    if confirm.lower() != "y":
        logger.info("Reset aborted")
        return
    
    # Reset pipeline
    reset_pipeline(config, keep_directories=not args.remove_dirs)

if __name__ == "__main__":
    main()
