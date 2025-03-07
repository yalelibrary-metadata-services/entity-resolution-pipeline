#!/usr/bin/env python3
"""
Fix embedding stage for entity resolution pipeline.

This script:
1. Fixes the path issue when loading field hash mapping
2. Creates direct file references for the embedding stage
3. Forces reprocessing of all strings
"""

import os
import logging
import json
import shutil
from pathlib import Path
import yaml
import pickle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fix_embedding")

def load_config(config_path="config.yml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return None

def fix_paths_and_checkpoints(config):
    """
    Fix paths and checkpoints for embedding stage.
    
    Args:
        config (dict): Configuration parameters
    """
    output_dir = Path(config['system']['output_dir'])
    checkpoint_dir = Path(config['system']['checkpoint_dir'])
    
    # 1. Fix field hash mapping - make sure it's directly accessible
    field_hash_path = output_dir / "field_hash_index.json"
    if field_hash_path.exists():
        # Ensure the field hash mapping is also in field_hash_mapping.json
        # as some code paths might look for this file
        direct_path = output_dir / "field_hash_mapping.json"
        shutil.copy(field_hash_path, direct_path)
        logger.info(f"Copied field hash mapping to {direct_path}")
        
        # Check the content
        with open(field_hash_path, 'r') as f:
            field_hash_mapping = json.load(f)
        
        # Check if any hashes have embedable field types
        fields_to_embed = config['embedding']['fields_to_embed']
        count_to_embed = 0
        for hash_value, fields in field_hash_mapping.items():
            if any(field in fields_to_embed for field in fields):
                count_to_embed += 1
        
        logger.info(f"Field hash mapping has {len(field_hash_mapping)} hashes")
        logger.info(f"{count_to_embed} hashes have fields that need embedding")
    else:
        logger.error(f"Field hash mapping not found at {field_hash_path}")
        return False
    
    # 2. Reset embedding stage checkpoints
    embedding_checkpoint = checkpoint_dir / "embedding_final.ckpt"
    if embedding_checkpoint.exists():
        # Rename the existing checkpoint for backup
        backup_path = checkpoint_dir / "embedding_final.ckpt.bak"
        shutil.move(embedding_checkpoint, backup_path)
        logger.info(f"Backed up embedding checkpoint to {backup_path}")
    
    # 3. Remove embedded_hashes.json to force reprocessing
    embedded_hashes_path = output_dir / "embedded_hashes.json"
    if embedded_hashes_path.exists():
        os.remove(embedded_hashes_path)
        logger.info(f"Removed {embedded_hashes_path} to force reprocessing")
    
    # 4. Create a direct file reference for field_hash_mapping
    # This is for the _load_field_hash_mapping function in embedding.py
    field_hash_dir = output_dir / "field_hash_mapping_direct"
    field_hash_dir.mkdir(exist_ok=True)
    
    with open(field_hash_dir / "field_hash_mapping.json", 'w') as f:
        json.dump(field_hash_mapping, f)
    
    # 5. Modify _load_field_hash_mapping function in embedding.py
    embedding_path = Path("src") / "embedding.py"
    if embedding_path.exists():
        with open(embedding_path, 'r') as f:
            code = f.read()
        
        # Find the _load_field_hash_mapping function
        # We need to insert a simple version that directly loads our file
        if "_load_field_hash_mapping" in code:
            # Backup the original file
            shutil.copy(embedding_path, str(embedding_path) + ".bak")
            logger.info(f"Backed up embedding.py to {embedding_path}.bak")
            
            # Replace the function
            new_func = '''
    def _load_field_hash_mapping(self):
        """
        Load field hash mapping with direct file path.
        """
        try:
            output_dir = Path(self.config['system']['output_dir'])
            
            # Try direct path first
            direct_path = output_dir / "field_hash_mapping.json"
            if direct_path.exists():
                with open(direct_path, 'r') as f:
                    return json.load(f)
            
            # Fall back to field_hash_index.json
            field_hash_path = output_dir / "field_hash_index.json"
            if field_hash_path.exists():
                with open(field_hash_path, 'r') as f:
                    return json.load(f)
            
            # Fall back to sample
            sample_path = output_dir / "field_hash_mapping_sample.json"
            if sample_path.exists():
                with open(sample_path, 'r') as f:
                    return json.load(f)
            
            logger.error("No field hash mapping found!")
            return {}
        
        except Exception as e:
            logger.error(f"Error loading field hash mapping: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
'''
            
            # Replace the function in the code
            import re
            pattern = r'def _load_field_hash_mapping\(self\):.*?return \{\}\s*\n'
            new_code = re.sub(pattern, new_func, code, flags=re.DOTALL)
            
            # Write the modified code
            with open(embedding_path, 'w') as f:
                f.write(new_code)
            
            logger.info(f"Modified _load_field_hash_mapping in {embedding_path}")
    
    return True

def main():
    """Main entry point."""
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yml"
    
    # Load configuration
    config = load_config(config_path)
    if not config:
        return
    
    # Fix paths and checkpoints
    if fix_paths_and_checkpoints(config):
        logger.info("Embedding stage fixed!")
        logger.info("\nNext steps:")
        logger.info("1. Run the embedding stage: python main.py --component embed")
        logger.info("2. Continue with the rest of the pipeline")
    else:
        logger.error("Failed to fix embedding stage")

if __name__ == "__main__":
    main()
