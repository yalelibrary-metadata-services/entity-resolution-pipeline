#!/usr/bin/env python3
"""
Entity Resolution Pipeline for Yale University Library Catalog

Main entry point for executing the complete entity resolution pipeline
or individual pipeline components.
"""

import os
import sys
import time
import argparse
import logging
import yaml
from pathlib import Path
import traceback

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("entity_resolution.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import pipeline components
try:
    from src.pipeline import Pipeline
    from src.batch_preprocessing import Preprocessor
    from src.embedding import Embedder
    from src.indexing import Indexer
    from src.imputation import Imputator
    from src.batch_querying import QueryEngine
    from src.parallel_features import FeatureEngineer
    from src.classification import Classifier
    from src.clustering import Clusterer
    from src.reporting import Reporter
    from src.analysis import Analyzer
    from src.utils import setup_monitoring, check_resources, validate_config, create_output_directories
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    traceback.print_exc()
    sys.exit(1)

def load_config(config_path="config.yml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Entity Resolution Pipeline for Yale University Library Catalog')
    
    parser.add_argument('--mode', choices=['dev', 'prod'], default=None,
                        help='Execution mode: dev (development, subset of data) or prod (production, full dataset)')
    
    parser.add_argument('--config', default='config.yml',
                        help='Path to configuration file')
    
    parser.add_argument('--component', choices=[
                            'all', 'preprocess', 'embed', 'index', 
                            'impute', 'query', 'features', 'classify', 
                            'cluster', 'analyze', 'report'
                         ], default='all',
                        help='Pipeline component to execute')
    
    parser.add_argument('--checkpoint', default=None,
                        help='Checkpoint to resume from')
    
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default=None, help='Logging level')
    
    return parser.parse_args()

def update_config_with_args(config, args):
    """Update configuration with command-line arguments."""
    if args.mode:
        config['system']['mode'] = args.mode
    
    if args.log_level:
        config['system']['log_level'] = args.log_level
    
    # Update logging level based on config
    log_level = getattr(logging, config['system']['log_level'])
    logger.setLevel(log_level)
    for handler in logger.handlers:
        handler.setLevel(log_level)
    
    return config

def main():
    """Main entry point for the entity resolution pipeline."""
    start_time = time.time()
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update configuration with command-line arguments
    config = update_config_with_args(config, args)
    
    # Validate configuration
    if not validate_config(config):
        logger.error("Invalid configuration")
        sys.exit(1)
    
    # Create necessary directories
    create_output_directories(config)
    
    # Check system resources
    check_resources(config)
    
    # Setup monitoring if enabled
    if config['monitoring']['prometheus_enabled']:
        setup_monitoring(config)
    
    # Initialize pipeline or individual component
    try:
        if args.component == 'all':
            # Execute complete pipeline
            pipeline = Pipeline(config)
            pipeline.execute(checkpoint=args.checkpoint)
        else:
            # Execute specific component
            component_map = {
                'preprocess': Preprocessor,
                'embed': Embedder,
                'index': Indexer,
                'impute': Imputator,
                'query': QueryEngine,
                'features': FeatureEngineer,
                'classify': Classifier,
                'cluster': Clusterer,
                'analyze': Analyzer,
                'report': Reporter
            }
            
            component_class = component_map[args.component]
            component = component_class(config)
            component.execute(checkpoint=args.checkpoint)
    
    except KeyboardInterrupt:
        logger.warning("Pipeline execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error executing pipeline: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Log execution time
    execution_time = time.time() - start_time
    logger.info(f"Execution completed in {execution_time:.2f} seconds")

if __name__ == "__main__":
    main()
