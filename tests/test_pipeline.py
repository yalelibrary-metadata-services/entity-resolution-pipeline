"""
Test script for the entity resolution pipeline.

This script provides tests for verifying the functionality of the
entity resolution pipeline and its components.
"""

import os
import sys
import logging
import yaml
import time
import argparse
from pathlib import Path

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import pipeline components
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
from src.utils import setup_monitoring, check_resources

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PipelineTest:
    """
    Test class for entity resolution pipeline.
    """
    
    def __init__(self, config_path="config.yml"):
        """
        Initialize the pipeline test.
        
        Args:
            config_path (str): Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Override configuration for testing
        self.config['system']['mode'] = 'dev'
        self.config['system']['dev_sample_size'] = 100
        self.config['system']['max_workers'] = 2
        self.config['system']['batch_size'] = 10
        
        # Override directories for testing
        self.test_dir = Path("test_output")
        self.test_dir.mkdir(exist_ok=True)
        
        self.config['system']['output_dir'] = str(self.test_dir / "output")
        self.config['system']['checkpoint_dir'] = str(self.test_dir / "checkpoints")
        self.config['system']['temp_dir'] = str(self.test_dir / "temp")
        
        # Create directories
        for dir_path in [self.config['system']['output_dir'], 
                        self.config['system']['checkpoint_dir'], 
                        self.config['system']['temp_dir']]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        logger.info("Pipeline test initialized with config: %s", config_path)

    def test_component(self, component_name):
        """
        Test a specific pipeline component.
        
        Args:
            component_name (str): Name of the component to test
            
        Returns:
            bool: True if test passed, False otherwise
        """
        try:
            logger.info("Testing component: %s", component_name)
            
            # Initialize component
            component = self._get_component(component_name)
            
            if component is None:
                logger.error("Invalid component name: %s", component_name)
                return False
            
            # Execute component
            start_time = time.time()
            result = component.execute()
            duration = time.time() - start_time
            
            # Check result
            if result is None:
                logger.error("Component returned None: %s", component_name)
                return False
            
            logger.info("Component test passed: %s (%.2f seconds)", component_name, duration)
            return True
        
        except Exception as e:
            logger.error("Error testing component %s: %s", component_name, str(e))
            return False

    def test_all_components(self):
        """
        Test all pipeline components individually.
        
        Returns:
            dict: Dictionary of component name -> test result
        """
        components = [
            'preprocessor',
            'embedder',
            'indexer',
            'imputator',
            'query_engine',
            'feature_engineer',
            'classifier',
            'clusterer',
            'analyzer',
            'reporter'
        ]
        
        results = {}
        
        for component_name in components:
            results[component_name] = self.test_component(component_name)
        
        return results

    def test_pipeline(self):
        """
        Test the complete pipeline.
        
        Returns:
            bool: True if test passed, False otherwise
        """
        try:
            logger.info("Testing complete pipeline")
            
            # Initialize pipeline
            pipeline = Pipeline(self.config)
            
            # Execute pipeline
            start_time = time.time()
            result = pipeline.execute()
            duration = time.time() - start_time
            
            # Check result
            if result is None:
                logger.error("Pipeline returned None")
                return False
            
            logger.info("Pipeline test passed (%.2f seconds)", duration)
            return True
        
        except Exception as e:
            logger.error("Error testing pipeline: %s", str(e))
            return False

    def verify_outputs(self):
        """
        Verify pipeline outputs.
        
        Returns:
            bool: True if verification passed, False otherwise
        """
        try:
            logger.info("Verifying pipeline outputs")
            
            output_dir = Path(self.config['system']['output_dir'])
            
            # Check for expected output files
            expected_files = [
                "pipeline_summary.json",
                "unique_strings_sample.json",
                "string_counts_sample.json",
                "field_hash_mapping_sample.json",
                "record_field_hashes_sample.json",
                "field_statistics.json"
            ]
            
            missing_files = []
            
            for file_name in expected_files:
                file_path = output_dir / file_name
                if not file_path.exists():
                    missing_files.append(file_name)
            
            if missing_files:
                logger.error("Missing output files: %s", ", ".join(missing_files))
                return False
            
            logger.info("Output verification passed")
            return True
        
        except Exception as e:
            logger.error("Error verifying outputs: %s", str(e))
            return False

    def _get_component(self, component_name):
        """
        Get a pipeline component instance.
        
        Args:
            component_name (str): Name of the component
            
        Returns:
            object: Component instance or None if invalid name
        """
        component_map = {
            'preprocessor': Preprocessor,
            'embedder': Embedder,
            'indexer': Indexer,
            'imputator': Imputator,
            'query_engine': QueryEngine,
            'feature_engineer': FeatureEngineer,
            'classifier': Classifier,
            'clusterer': Clusterer,
            'analyzer': Analyzer,
            'reporter': Reporter
        }
        
        if component_name not in component_map:
            return None
        
        component_class = component_map[component_name]
        return component_class(self.config)

def main():
    """
    Main entry point for pipeline test script.
    """
    parser = argparse.ArgumentParser(description='Test entity resolution pipeline')
    
    parser.add_argument('--config', default='config.yml',
                        help='Path to configuration file')
    
    parser.add_argument('--component', choices=[
                            'all', 'pipeline', 'preprocessor', 'embedder', 'indexer', 
                            'imputator', 'query_engine', 'feature_engineer', 'classifier', 
                            'clusterer', 'analyzer', 'reporter'
                         ], default='all',
                        help='Component to test')
    
    parser.add_argument('--verify', action='store_true',
                        help='Verify pipeline outputs')
    
    args = parser.parse_args()
    
    # Initialize test
    test = PipelineTest(args.config)
    
    # Run tests
    if args.component == 'all':
        results = test.test_all_components()
        
        # Print results
        print("\nTest Results:")
        print("-" * 50)
        
        for component, passed in results.items():
            status = "PASSED" if passed else "FAILED"
            print(f"{component.ljust(20)}: {status}")
        
        print("-" * 50)
        
        # Overall result
        overall = all(results.values())
        print(f"Overall: {'PASSED' if overall else 'FAILED'}")
    
    elif args.component == 'pipeline':
        passed = test.test_pipeline()
        print(f"\nPipeline Test: {'PASSED' if passed else 'FAILED'}")
    
    else:
        passed = test.test_component(args.component)
        print(f"\n{args.component} Test: {'PASSED' if passed else 'FAILED'}")
    
    # Verify outputs if requested
    if args.verify:
        verified = test.verify_outputs()
        print(f"\nOutput Verification: {'PASSED' if verified else 'FAILED'}")

if __name__ == '__main__':
    main()
