"""
Integration module for entity resolution pipeline.

This module demonstrates how the redesigned pipeline components work together,
providing a simplified interface for running the entity resolution process.
"""

import os
import logging
import json
import time
from pathlib import Path
import argparse
import yaml

from src.data_manager import DataManager
from src.pipeline import Pipeline
from src.utils import Timer, get_memory_usage

logger = logging.getLogger(__name__)

class EntityResolutionIntegration:
    """
    Integration class for the entity resolution pipeline.
    
    Provides a simplified interface for configuring and running the 
    entity resolution process with proper data management and error handling.
    """
    
    def __init__(self, config_path=None, config=None):
        """
        Initialize the integration.
        
        Args:
            config_path (str, optional): Path to config file. Defaults to None.
            config (dict, optional): Configuration dictionary. Defaults to None.
        """
        # Load configuration
        if config:
            self.config = config
        elif config_path:
            self.config = self._load_config(config_path)
        else:
            raise ValueError("Either config_path or config must be provided")
        
        # Initialize data manager
        self.data_manager = DataManager(self.config)
        
        # Initialize pipeline
        self.pipeline = Pipeline(self.config)
        
        logger.info("EntityResolutionIntegration initialized")
    
    def _load_config(self, config_path):
        """
        Load configuration from YAML file.
        
        Args:
            config_path (str): Path to config file
            
        Returns:
            dict: Configuration
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Loaded configuration from {config_path}")
            return config
        
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def run_pipeline(self, start_from=None, end_at=None, checkpoint=None):
        """
        Run the entity resolution pipeline.
        
        Args:
            start_from (str, optional): Stage to start from. Defaults to None.
            end_at (str, optional): Stage to end at. Defaults to None.
            checkpoint (str, optional): Checkpoint to resume from. Defaults to None.
            
        Returns:
            dict: Pipeline execution results
        """
        try:
            logger.info("Starting entity resolution pipeline")
            
            with Timer() as timer:
                # Execute pipeline
                results = self.pipeline.execute(
                    start_from=start_from,
                    end_at=end_at,
                    checkpoint=checkpoint
                )
                
                # Add timing information
                results['total_duration'] = timer.duration
                results['end_memory_usage'] = get_memory_usage()
            
            logger.info(f"Entity resolution completed in {timer.duration:.2f} seconds")
            
            # Save final results
            self.data_manager.save('pipeline_final_results', results)
            
            return results
        
        except Exception as e:
            logger.error(f"Error running entity resolution pipeline: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                'error': str(e),
                'status': 'failed'
            }
    
    def get_results_summary(self):
        """
        Get a summary of the entity resolution results.
        
        Returns:
            dict: Results summary
        """
        try:
            # Load data from each stage
            classification_metrics = self.data_manager.load('classification_metrics') or {}
            clustering_metrics = self.data_manager.load('clustering_metrics') or {}
            
            # Get classification results
            if not classification_metrics:
                classification_results = self.data_manager.load_classification_results()
                if classification_results and 'metrics' in classification_results:
                    classification_metrics = classification_results['metrics']
            
            # Get clustering results
            if not clustering_metrics:
                clustering_results = self.data_manager.load_cluster_data()
                if clustering_results and 'metrics' in clustering_results:
                    clustering_metrics = clustering_results['metrics']
            
            # Create summary
            summary = {
                'classification': {
                    'precision': classification_metrics.get('precision', 0.0),
                    'recall': classification_metrics.get('recall', 0.0),
                    'f1': classification_metrics.get('f1', 0.0),
                    'accuracy': classification_metrics.get('accuracy', 0.0),
                    'true_positives': classification_metrics.get('true_positives', 0),
                    'false_positives': classification_metrics.get('false_positives', 0),
                    'true_negatives': classification_metrics.get('true_negatives', 0),
                    'false_negatives': classification_metrics.get('false_negatives', 0)
                },
                'clustering': {
                    'cluster_count': clustering_metrics.get('cluster_count', 0),
                    'total_entities': clustering_metrics.get('total_entities', 0),
                    'singleton_clusters': clustering_metrics.get('singleton_clusters', 0),
                    'max_cluster_size': clustering_metrics.get('max_cluster_size', 0),
                    'mean_cluster_size': clustering_metrics.get('mean_cluster_size', 0.0)
                },
                'pipeline': self.pipeline.get_status()
            }
            
            return summary
        
        except Exception as e:
            logger.error(f"Error getting results summary: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                'error': str(e),
                'status': 'failed'
            }
    
    def export_results(self, output_dir=None):
        """
        Export entity resolution results to files.
        
        Args:
            output_dir (str, optional): Output directory. Defaults to None.
            
        Returns:
            dict: Export results
        """
        try:
            # Use configured output directory if not specified
            if not output_dir:
                output_dir = self.config['system']['output_dir']
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Export classification results
            classification_results = self.data_manager.load_classification_results()
            if classification_results:
                with open(output_path / "classification_results.json", 'w') as f:
                    json.dump(classification_results.get('metrics', {}), f, indent=2)
            
            # Export clustering results
            clustering_results = self.data_manager.load_cluster_data()
            if clustering_results and 'clusters' in clustering_results:
                # Export cluster metrics
                with open(output_path / "clustering_metrics.json", 'w') as f:
                    json.dump(clustering_results.get('metrics', {}), f, indent=2)
                
                # Export clusters as CSV
                from src.clusterer import Clusterer
                clusterer = Clusterer(self.config)
                clusterer.clusters = clustering_results['clusters']
                csv_path = clusterer.export_clusters_csv(output_path / "entity_clusters.csv")
            
            # Export summary
            summary = self.get_results_summary()
            with open(output_path / "entity_resolution_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Exported entity resolution results to {output_path}")
            
            return {
                'output_dir': str(output_path),
                'exported_files': [
                    "classification_results.json",
                    "clustering_metrics.json",
                    "entity_clusters.csv",
                    "entity_resolution_summary.json"
                ],
                'status': 'success'
            }
        
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                'error': str(e),
                'status': 'failed'
            }
    
    def validate_input_data(self):
        """
        Validate input data for entity resolution.
        
        Returns:
            dict: Validation results
        """
        try:
            # Check if ground truth data exists
            ground_truth_file = Path(self.config['data']['ground_truth_file'])
            
            if not ground_truth_file.exists():
                return {
                    'valid': False,
                    'error': f"Ground truth file not found: {ground_truth_file}"
                }
            
            # Check if data files exist
            data_dir = Path(self.config['data']['input_dir'])
            
            if not data_dir.exists():
                return {
                    'valid': False,
                    'error': f"Input directory not found: {data_dir}"
                }
            
            # Check if data files have expected format
            csv_files = list(data_dir.glob('*.csv'))
            
            if not csv_files:
                return {
                    'valid': False,
                    'error': f"No CSV files found in input directory: {data_dir}"
                }
            
            # Check file structure (sample first file)
            import pandas as pd
            
            sample_file = csv_files[0]
            try:
                df = pd.read_csv(sample_file)
                
                # Check for required columns
                required_columns = ['composite', 'person', 'roles', 'title', 'personId']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    return {
                        'valid': False,
                        'error': f"Missing required columns: {missing_columns}"
                    }
                
            except Exception as e:
                return {
                    'valid': False,
                    'error': f"Error reading sample file {sample_file}: {e}"
                }
            
            # All checks passed
            return {
                'valid': True,
                'files_found': len(csv_files),
                'ground_truth_file': str(ground_truth_file),
                'sample_file': str(sample_file)
            }
        
        except Exception as e:
            logger.error(f"Error validating input data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                'valid': False,
                'error': str(e)
            }


def main():
    """
    Command-line interface for entity resolution.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Entity Resolution Integration")
    
    parser.add_argument('--config', default='config.yml',
                      help='Path to configuration file')
    
    parser.add_argument('--start-from',
                      help='Stage to start from')
    
    parser.add_argument('--end-at',
                      help='Stage to end at')
    
    parser.add_argument('--checkpoint',
                      help='Checkpoint to resume from')
    
    parser.add_argument('--validate-only', action='store_true',
                      help='Only validate input data, don\'t run pipeline')
    
    parser.add_argument('--export-dir',
                      help='Directory to export results to')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("entity_resolution.log"),
            logging.StreamHandler()
        ]
    )
    
    # Initialize integration
    try:
        integration = EntityResolutionIntegration(config_path=args.config)
        
        # Validate input data
        validation_results = integration.validate_input_data()
        
        if not validation_results['valid']:
            logger.error(f"Input data validation failed: {validation_results['error']}")
            return 1
        
        logger.info(f"Input data validation successful: {validation_results}")
        
        # Exit if only validating
        if args.validate_only:
            return 0
        
        # Run pipeline
        results = integration.run_pipeline(
            start_from=args.start_from,
            end_at=args.end_at,
            checkpoint=args.checkpoint
        )
        
        if 'error' in results:
            logger.error(f"Pipeline execution failed: {results['error']}")
            return 1
        
        # Export results
        if args.export_dir:
            export_results = integration.export_results(args.export_dir)
            
            if export_results['status'] != 'success':
                logger.error(f"Results export failed: {export_results.get('error', 'Unknown error')}")
                return 1
            
            logger.info(f"Results exported to {export_results['output_dir']}")
        
        # Print summary
        summary = integration.get_results_summary()
        
        print("\nEntity Resolution Results:")
        print(f"Classification: Precision={summary['classification']['precision']:.4f}, "
             f"Recall={summary['classification']['recall']:.4f}, "
             f"F1={summary['classification']['f1']:.4f}")
        print(f"Clustering: {summary['clustering']['cluster_count']} clusters, "
             f"{summary['clustering']['total_entities']} entities")
        print(f"Pipeline: {len(summary['pipeline']['completed_stages'])}/{10} stages completed "
             f"in {summary['pipeline']['duration']:.2f} seconds")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error in entity resolution integration: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
