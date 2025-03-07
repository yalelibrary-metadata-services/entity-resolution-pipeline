"""
Pipeline orchestration module for entity resolution.

This module provides the Pipeline class, which coordinates the execution
of the complete entity resolution pipeline or specific pipeline components.
"""

import logging
import time
from pathlib import Path
import json

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
from src.utils import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)

class Pipeline:
    """
    Orchestrates the execution of the entity resolution pipeline.
    
    The pipeline consists of several stages:
    1. Preprocessing: Extract and deduplicate fields from the dataset
    2. Embedding: Generate vector embeddings for unique strings
    3. Indexing: Index embeddings in Weaviate for efficient similarity search
    4. Imputation: Impute missing values using vector-based hot deck approach
    5. Query: Retrieve match candidates using vector similarity
    6. Feature Engineering: Construct feature vectors for record pairs
    7. Classification: Train and apply classifier to determine matches
    8. Clustering: Group matches into entity clusters
    9. Analysis: Analyze results and error patterns
    10. Reporting: Generate reports and visualizations
    
    Attributes:
        config (dict): Configuration parameters for the pipeline
        components (dict): Dictionary of pipeline component instances
        state (dict): Current state of the pipeline (checkpointing)
    """
    
    def __init__(self, config):
        """
        Initialize the pipeline with configuration parameters.
        
        Args:
            config (dict): Configuration parameters for the pipeline
        """
        self.config = config
        self.state = {
            'started_at': time.time(),
            'completed_stages': [],
            'current_stage': None,
            'metrics': {}
        }
        
        # Initialize pipeline components
        self.components = {
            'preprocessor': Preprocessor(config),
            'embedder': Embedder(config),
            'indexer': Indexer(config),
            'imputator': Imputator(config),
            'query_engine': QueryEngine(config),
            'feature_engineer': FeatureEngineer(config),
            'classifier': Classifier(config),
            'clusterer': Clusterer(config),
            'analyzer': Analyzer(config),
            'reporter': Reporter(config)
        }
        
        logger.info("Pipeline initialized with %d components", len(self.components))

    def execute(self, start_from=None, end_at=None, checkpoint=None):
        """
        Execute the complete pipeline or specific stages.
        
        Args:
            start_from (str, optional): Stage to start from. Defaults to None (start from beginning).
            end_at (str, optional): Stage to end at. Defaults to None (run to completion).
            checkpoint (str, optional): Checkpoint to resume from. Defaults to None.
        
        Returns:
            dict: Pipeline execution results and metrics
        """
        # Load checkpoint if provided
        if checkpoint:
            self.state = load_checkpoint(checkpoint)
            logger.info("Resumed pipeline execution from checkpoint: %s", checkpoint)
        
        # Define pipeline stages in execution order
        stages = [
            ('preprocess', self.components['preprocessor']),
            ('embed', self.components['embedder']),
            ('index', self.components['indexer']),
            ('impute', self.components['imputator']),
            ('query', self.components['query_engine']),
            ('features', self.components['feature_engineer']),
            ('classify', self.components['classifier']),
            ('cluster', self.components['clusterer']),
            ('analyze', self.components['analyzer']),
            ('report', self.components['reporter'])
        ]
        
        # Determine start and end stages
        start_idx = 0
        end_idx = len(stages) - 1
        
        if start_from:
            start_idx = next((i for i, (name, _) in enumerate(stages) if name == start_from), 0)
        
        if end_at:
            end_idx = next((i for i, (name, _) in enumerate(stages) if name == end_at), len(stages) - 1)
        
        # Skip completed stages if resuming from checkpoint
        if checkpoint and self.state['completed_stages']:
            last_completed = self.state['completed_stages'][-1]
            start_idx = next((i for i, (name, _) in enumerate(stages) if name == last_completed), 0) + 1
        
        # Execute selected stages
        results = {}
        start_time = time.time()
        
        try:
            for i, (stage_name, component) in enumerate(stages[start_idx:end_idx+1], start_idx):
                stage_start = time.time()
                self.state['current_stage'] = stage_name
                
                logger.info("Executing pipeline stage: %s (%d/%d)", stage_name, i+1, end_idx+1)
                
                # Execute stage and get results
                stage_result = component.execute()
                results[stage_name] = stage_result
                
                # Update state
                stage_duration = time.time() - stage_start
                self.state['completed_stages'].append(stage_name)
                self.state['metrics'][stage_name] = {
                    'duration': stage_duration,
                    'records_processed': stage_result.get('records_processed', 0) if stage_result else 0
                }
                
                # Save checkpoint
                checkpoint_path = Path(self.config['system']['checkpoint_dir']) / f"pipeline_{stage_name}.ckpt"
                save_checkpoint(self.state, checkpoint_path)
                
                # After executing the feature engineering stage:
                if stage_name == 'features':
                    logger.info("Performing feature files diagnostic check...")
                    feature_engineer = self.components['feature_engineer']
                    feature_engineer._check_feature_files()

                logger.info("Completed stage: %s in %.2f seconds", stage_name, stage_duration)
        
        except Exception as e:
            # Save checkpoint on error
            error_checkpoint = Path(self.config['system']['checkpoint_dir']) / "pipeline_error.ckpt"
            save_checkpoint(self.state, error_checkpoint)
            logger.error("Pipeline execution failed at stage: %s - %s", self.state['current_stage'], str(e))
            raise
        
        # Finalize pipeline execution
        total_duration = time.time() - start_time
        self.state['completed_at'] = time.time()
        self.state['total_duration'] = total_duration
        
        # Save final state
        final_checkpoint = Path(self.config['system']['checkpoint_dir']) / "pipeline_complete.ckpt"
        save_checkpoint(self.state, final_checkpoint)

        # Save summary results
        summary_path = Path(self.config['system']['output_dir']) / "pipeline_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'duration': total_duration,
                'stages': self.state['metrics'],
                'mode': self.config['system']['mode'],
                'timestamp': time.time()
            }, f, indent=2)

        # Clean up resources
        self._cleanup_resources()

        logger.info("Pipeline execution completed in %.2f seconds", total_duration)
        return results

    def get_status(self):
        """
        Get current status of the pipeline.
        
        Returns:
            dict: Pipeline status information
        """
        return {
            'current_stage': self.state['current_stage'],
            'completed_stages': self.state['completed_stages'],
            'duration': time.time() - self.state['started_at'],
            'metrics': self.state['metrics']
        }

    def _cleanup_resources(self):
        """
        Clean up resources after pipeline execution.
        """
        for component_name, component in self.components.items():
            if hasattr(component, 'client') and component.client:
                logger.info(f"Cleaning up resources for {component_name}")
                try:
                    component.client.close()  # Properly close the client
                    component.client = None   # Then set to None
                except Exception as e:
                    logger.error(f"Error closing client for {component_name}: {e}")