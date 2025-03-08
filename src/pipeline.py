"""
Redesigned pipeline orchestration module for entity resolution.

This module provides the Pipeline class, which coordinates the execution
of the complete entity resolution pipeline with improved data management
and component integration.
"""

import os
import logging
import json
import time
from pathlib import Path
from tqdm import tqdm
import traceback

from src.utils import Timer
from src.data_manager import DataManager
from src.batch_preprocessing import Preprocessor
from src.embedding import Embedder
from src.indexing import Indexer
from src.imputation import Imputator
from src.batch_querying import QueryEngine
from src.feature_engineer import FeatureEngineer
from src.classification import Classifier
from src.clustering import Clusterer
from src.reporting import Reporter
from src.analysis import Analyzer

logger = logging.getLogger(__name__)

class Pipeline:
    """
    Orchestrates the execution of the entity resolution pipeline with
    improved data management and component integration.
    
    Features:
    - Centralized data management
    - Consistent state tracking
    - Proper error handling and recovery
    - Enhanced checkpointing and resumability
    """
    
    def __init__(self, config):
        """
        Initialize the pipeline with configuration parameters.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        
        # Initialize data manager
        self.data_manager = DataManager(config)
        
        # Initialize pipeline state
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
            self._load_pipeline_checkpoint(checkpoint)
            logger.info(f"Resumed pipeline execution from checkpoint: {checkpoint}")
        
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
        
        # Check for completed stages
        self._check_completed_stages(stages)
        
        # Skip completed stages if resuming
        if checkpoint and self.state.get('completed_stages'):
            last_completed = self.state['completed_stages'][-1] if self.state['completed_stages'] else None
            if last_completed:
                last_completed_idx = next((i for i, (name, _) in enumerate(stages) if name == last_completed), -1)
                start_idx = max(start_idx, last_completed_idx + 1)
                logger.info(f"Resuming from after stage {last_completed} (index {last_completed_idx})")
        
        # Execute selected stages
        results = {}
        start_time = time.time()
        
        try:
            for i, (stage_name, component) in enumerate(stages[start_idx:end_idx+1], start_idx):
                # Skip already completed stages
                if stage_name in self.state.get('completed_stages', []):
                    logger.info(f"Skipping already completed stage: {stage_name}")
                    continue
                
                # Update state
                stage_start = time.time()
                self.state['current_stage'] = stage_name
                
                logger.info(f"Executing pipeline stage: {stage_name} ({i+1}/{end_idx+1})")
                
                # Execute component
                try:
                    # Check for stage-specific checkpoint
                    stage_checkpoint = self._get_stage_checkpoint(stage_name)
                    
                    # Execute component with checkpoint if available
                    if stage_checkpoint:
                        logger.info(f"Using existing checkpoint for stage {stage_name}: {stage_checkpoint}")
                        stage_result = component.execute(checkpoint=stage_checkpoint)
                    else:
                        stage_result = component.execute()
                    
                    # Store result
                    results[stage_name] = stage_result
                    
                    # Update state
                    stage_duration = time.time() - stage_start
                    self.state['completed_stages'].append(stage_name)
                    self.state['metrics'][stage_name] = {
                        'duration': stage_duration,
                        'records_processed': stage_result.get('records_processed', 0) if isinstance(stage_result, dict) else 0
                    }
                    
                    # Save checkpoint after each stage
                    self._save_pipeline_checkpoint(f"pipeline_{stage_name}")
                    
                    logger.info(f"Completed stage: {stage_name} in {stage_duration:.2f} seconds")
                
                except Exception as e:
                    logger.error(f"Error executing stage {stage_name}: {str(e)}")
                    traceback.print_exc()
                    
                    # Save error checkpoint
                    self._save_pipeline_checkpoint(f"pipeline_error_{stage_name}")
                    
                    # Determine whether to continue
                    if self.config['system'].get('continue_on_error', False):
                        logger.warning(f"Continuing after error in stage {stage_name}")
                        continue
                    else:
                        logger.error(f"Stopping pipeline execution after error in stage {stage_name}")
                        raise
        
        except Exception as e:
            # Save error checkpoint if not already saved
            error_checkpoint = Path(self.config['system']['checkpoint_dir']) / "pipeline_error.ckpt"
            self._save_pipeline_checkpoint("pipeline_error")
            
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise
        
        # Finalize pipeline execution
        total_duration = time.time() - start_time
        self.state['completed_at'] = time.time()
        self.state['total_duration'] = total_duration
        
        # Save final state
        self._save_pipeline_checkpoint("pipeline_complete")
        
        # Save summary results
        summary = {
            'duration': total_duration,
            'stages': self.state['metrics'],
            'mode': self.config['system']['mode'],
            'timestamp': time.time(),
            'completed_stages': self.state['completed_stages']
        }
        
        self.data_manager.save('pipeline_summary', summary)
        
        # Generate dashboard data for visualization
        self._generate_dashboard_data(summary)
        
        logger.info(f"Pipeline execution completed in {total_duration:.2f} seconds")
        
        return {
            'completed_stages': self.state['completed_stages'],
            'duration': total_duration,
            'results': results
        }
    
    def _check_completed_stages(self, stages):
        """
        Check which stages have already been completed based on output data.
        
        Args:
            stages (list): List of pipeline stages
        """
        # Get list of existing datasets
        datasets = self.data_manager.list_datasets()
        
        # Check for stage-specific completion indicators
        completion_indicators = {
            'preprocess': ['unique_strings', 'record_field_hashes'],
            'embed': ['embedding_index', 'embedded_hashes'],
            'index': ['collection_statistics', 'indexing_metadata'],
            'impute': ['imputed_values', 'imputation_statistics'],
            'query': ['candidate_pairs'],
            'features': ['feature_vectors', 'labels', 'feature_names'],
            'classify': ['classifier_model', 'classification_metrics'],
            'cluster': ['entity_clusters', 'clustering_metrics'],
            'analyze': ['analysis_results'],
            'report': ['report_metadata']
        }
        
        # Check each stage
        for stage_name, indicators in completion_indicators.items():
            # Check if all indicators exist
            completed = all(indicator in datasets for indicator in indicators)
            
            if completed and stage_name not in self.state['completed_stages']:
                logger.info(f"Found existing outputs for stage {stage_name}, marking as completed")
                self.state['completed_stages'].append(stage_name)
    
    def _get_stage_checkpoint(self, stage_name):
        """
        Get checkpoint path for a specific stage.
        
        Args:
            stage_name (str): Pipeline stage name
            
        Returns:
            str: Checkpoint path or None if not found
        """
        # Check specific checkpoint locations
        checkpoint_dir = Path(self.config['system']['checkpoint_dir'])
        checkpoint_paths = [
            checkpoint_dir / f"{stage_name}_final.ckpt",
            checkpoint_dir / f"pipeline_{stage_name}.ckpt"
        ]
        
        for path in checkpoint_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def _load_pipeline_checkpoint(self, checkpoint):
        """
        Load pipeline state from checkpoint.
        
        Args:
            checkpoint (str): Checkpoint path or name
        """
        # Handle both full paths and checkpoint names
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            # Try in checkpoint directory
            checkpoint_dir = Path(self.config['system']['checkpoint_dir'])
            checkpoint_path = checkpoint_dir / checkpoint
            
            # Add extension if needed
            if not checkpoint_path.exists() and not checkpoint.endswith('.ckpt'):
                checkpoint_path = checkpoint_dir / f"{checkpoint}.ckpt"
        
        # Load state if checkpoint exists
        if checkpoint_path.exists():
            state = self.data_manager.load(checkpoint_path.stem)
            
            if state:
                self.state.update(state)
                logger.info(f"Loaded pipeline state from {checkpoint_path}")
            else:
                logger.warning(f"Failed to load pipeline state from {checkpoint_path}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
    
    def _save_pipeline_checkpoint(self, checkpoint_name):
        """
        Save pipeline state to checkpoint.
        
        Args:
            checkpoint_name (str): Checkpoint name
        """
        # Update state timestamp
        self.state['updated_at'] = time.time()
        
        # Save state
        self.data_manager.save(checkpoint_name, self.state, 
                              metadata={
                                  'pipeline_mode': self.config['system']['mode'],
                                  'current_stage': self.state['current_stage'],
                                  'completed_stages': len(self.state['completed_stages'])
                              })
        
        logger.info(f"Saved pipeline state to checkpoint: {checkpoint_name}")
    
    def _generate_dashboard_data(self, summary):
        """
        Generate dashboard data for visualization.
        
        Args:
            summary (dict): Pipeline summary
        """
        try:
            # Create dashboard data
            dashboard = {
                'pipeline': {
                    'total_duration': summary['duration'],
                    'mode': summary['mode'],
                    'completed_stages': summary['completed_stages'],
                    'stage_durations': {stage: metrics['duration'] for stage, metrics in summary['stages'].items()}
                },
                'data': {
                    'record_count': self._get_record_count(),
                    'unique_strings_count': self._get_unique_strings_count(),
                    'embedded_vectors_count': self._get_embedded_vectors_count()
                },
                'classification': self._get_classification_metrics(),
                'clustering': self._get_clustering_metrics()
            }
            
            # Save dashboard data
            self.data_manager.save('dashboard_data', dashboard)
            
            logger.info("Generated dashboard data")
        
        except Exception as e:
            logger.error(f"Error generating dashboard data: {str(e)}")
    
    def _get_record_count(self):
        """
        Get count of records in the dataset.
        
        Returns:
            int: Record count
        """
        # Try different data sources
        record_field_hashes = self.data_manager.load('record_field_hashes')
        if record_field_hashes:
            return len(record_field_hashes)
        
        return 0
    
    def _get_unique_strings_count(self):
        """
        Get count of unique strings.
        
        Returns:
            int: Unique strings count
        """
        # Try different data sources
        unique_strings = self.data_manager.load('unique_strings')
        if unique_strings:
            return len(unique_strings)
        
        # Try metadata
        metadata = self.data_manager.get_metadata('unique_strings')
        if metadata and 'count' in metadata:
            return metadata['count']
        
        return 0
    
    def _get_embedded_vectors_count(self):
        """
        Get count of embedded vectors.
        
        Returns:
            int: Embedded vectors count
        """
        # Try different data sources
        embedding_index = self.data_manager.load('embedding_index')
        if embedding_index and 'count' in embedding_index:
            return embedding_index['count']
        
        # Try metadata
        metadata = self.data_manager.get_metadata('embedding_index')
        if metadata and 'count' in metadata:
            return metadata['count']
        
        return 0
    
    def _get_classification_metrics(self):
        """
        Get classification metrics.
        
        Returns:
            dict: Classification metrics
        """
        # Try to load metrics
        metrics = self.data_manager.load('classification_metrics')
        if metrics:
            return metrics
        
        # Try to load from classification results
        results = self.data_manager.load_classification_results()
        if results and 'metrics' in results:
            return results['metrics']
        
        return {}
    
    def _get_clustering_metrics(self):
        """
        Get clustering metrics.
        
        Returns:
            dict: Clustering metrics
        """
        # Try to load metrics
        metrics = self.data_manager.load('clustering_metrics')
        if metrics:
            return metrics
        
        # Try to load from clustering results
        results = self.data_manager.load_cluster_data()
        if results and 'metrics' in results:
            return results['metrics']
        
        return {}
    
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
    
    def get_stage_metric(self, stage, metric):
        """
        Get a specific metric for a pipeline stage.
        
        Args:
            stage (str): Pipeline stage name
            metric (str): Metric name
            
        Returns:
            value: Metric value or None if not found
        """
        if stage in self.state['metrics'] and metric in self.state['metrics'][stage]:
            return self.state['metrics'][stage][metric]
        
        return None
