"""
Reporting module for entity resolution pipeline.

This module provides the Reporter class, which handles generation of reports
and visualizations from pipeline results.
"""

import os
import logging
import json
import csv
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from src.utils import Timer

logger = logging.getLogger(__name__)

class Reporter:
    """
    Handles generation of reports and visualizations from pipeline results.
    
    Features:
    - Summary reports for pipeline stages
    - Performance metrics visualization
    - Feature importance reports
    - Cluster statistics and visualization
    - Entity relationship graphs
    - Data quality reports
    - Misclassified pairs analysis
    """
    
    def __init__(self, config):
        """
        Initialize the reporter with configuration parameters.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        
        # Initialize reporting metrics
        self.metrics_to_report = config['reporting']['metrics_to_report']
        self.reports = {}
        
        logger.info("Reporter initialized with %d metrics", len(self.metrics_to_report))

    def execute(self, checkpoint=None):
        """
        Execute report generation.
        
        Args:
            checkpoint (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Reporting results
        """
        logger.info("Starting report generation")
        
        # Verify data consistency
        if not self._verify_data_consistency():
            logger.warning("Data consistency check found issues, but will attempt to continue with report generation")
        
        with Timer() as timer:
            # Create reports directory
            reports_dir = Path(self.config['system']['output_dir']) / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate summary report
            self.reports['summary'] = self._generate_summary_report(reports_dir)
            
            # Generate classification report
            self.reports['classification'] = self._generate_classification_report(reports_dir)
            
            # Generate misclassified pairs report
            self.reports['misclassified'] = self._generate_misclassified_report(reports_dir)
            
            # Generate feature importance report
            self.reports['feature_importance'] = self._generate_feature_importance_report(reports_dir)
            
            # Generate feature analysis report
            self.reports['feature_analysis'] = self._generate_feature_analysis_report(reports_dir)
            
            # Generate RFE analysis report
            self.reports['rfe_analysis'] = self._generate_rfe_analysis_report(reports_dir)
            
            # Generate clustering report
            self.reports['clustering'] = self._generate_clustering_report(reports_dir)
            
            # Generate performance report
            self.reports['performance'] = self._generate_performance_report(reports_dir)
            
            # Generate data quality report
            self.reports['data_quality'] = self._generate_data_quality_report(reports_dir)
            
            # Generate comprehensive HTML report
            self._generate_html_report(reports_dir)
        
        logger.info("Report generation completed in %.2f seconds", timer.duration)
        
        # Save report metadata
        self._save_report_metadata(reports_dir)
        
        # Count successful reports
        successful_reports = sum(1 for report in self.reports.values() if report)
        
        return {
            'reports_generated': successful_reports,
            'reports_directory': str(reports_dir),
            'duration': timer.duration
        }

    def _generate_summary_report(self, reports_dir):
        """
        Generate summary report of pipeline execution.
        
        Args:
            reports_dir (Path): Reports directory
            
        Returns:
            dict: Summary report information
        """
        try:
            # Load pipeline summary
            output_dir = Path(self.config['system']['output_dir'])
            summary_path = output_dir / "pipeline_summary.json"
            
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    pipeline_summary = json.load(f)
            else:
                pipeline_summary = {
                    'duration': 0,
                    'stages': {},
                    'mode': self.config['system']['mode'],
                    'timestamp': time.time()
                }
            
            # Create summary report
            summary = {
                'execution_date': datetime.fromtimestamp(pipeline_summary.get('timestamp', time.time())).strftime('%Y-%m-%d %H:%M:%S'),
                'total_duration': pipeline_summary.get('duration', 0),
                'mode': pipeline_summary.get('mode', self.config['system']['mode']),
                'stages': []
            }
            
            # Add stage information
            for stage_name, stage_metrics in pipeline_summary.get('stages', {}).items():
                summary['stages'].append({
                    'name': stage_name,
                    'duration': stage_metrics.get('duration', 0),
                    'records_processed': stage_metrics.get('records_processed', 0)
                })
            
            # Sort stages by execution order
            stage_order = ['preprocess', 'embed', 'index', 'impute', 'query', 'features', 'classify', 'cluster', 'analyze', 'report']
            summary['stages'] = sorted(summary['stages'], key=lambda x: stage_order.index(x['name']) if x['name'] in stage_order else 999)
            
            # Write summary to CSV
            summary_path = reports_dir / "pipeline_summary.csv"
            with open(summary_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Stage', 'Duration (s)', 'Records Processed'])
                
                for stage in summary['stages']:
                    writer.writerow([stage['name'], f"{stage['duration']:.2f}", stage['records_processed']])
            
            # Generate summary plot
            self._plot_stage_durations(summary['stages'], reports_dir / "stage_durations.png")
            
            logger.info("Summary report saved to %s", summary_path)
            return summary
        
        except Exception as e:
            logger.error("Error generating summary report: %s", str(e))
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def _generate_classification_report(self, reports_dir):
        """
        Generate classification performance report with proper array alignment.
        
        Args:
            reports_dir (Path): Reports directory
                
        Returns:
            dict: Classification report information
        """
        try:
            logger.info("Generating classification report")
            
            # Load classification metrics and predictions/labels
            output_dir = Path(self.config['system']['output_dir'])
            metrics_path = output_dir / "classification_metrics.json"
            predictions_path = output_dir / "predictions.npy"
            labels_path = output_dir / "labels.npy"
            feature_vectors_path = output_dir / "feature_vectors.npy"
            test_indices_path = output_dir / "test_indices.npy"
            
            # Initialize report
            report = {}
            
            # Load pre-computed metrics if available
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    classification_metrics = json.load(f)
                
                report['metrics'] = classification_metrics
                report['confusion_matrix'] = {
                    'true_positives': classification_metrics.get('true_positives', 0),
                    'false_positives': classification_metrics.get('false_positives', 0),
                    'true_negatives': classification_metrics.get('true_negatives', 0),
                    'false_negatives': classification_metrics.get('false_negatives', 0)
                }
            
            # Load predictions and labels if available to compute metrics
            predictions = None
            labels = None
            
            if predictions_path.exists() and labels_path.exists():
                try:
                    predictions = np.load(predictions_path)
                    labels = np.load(labels_path)
                    
                    # Check if we need to align arrays
                    if len(predictions) != len(labels):
                        logger.warning(f"Predictions ({len(predictions)}) and labels ({len(labels)}) have different lengths")
                        
                        # Try to use test indices
                        if test_indices_path.exists():
                            test_indices = np.load(test_indices_path)
                            logger.info(f"Using test indices to align arrays. Test set size: {len(test_indices)}")
                            if len(labels) > len(test_indices):
                                # Assume labels contain both training and test data
                                labels = labels[test_indices]
                            else:
                                logger.warning("Cannot use test indices, array sizes don't match expectations")
                        
                        # If still mismatched, truncate to the smaller size
                        if len(predictions) != len(labels):
                            logger.warning("Truncating arrays to match lengths")
                            min_len = min(len(predictions), len(labels))
                            predictions = predictions[:min_len]
                            labels = labels[:min_len]
                    
                    # Ensure predictions and labels are binary for metrics calculation only
                    predictions_binary = (predictions > 0.5).astype(int) if predictions.dtype != 'int64' else predictions
                    labels_binary = (labels > 0.5).astype(int) if labels.dtype != 'int64' else labels
                    
                    # Compute metrics
                    precision = precision_score(labels_binary, predictions_binary)
                    recall = recall_score(labels_binary, predictions_binary)
                    f1 = f1_score(labels_binary, predictions_binary)
                    accuracy = accuracy_score(labels_binary, predictions_binary)
                    
                    # Compute confusion matrix components
                    cm = confusion_matrix(labels_binary, predictions_binary)
                    tn, fp, fn, tp = cm.ravel()
                    
                    # Update or create report metrics
                    report['metrics'] = {
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1': float(f1),
                        'accuracy': float(accuracy)
                    }
                    
                    report['confusion_matrix'] = {
                        'true_positives': int(tp),
                        'false_positives': int(fp),
                        'true_negatives': int(tn),
                        'false_negatives': int(fn)
                    }
                    
                    # Save computed metrics for future use
                    report['metrics'].update({
                        'true_positives': int(tp),
                        'false_positives': int(fp),
                        'true_negatives': int(tn),
                        'false_negatives': int(fn)
                    })
                    
                    # Load feature vectors if available to recreate the original DataFrame
                    feature_vectors = None
                    if feature_vectors_path.exists():
                        try:
                            feature_vectors = np.load(feature_vectors_path)
                            logger.info(f"Loaded feature vectors with shape: {feature_vectors.shape}")
                        except Exception as e:
                            logger.error(f"Error loading feature vectors: {e}")
                            
                    # Create DataFrame with feature data if available
                    if feature_vectors is not None and len(feature_vectors) == len(predictions):
                        # Create DataFrame with feature vectors and labels
                        df = pd.DataFrame(feature_vectors, columns=[f"feature_{i}" for i in range(feature_vectors.shape[1])])
                        df['true_label'] = labels
                        df['predicted_label'] = predictions_binary
                        df['prediction_confidence'] = predictions  # This preserves the actual probability values
                        
                        # Save the complete DataFrame with all features to CSV
                        classified_pairs_path = reports_dir / "classified_pairs.csv"
                        df.to_csv(classified_pairs_path, index=False)
                        logger.info(f"Saved classified pairs with {feature_vectors.shape[1]} features to {classified_pairs_path}")
                    else:
                        logger.warning("Feature vectors not available or dimensions don't match predictions")
                        # Create simplified DataFrame with just labels and predictions
                        df = pd.DataFrame({
                            'true_label': labels,
                            'predicted_label': predictions_binary,
                            'prediction_confidence': predictions
                        })
                        classified_pairs_path = reports_dir / "classified_pairs.csv"
                        df.to_csv(classified_pairs_path, index=False)
                        logger.info(f"Saved simplified classified pairs to {classified_pairs_path}")
                    
                    with open(reports_dir / "computed_metrics.json", 'w') as f:
                        json.dump(report['metrics'], f, indent=2)
                    
                except Exception as e:
                    logger.error(f"Error computing metrics from predictions and labels: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            else:
                logger.warning("Predictions or labels files not found, using pre-computed metrics only")
            
            # Early return if no metrics available
            if not report.get('metrics'):
                logger.error("No classification metrics available")
                return {}
            
            # Write metrics to CSV
            metrics_csv_path = reports_dir / "classification_metrics.csv"
            with open(metrics_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value'])
                for metric, value in report['metrics'].items():
                    writer.writerow([metric, f"{value:.4f}" if isinstance(value, float) else value])
            
            logger.info(f"Classification metrics saved to {metrics_csv_path}")
            
            # Generate confusion matrix plot
            if report.get('confusion_matrix'):
                cm_path = reports_dir / "confusion_matrix.png"
                self._plot_confusion_matrix(report['confusion_matrix'], cm_path)
                logger.info(f"Confusion matrix plot saved to {cm_path}")
            
            # Generate ROC curve if probabilities are available
            if predictions is not None and labels is not None and predictions.dtype == float:
                try:
                    roc_path = reports_dir / "roc_curve.png"
                    self._plot_roc_curve(labels, predictions, roc_path)
                    logger.info(f"ROC curve plot saved to {roc_path}")
                except Exception as e:
                    logger.error(f"Error generating ROC curve: {e}")
            
            return report
        
        except Exception as e:
            logger.error("Error generating classification report: %s", str(e))
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def _generate_misclassified_report(self, reports_dir):
        """
        Generate report of misclassified pairs with proper array alignment.
        
        Args:
            reports_dir (Path): Reports directory
            
        Returns:
            dict: Misclassified pairs report information
        """
        try:
            logger.info("Generating misclassified pairs report")
            
            # Load necessary data
            output_dir = Path(self.config['system']['output_dir'])
            candidate_pairs_path = output_dir / "candidate_pairs.json"
            pair_ids_path = output_dir / "pair_ids.json"  # Optional file with pair IDs
            predictions_path = output_dir / "predictions.npy"
            labels_path = output_dir / "labels.npy"
            
            # Early return if files don't exist
            if not (predictions_path.exists() and labels_path.exists()):
                logger.warning("Predictions or labels files not found, skipping misclassified report")
                return {}
            
            # Load predictions and labels
            predictions = np.load(predictions_path)
            labels = np.load(labels_path)
            
            # Store original prediction confidence values before converting to binary
            prediction_confidence = predictions.copy()
            
            # Convert to binary predictions for classification analysis
            if predictions.dtype != 'int64':
                predictions = (predictions > 0.5).astype(int)
            
            # Create DataFrame with available data
            df = None
            
            # Try loading pair data if available
            if candidate_pairs_path.exists():
                try:
                    with open(candidate_pairs_path, 'r') as f:
                        pairs = json.load(f)
                    
                    if len(pairs) == len(predictions) == len(labels):
                        # Perfect match, use all data
                        df = pd.DataFrame(pairs)
                        df['predicted_label'] = predictions
                        df['prediction_confidence'] = prediction_confidence
                        df['true_label'] = labels
                    elif pair_ids_path.exists():
                        # Try using pair IDs for alignment
                        with open(pair_ids_path, 'r') as f:
                            pair_ids = json.load(f)
                        
                        # Create dictionary lookup from pairs data
                        pairs_dict = {pair['pair_id'] if 'pair_id' in pair else f"{pair.get('record1_id', '')}|{pair.get('record2_id', '')}": pair for pair in pairs}
                        
                        # Create aligned dataframe
                        aligned_pairs = [pairs_dict.get(pair_id, {}) for pair_id in pair_ids]
                        df = pd.DataFrame(aligned_pairs)
                        
                        # Ensure length matches
                        min_len = min(len(df), len(predictions), len(labels))
                        df = df.iloc[:min_len]
                        df['predicted_label'] = predictions[:min_len]
                        df['prediction_confidence'] = prediction_confidence[:min_len]
                        df['true_label'] = labels[:min_len]
                    else:
                        # Create minimal dataframe with predictions and labels
                        min_len = min(len(predictions), len(labels))
                        df = pd.DataFrame({
                            'index': range(min_len),
                            'predicted_label': predictions[:min_len],
                            'prediction_confidence': prediction_confidence[:min_len],
                            'true_label': labels[:min_len]
                        })
                except Exception as e:
                    logger.error(f"Error loading pair data: {e}")
            
            # If still no dataframe, create minimal one
            if df is None:
                min_len = min(len(predictions), len(labels))
                df = pd.DataFrame({
                    'index': range(min_len),
                    'predicted_label': predictions[:min_len],
                    'prediction_confidence': prediction_confidence[:min_len],
                    'true_label': labels[:min_len]
                })
            
            # Find misclassified pairs
            misclassified = df[df['predicted_label'] != df['true_label']]
            
            # Calculate misclassification statistics
            false_positives = df[(df['predicted_label'] == 1) & (df['true_label'] == 0)]
            false_negatives = df[(df['predicted_label'] == 0) & (df['true_label'] == 1)]
            
            stats = {
                'total_pairs': len(df),
                'misclassified_count': len(misclassified),
                'misclassification_rate': len(misclassified) / len(df) if len(df) > 0 else 0,
                'false_positives': len(false_positives),
                'false_negatives': len(false_negatives)
            }
            
            # Write misclassified pairs to CSV
            misclassified_path = reports_dir / "misclassified_pairs.csv"
            misclassified.to_csv(misclassified_path, index=False)
            
            # Write statistics to JSON
            stats_path = reports_dir / "misclassification_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Misclassified pairs report saved to {misclassified_path}")
            logger.info(f"Misclassification stats: {stats['misclassified_count']} pairs ({stats['misclassification_rate']:.2%})")
            
            return {
                'stats': stats,
                'sample_misclassified': misclassified.head(10).to_dict('records') if len(misclassified) > 0 else []
            }
        
        except Exception as e:
            logger.error("Error generating misclassified pairs report: %s", str(e))
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def _generate_feature_importance_report(self, reports_dir):
        """
        Generate feature importance report.
        
        Args:
            reports_dir (Path): Reports directory
            
        Returns:
            dict: Feature importance report information
        """
        try:
            # Load feature importance
            output_dir = Path(self.config['system']['output_dir'])
            importance_path = output_dir / "feature_importance.json"
            
            if importance_path.exists():
                with open(importance_path, 'r') as f:
                    feature_importance = json.load(f)
            else:
                logger.warning("Feature importance file not found")
                return {}
            
            # Create feature importance report
            report = {
                'importance': feature_importance
            }
            
            # Write feature importance to CSV
            importance_path = reports_dir / "feature_importance.csv"
            with open(importance_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Feature', 'Importance'])
                
                for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
                    writer.writerow([feature, f"{importance:.6f}"])
            
            # Generate feature importance plot
            plot_path = reports_dir / "feature_importance.png"
            self._plot_feature_importance(feature_importance, plot_path)
            
            logger.info(f"Feature importance report saved to {importance_path}")
            return report
        
        except Exception as e:
            logger.error("Error generating feature importance report: %s", str(e))
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def _generate_feature_analysis_report(self, reports_dir):
        """
        Generate feature analysis report.
        
        Args:
            reports_dir (Path): Reports directory
            
        Returns:
            dict: Feature analysis report information
        """
        try:
            logger.info("Generating feature analysis report")
            
            # Load feature vectors and names
            output_dir = Path(self.config['system']['output_dir'])
            vectors_path = output_dir / "feature_vectors.npy"
            names_path = output_dir / "feature_names.json"
            
            if not (vectors_path.exists() and names_path.exists()):
                logger.warning("Feature vectors or names not found")
                return {}
            
            # Load data
            feature_vectors = np.load(vectors_path)
            with open(names_path, 'r') as f:
                feature_names = json.load(f)
            
            # Ensure same number of features
            if feature_vectors.shape[1] != len(feature_names):
                logger.warning(f"Feature count mismatch: {feature_vectors.shape[1]} features in vectors, {len(feature_names)} in names")
                # Truncate longer list
                if feature_vectors.shape[1] > len(feature_names):
                    feature_vectors = feature_vectors[:, :len(feature_names)]
                else:
                    feature_names = feature_names[:feature_vectors.shape[1]]
            
            # Create DataFrame for analysis
            df = pd.DataFrame(feature_vectors, columns=feature_names)
            
            # Calculate basic statistics
            feature_stats = df.describe().to_dict()
            
            # Find constant features
            constant_features = [col for col in df.columns if df[col].nunique() <= 1]
            
            # Calculate correlation matrix (with error handling for constant features)
            correlation_matrix = {}
            non_constant_cols = [col for col in df.columns if col not in constant_features]
            
            if non_constant_cols:
                try:
                    corr_df = df[non_constant_cols].corr().fillna(0)
                    correlation_matrix = corr_df.to_dict()
                except Exception as e:
                    logger.error(f"Error calculating correlation matrix: {e}")
            
            # Find highly correlated features
            highly_correlated = []
            for i, feature1 in enumerate(non_constant_cols):
                for feature2 in non_constant_cols[i+1:]:
                    if feature1 in correlation_matrix and feature2 in correlation_matrix[feature1]:
                        correlation = correlation_matrix[feature1][feature2]
                        if abs(correlation) > 0.8:
                            highly_correlated.append({
                                'feature1': feature1,
                                'feature2': feature2,
                                'correlation': correlation
                            })
            
            # Check for outliers using z-score
            outliers_by_feature = {}
            for feature in feature_names:
                if feature not in constant_features:
                    z_scores = np.abs((df[feature] - df[feature].mean()) / df[feature].std())
                    outliers = np.where(z_scores > 3)[0]
                    if len(outliers) > 0:
                        outliers_by_feature[feature] = len(outliers)
            
            # Create report
            report = {
                'feature_count': len(feature_names),
                'sample_count': len(feature_vectors),
                'constant_features': constant_features,
                'highly_correlated_pairs': highly_correlated,
                'outlier_counts': outliers_by_feature,
                'statistics': {
                    feature: {
                        'mean': feature_stats[feature]['mean'],
                        'std': feature_stats[feature]['std'],
                        'min': feature_stats[feature]['min'],
                        'max': feature_stats[feature]['max']
                    } for feature in feature_names
                }
            }
            
            # Save report to JSON
            report_path = reports_dir / "feature_analysis.json"
            with open(report_path, 'w') as f:
                # Convert NumPy values to Python types
                def convert_numpy(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert_numpy(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy(i) for i in obj]
                    else:
                        return obj
                
                json.dump(convert_numpy(report), f, indent=2)
            
            # Generate correlation heatmap
            if non_constant_cols and len(non_constant_cols) > 1:
                try:
                    corr_path = reports_dir / "feature_correlation.png"
                    self._plot_correlation_heatmap(df[non_constant_cols], corr_path)
                except Exception as e:
                    logger.error(f"Error generating correlation heatmap: {e}")
            
            logger.info(f"Feature analysis report saved to {report_path}")
            return report
        
        except Exception as e:
            logger.error("Error generating feature analysis report: %s", str(e))
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def _generate_rfe_analysis_report(self, reports_dir):
        """
        Generate RFE analysis report with feature mapping.
        
        Args:
            reports_dir (Path): Reports directory
            
        Returns:
            dict: RFE analysis report information
        """
        try:
            logger.info("Generating RFE analysis report")
            
            # Load feature importance
            output_dir = Path(self.config['system']['output_dir'])
            selected_features_path = output_dir / "selected_features.json"
            feature_names_path = output_dir / "feature_names.json"
            
            if not selected_features_path.exists():
                logger.warning("Selected features file not found")
                return {}
            
            # Load data
            with open(selected_features_path, 'r') as f:
                selected_features = json.load(f)
            
            # Load current feature names if available
            current_feature_names = []
            if feature_names_path.exists():
                with open(feature_names_path, 'r') as f:
                    current_feature_names = json.load(f)
            
            # Extract RFE information
            rfe_indices = selected_features.get('indices', [])
            rfe_names = selected_features.get('names', [])
            
            # Check for feature count mismatch
            if current_feature_names and len(rfe_indices) != len(current_feature_names):
                logger.warning(f"Feature count mismatch: RFE was performed with {len(rfe_indices)} features, but current feature set has {len(current_feature_names)} features")
                
                # Map RFE indices to current features if possible
                mapped_indices = []
                mapped_names = []
                
                if rfe_names:
                    # Try to map by name
                    for name in rfe_names:
                        if name in current_feature_names:
                            idx = current_feature_names.index(name)
                            mapped_indices.append(idx)
                            mapped_names.append(name)
                    
                    if mapped_indices:
                        logger.info(f"Mapped {len(mapped_indices)}/{len(rfe_names)} RFE features to current feature set")
                        rfe_indices = mapped_indices
                        rfe_names = mapped_names
            
            # Create report
            report = {
                'rfe_feature_count': len(rfe_indices),
                'selected_indices': rfe_indices,
                'selected_names': rfe_names if rfe_names else [f"feature_{i}" for i in rfe_indices],
                'current_feature_count': len(current_feature_names)
            }
            
            # Save report to JSON
            report_path = reports_dir / "rfe_analysis.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Write selected features to CSV
            csv_path = reports_dir / "selected_features.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Index', 'Feature'])
                
                for i, name in enumerate(report['selected_names']):
                    writer.writerow([report['selected_indices'][i], name])
            
            logger.info(f"RFE analysis report saved to {report_path}")
            return report
        
        except Exception as e:
            logger.error("Error generating RFE analysis report: %s", str(e))
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def _generate_clustering_report(self, reports_dir):
        """
        Generate clustering report.
        
        Args:
            reports_dir (Path): Reports directory
            
        Returns:
            dict: Clustering report information
        """
        try:
            # Load clustering metrics
            output_dir = Path(self.config['system']['output_dir'])
            metrics_path = output_dir / "clustering_metrics.json"
            clusters_path = output_dir / "entity_clusters.json"
            
            if not metrics_path.exists():
                logger.warning("Clustering metrics file not found")
                return {}
            
            # Load metrics
            with open(metrics_path, 'r') as f:
                clustering_metrics = json.load(f)
            
            # Try to load actual clusters if available
            clusters = []
            if clusters_path.exists():
                with open(clusters_path, 'r') as f:
                    try:
                        clusters = json.load(f)
                    except Exception as e:
                        logger.error(f"Error loading clusters: {e}")
            
            # Create clustering report
            report = {
                'metrics': clustering_metrics,
                'cluster_count': clustering_metrics.get('cluster_count', 0),
                'has_clusters': len(clusters) > 0
            }
            
            # Calculate additional metrics from clusters
            if clusters:
                # Cluster size distribution
                sizes = [len(cluster) for cluster in clusters]
                size_distribution = {}
                for size in sizes:
                    size_distribution[size] = size_distribution.get(size, 0) + 1
                
                # Add to report
                report['cluster_sizes'] = sizes
                report['size_distribution'] = size_distribution
                report['largest_clusters'] = sorted([(i, len(cluster)) for i, cluster in enumerate(clusters)], 
                                                  key=lambda x: x[1], reverse=True)[:10]
            
            # Write cluster statistics to CSV
            stats_path = reports_dir / "cluster_statistics.csv"
            with open(stats_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Statistic', 'Value'])
                for key, value in clustering_metrics.items():
                    if key != 'size_distribution':  # Skip complex nested structure
                        writer.writerow([key, value])
            
            # Write cluster size distribution to CSV if available
            if 'size_distribution' in clustering_metrics:
                dist_path = reports_dir / "cluster_size_distribution.csv"
                with open(dist_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Cluster Size', 'Count'])
                    
                    for size, count in sorted(clustering_metrics['size_distribution'].items(), key=lambda x: int(x[0])):
                        writer.writerow([size, count])
            
            # Generate cluster size distribution plot
            if clusters:
                plot_path = reports_dir / "cluster_size_distribution.png"
                self._plot_cluster_size_distribution(clusters, plot_path)
            
            logger.info(f"Clustering report saved to {stats_path}")
            return report
        
        except Exception as e:
            logger.error("Error generating clustering report: %s", str(e))
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def _generate_performance_report(self, reports_dir):
        """
        Generate performance report.
        
        Args:
            reports_dir (Path): Reports directory
            
        Returns:
            dict: Performance report information
        """
        try:
            # Load pipeline summary
            output_dir = Path(self.config['system']['output_dir'])
            summary_path = output_dir / "pipeline_summary.json"
            
            if not summary_path.exists():
                logger.warning("Pipeline summary file not found")
                return {}
            
            # Load summary
            with open(summary_path, 'r') as f:
                pipeline_summary = json.load(f)
            
            # Create performance report
            report = {
                'total_duration': pipeline_summary.get('duration', 0),
                'stage_durations': {
                    stage: metrics.get('duration', 0)
                    for stage, metrics in pipeline_summary.get('stages', {}).items()
                },
                'throughput': {
                    stage: metrics.get('records_processed', 0) / metrics.get('duration', 1)
                    for stage, metrics in pipeline_summary.get('stages', {}).items()
                    if metrics.get('duration', 0) > 0 and metrics.get('records_processed', 0) > 0
                }
            }
            
            # Write performance metrics to CSV
            performance_path = reports_dir / "performance_metrics.csv"
            with open(performance_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Stage', 'Duration (s)', 'Records Processed', 'Throughput (records/s)'])
                
                for stage, metrics in pipeline_summary.get('stages', {}).items():
                    duration = metrics.get('duration', 0)
                    records = metrics.get('records_processed', 0)
                    throughput = records / duration if duration > 0 else 0
                    
                    writer.writerow([stage, f"{duration:.2f}", records, f"{throughput:.2f}"])
            
            # Generate performance visualization
            plot_path = reports_dir / "performance_comparison.png"
            self._plot_performance_comparison(report, plot_path)
            
            logger.info(f"Performance report saved to {performance_path}")
            return report
        
        except Exception as e:
            logger.error("Error generating performance report: %s", str(e))
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def _generate_data_quality_report(self, reports_dir):
        """
        Generate data quality report.
        
        Args:
            reports_dir (Path): Reports directory
            
        Returns:
            dict: Data quality report information
        """
        try:
            # Load field statistics
            output_dir = Path(self.config['system']['output_dir'])
            stats_path = output_dir / "field_statistics.json"
            
            if not stats_path.exists():
                logger.warning("Field statistics file not found")
                return {}
            
            # Load statistics
            with open(stats_path, 'r') as f:
                field_stats = json.load(f)
            
            # Create data quality report
            report = {
                'field_statistics': field_stats,
                'total_fields': len(field_stats),
                'overall_stats': {
                    'total_unique_values': sum(stats.get('unique_values', 0) for stats in field_stats.values()),
                    'total_occurrences': sum(stats.get('total_occurrences', 0) for stats in field_stats.values())
                }
            }
            
            # Add coverage metrics
            coverage = {}
            for field, stats in field_stats.items():
                unique = stats.get('unique_values', 0)
                occurrences = stats.get('total_occurrences', 0)
                
                if occurrences > 0:
                    coverage[field] = {
                        'unique_percentage': unique / occurrences if occurrences > 0 else 0,
                        'unique': unique,
                        'occurrences': occurrences
                    }
            
            report['coverage'] = coverage
            
            # Write field statistics to CSV
            stats_path = reports_dir / "field_statistics.csv"
            with open(stats_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Field', 'Total Occurrences', 'Unique Values', 'Unique Percentage'])
                
                for field, stats in sorted(field_stats.items()):
                    occurrences = stats.get('total_occurrences', 0)
                    unique = stats.get('unique_values', 0)
                    unique_pct = unique / occurrences if occurrences > 0 else 0
                    
                    writer.writerow([
                        field,
                        occurrences,
                        unique,
                        f"{unique_pct:.2%}"
                    ])
            
            # Generate data quality visualization
            plot_path = reports_dir / "field_distribution.png"
            self._plot_field_distribution(field_stats, plot_path)
            
            logger.info(f"Data quality report saved to {stats_path}")
            return report
        
        except Exception as e:
            logger.error("Error generating data quality report: %s", str(e))
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def _generate_html_report(self, reports_dir):
        """
        Generate comprehensive HTML report.
        
        Args:
            reports_dir (Path): Reports directory
        """
        try:
            # Create HTML report
            html_path = reports_dir / "entity_resolution_report.html"
            
            with open(html_path, 'w') as f:
                # Write HTML header
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Entity Resolution Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1 {{ color: #2c3e50; }}
                        h2 {{ color: #3498db; margin-top: 30px; }}
                        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                        .metrics {{ display: flex; flex-wrap: wrap; }}
                        .metric-box {{ background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px; padding: 15px; margin: 10px; min-width: 200px; }}
                        .metric-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
                        .metric-name {{ font-size: 14px; color: #7f8c8d; }}
                        img {{ max-width: 100%; height: auto; margin: 20px 0; }}
                    </style>
                </head>
                <body>
                    <h1>Entity Resolution Pipeline Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                """)
                
                # Write summary section
                f.write("""
                    <h2>Pipeline Summary</h2>
                """)
                
                if 'summary' in self.reports and self.reports['summary']:
                    summary = self.reports['summary']
                    
                    f.write(f"""
                    <div class="metrics">
                        <div class="metric-box">
                            <div class="metric-value">{summary.get('total_duration', 0):.2f}s</div>
                            <div class="metric-name">Total Duration</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{summary.get('mode', 'dev')}</div>
                            <div class="metric-name">Execution Mode</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{len(summary.get('stages', []))}</div>
                            <div class="metric-name">Pipeline Stages</div>
                        </div>
                    </div>
                    """)
                    
                    # Write stage durations table
                    f.write("""
                    <h3>Stage Durations</h3>
                    <table>
                        <tr>
                            <th>Stage</th>
                            <th>Duration (s)</th>
                            <th>Records Processed</th>
                        </tr>
                    """)
                    
                    for stage in summary.get('stages', []):
                        f.write(f"""
                        <tr>
                            <td>{stage.get('name', '')}</td>
                            <td>{stage.get('duration', 0):.2f}</td>
                            <td>{stage.get('records_processed', 0)}</td>
                        </tr>
                        """)
                    
                    f.write("</table>")
                    
                    # Include stage durations plot
                    if (reports_dir / "stage_durations.png").exists():
                        f.write("""
                        <img src="stage_durations.png" alt="Stage Durations">
                        """)
                
                # Write classification section
                f.write("""
                    <h2>Classification Performance</h2>
                """)
                
                if 'classification' in self.reports and self.reports['classification']:
                    classification = self.reports['classification']
                    metrics = classification.get('metrics', {})
                    
                    f.write(f"""
                    <div class="metrics">
                        <div class="metric-box">
                            <div class="metric-value">{metrics.get('precision', 0):.4f}</div>
                            <div class="metric-name">Precision</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{metrics.get('recall', 0):.4f}</div>
                            <div class="metric-name">Recall</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{metrics.get('f1', 0):.4f}</div>
                            <div class="metric-name">F1 Score</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{metrics.get('accuracy', 0):.4f}</div>
                            <div class="metric-name">Accuracy</div>
                        </div>
                    </div>
                    """)
                    
                    # Include confusion matrix plot
                    if (reports_dir / "confusion_matrix.png").exists():
                        f.write("""
                        <h3>Confusion Matrix</h3>
                        <img src="confusion_matrix.png" alt="Confusion Matrix">
                        """)
                    
                    # Include ROC curve if available
                    if (reports_dir / "roc_curve.png").exists():
                        f.write("""
                        <h3>ROC Curve</h3>
                        <img src="roc_curve.png" alt="ROC Curve">
                        """)
                
                # Write misclassification section if available
                if 'misclassified' in self.reports and self.reports['misclassified']:
                    misclassified = self.reports['misclassified']
                    stats = misclassified.get('stats', {})
                    
                    f.write("""
                    <h2>Misclassification Analysis</h2>
                    """)
                    
                    f.write(f"""
                    <div class="metrics">
                        <div class="metric-box">
                            <div class="metric-value">{stats.get('misclassified_count', 0)}</div>
                            <div class="metric-name">Misclassified Pairs</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{stats.get('misclassification_rate', 0):.2%}</div>
                            <div class="metric-name">Error Rate</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{stats.get('false_positives', 0)}</div>
                            <div class="metric-name">False Positives</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{stats.get('false_negatives', 0)}</div>
                            <div class="metric-name">False Negatives</div>
                        </div>
                    </div>
                    """)
                
                # Write feature importance section
                f.write("""
                    <h2>Feature Importance</h2>
                """)
                
                if 'feature_importance' in self.reports and self.reports['feature_importance']:
                    importance = self.reports['feature_importance'].get('importance', {})
                    
                    # Write feature importance table
                    f.write("""
                    <table>
                        <tr>
                            <th>Feature</th>
                            <th>Importance</th>
                        </tr>
                    """)
                    
                    for feature, importance_value in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]:
                        f.write(f"""
                        <tr>
                            <td>{feature}</td>
                            <td>{importance_value:.6f}</td>
                        </tr>
                        """)
                    
                    f.write("</table>")
                    
                    # Include feature importance plot
                    if (reports_dir / "feature_importance.png").exists():
                        f.write("""
                        <img src="feature_importance.png" alt="Feature Importance">
                        """)
                
                # Write feature analysis section
                if 'feature_analysis' in self.reports and self.reports['feature_analysis']:
                    feature_analysis = self.reports['feature_analysis']
                    
                    f.write("""
                    <h2>Feature Analysis</h2>
                    """)
                    
                    # Basic metrics
                    f.write(f"""
                    <p>
                        Analyzed {feature_analysis.get('feature_count', 0)} features across
                        {feature_analysis.get('sample_count', 0)} samples.
                    </p>
                    """)
                    
                    # Constant features
                    constant_features = feature_analysis.get('constant_features', [])
                    if constant_features:
                        f.write(f"""
                        <h3>Constant Features</h3>
                        <p>Found {len(constant_features)} features with constant values:</p>
                        <ul>
                        """)
                        
                        for feature in constant_features[:10]:  # Limit to first 10
                            f.write(f"<li>{feature}</li>")
                        
                        if len(constant_features) > 10:
                            f.write(f"<li>... and {len(constant_features) - 10} more</li>")
                        
                        f.write("</ul>")
                    
                    # Highly correlated features
                    correlated_pairs = feature_analysis.get('highly_correlated_pairs', [])
                    if correlated_pairs:
                        f.write(f"""
                        <h3>Highly Correlated Features</h3>
                        <p>Found {len(correlated_pairs)} pairs of highly correlated features:</p>
                        <table>
                            <tr>
                                <th>Feature 1</th>
                                <th>Feature 2</th>
                                <th>Correlation</th>
                            </tr>
                        """)
                        
                        for pair in correlated_pairs[:10]:  # Limit to first 10
                            f.write(f"""
                            <tr>
                                <td>{pair.get('feature1', '')}</td>
                                <td>{pair.get('feature2', '')}</td>
                                <td>{pair.get('correlation', 0):.4f}</td>
                            </tr>
                            """)
                        
                        f.write("</table>")
                    
                    # Include correlation heatmap if available
                    if (reports_dir / "feature_correlation.png").exists():
                        f.write("""
                        <h3>Feature Correlation Heatmap</h3>
                        <img src="feature_correlation.png" alt="Feature Correlation">
                        """)
                
                # Write clustering section
                f.write("""
                    <h2>Clustering Results</h2>
                """)
                
                if 'clustering' in self.reports and self.reports['clustering']:
                    clustering = self.reports['clustering']
                    metrics = clustering.get('metrics', {})
                    
                    f.write(f"""
                    <div class="metrics">
                        <div class="metric-box">
                            <div class="metric-value">{metrics.get('cluster_count', 0)}</div>
                            <div class="metric-name">Total Clusters</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{metrics.get('total_entities', 0)}</div>
                            <div class="metric-name">Total Entities</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{metrics.get('singleton_clusters', 0)}</div>
                            <div class="metric-name">Singleton Clusters</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{metrics.get('max_cluster_size', 0)}</div>
                            <div class="metric-name">Maximum Cluster Size</div>
                        </div>
                    </div>
                    """)
                    
                    # Include cluster size distribution plot
                    if (reports_dir / "cluster_size_distribution.png").exists():
                        f.write("""
                        <h3>Cluster Size Distribution</h3>
                        <img src="cluster_size_distribution.png" alt="Cluster Size Distribution">
                        """)
                
                # Write data quality section
                f.write("""
                    <h2>Data Quality</h2>
                """)
                
                if 'data_quality' in self.reports and self.reports['data_quality']:
                    data_quality = self.reports['data_quality']
                    field_stats = data_quality.get('field_statistics', {})
                    
                    # Write field statistics table
                    f.write("""
                    <table>
                        <tr>
                            <th>Field</th>
                            <th>Total Occurrences</th>
                            <th>Unique Values</th>
                            <th>Unique %</th>
                        </tr>
                    """)
                    
                    for field, stats in sorted(field_stats.items()):
                        occurrences = stats.get('total_occurrences', 0)
                        unique = stats.get('unique_values', 0)
                        unique_pct = (unique / occurrences * 100) if occurrences > 0 else 0
                        
                        f.write(f"""
                        <tr>
                            <td>{field}</td>
                            <td>{occurrences}</td>
                            <td>{unique}</td>
                            <td>{unique_pct:.2f}%</td>
                        </tr>
                        """)
                    
                    f.write("</table>")
                    
                    # Include field distribution plot
                    if (reports_dir / "field_distribution.png").exists():
                        f.write("""
                        <h3>Field Distribution</h3>
                        <img src="field_distribution.png" alt="Field Distribution">
                        """)
                
                # Write performance section
                f.write("""
                    <h2>Performance Analysis</h2>
                """)
                
                if 'performance' in self.reports and self.reports['performance']:
                    performance = self.reports['performance']
                    
                    f.write(f"""
                    <p>Total pipeline duration: {performance.get('total_duration', 0):.2f} seconds</p>
                    """)
                    
                    # Include performance comparison plot
                    if (reports_dir / "performance_comparison.png").exists():
                        f.write("""
                        <h3>Performance Comparison</h3>
                        <img src="performance_comparison.png" alt="Performance Comparison">
                        """)
                
                # Close HTML document
                f.write("""
                </body>
                </html>
                """)
            
            logger.info("Generated HTML report: %s", html_path)
        
        except Exception as e:
            logger.error("Error generating HTML report: %s", str(e))
            import traceback
            logger.error(traceback.format_exc())

    def _plot_stage_durations(self, stages, output_path):
        """
        Generate plot of pipeline stage durations.
        
        Args:
            stages (list): List of stage information
            output_path (Path): Output file path
        """
        try:
            # Extract stage names and durations
            stage_names = [stage.get('name', '') for stage in stages]
            durations = [stage.get('duration', 0) for stage in stages]
            
            # Create plot
            plt.figure(figsize=(12, 6))
            bars = plt.barh(stage_names, durations)
            
            # Add duration labels
            for bar, duration in zip(bars, durations):
                plt.text(duration + 0.1, bar.get_y() + bar.get_height()/2, f"{duration:.2f}s",
                         va='center', fontsize=10)
            
            plt.xlabel('Duration (seconds)')
            plt.ylabel('Pipeline Stage')
            plt.title('Pipeline Stage Durations')
            plt.grid(True, axis='x', alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(output_path)
            plt.close()
        
        except Exception as e:
            logger.error("Error generating stage durations plot: %s", str(e))
            import traceback
            logger.error(traceback.format_exc())

    def _plot_confusion_matrix(self, confusion_matrix, output_path):
        """
        Generate confusion matrix plot.
        
        Args:
            confusion_matrix (dict): Confusion matrix values
            output_path (Path): Output file path
        """
        try:
            # Extract values
            tn = confusion_matrix.get('true_negatives', 0)
            fp = confusion_matrix.get('false_positives', 0)
            fn = confusion_matrix.get('false_negatives', 0)
            tp = confusion_matrix.get('true_positives', 0)
            
            # Create matrix
            cm = np.array([[tn, fp], [fn, tp]])
            
            # Create plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Negative', 'Positive'],
                        yticklabels=['Negative', 'Positive'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            
            # Save plot
            plt.savefig(output_path)
            plt.close()
        
        except Exception as e:
            logger.error("Error generating confusion matrix plot: %s", str(e))
            import traceback
            logger.error(traceback.format_exc())

    def _plot_roc_curve(self, y_true, y_score, output_path):
        """
        Generate ROC curve plot.
        
        Args:
            y_true (numpy.ndarray): True labels
            y_score (numpy.ndarray): Predicted scores/probabilities
            output_path (Path): Output file path
        """
        try:
            # Ensure binary labels
            y_true_binary = (y_true > 0.5).astype(int)
            
            # Compute ROC curve
            fpr, tpr, thresholds = roc_curve(y_true_binary, y_score)
            roc_auc = auc(fpr, tpr)
            
            # Create plot
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(output_path)
            plt.close()
        
        except Exception as e:
            logger.error("Error generating ROC curve plot: %s", str(e))
            import traceback
            logger.error(traceback.format_exc())

    def _plot_feature_importance(self, feature_importance, output_path):
        """
        Generate feature importance plot.
        
        Args:
            feature_importance (dict): Feature importance values
            output_path (Path): Output file path
        """
        try:
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Limit to top 20 features
            top_features = sorted_features[:min(20, len(sorted_features))]
            feature_names = [item[0] for item in top_features]
            importance_values = [item[1] for item in top_features]
            
            # Create plot
            plt.figure(figsize=(12, 8))
            bars = plt.barh(feature_names, importance_values)
            
            # Add value labels
            for bar, value in zip(bars, importance_values):
                plt.text(value + 0.01, bar.get_y() + bar.get_height()/2, f"{value:.4f}",
                         va='center', fontsize=8)
            
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title('Feature Importance')
            plt.grid(True, axis='x', alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(output_path)
            plt.close()
        
        except Exception as e:
            logger.error("Error generating feature importance plot: %s", str(e))
            import traceback
            logger.error(traceback.format_exc())

    def _plot_correlation_heatmap(self, df, output_path):
        """
        Generate correlation heatmap for features.
        
        Args:
            df (pandas.DataFrame): DataFrame with features
            output_path (Path): Output file path
        """
        try:
            # Calculate correlation matrix
            corr = df.corr()
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr, dtype=bool))
            
            # Create plot
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=False, 
                       center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            
            # Save plot
            plt.savefig(output_path)
            plt.close()
        
        except Exception as e:
            logger.error("Error generating correlation heatmap: %s", str(e))
            import traceback
            logger.error(traceback.format_exc())

    def _plot_cluster_size_distribution(self, clusters, output_path):
        """
        Generate cluster size distribution plot.
        
        Args:
            clusters (list): Entity clusters
            output_path (Path): Output file path
        """
        try:
            # Calculate cluster sizes
            sizes = [len(cluster) for cluster in clusters]
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.hist(sizes, bins=30, log=True)
            plt.xlabel('Cluster Size')
            plt.ylabel('Number of Clusters (log scale)')
            plt.title('Cluster Size Distribution')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(output_path)
            plt.close()
        
        except Exception as e:
            logger.error("Error generating cluster size distribution plot: %s", str(e))
            import traceback
            logger.error(traceback.format_exc())

    def _plot_field_distribution(self, field_stats, output_path):
        """
        Generate field distribution plot.
        
        Args:
            field_stats (dict): Field statistics
            output_path (Path): Output file path
        """
        try:
            # Extract field counts
            fields = []
            total_occurrences = []
            unique_values = []
            
            for field, stats in field_stats.items():
                fields.append(field)
                total_occurrences.append(stats.get('total_occurrences', 0))
                unique_values.append(stats.get('unique_values', 0))
            
            # Create DataFrame for plotting
            df = pd.DataFrame({
                'Field': fields,
                'Total Occurrences': total_occurrences,
                'Unique Values': unique_values
            })
            
            # Sort by total occurrences
            df = df.sort_values('Total Occurrences', ascending=False)
            
            # Create plot
            plt.figure(figsize=(12, 6))
            
            x = np.arange(len(df))
            width = 0.35
            
            plt.bar(x - width/2, df['Total Occurrences'], width, label='Total Occurrences')
            plt.bar(x + width/2, df['Unique Values'], width, label='Unique Values')
            
            plt.xlabel('Field')
            plt.ylabel('Count')
            plt.title('Field Distribution')
            plt.xticks(x, df['Field'], rotation=45, ha='right')
            plt.legend()
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(output_path)
            plt.close()
        
        except Exception as e:
            logger.error("Error generating field distribution plot: %s", str(e))
            import traceback
            logger.error(traceback.format_exc())

    def _plot_performance_comparison(self, performance, output_path):
        """
        Generate performance comparison plot.
        
        Args:
            performance (dict): Performance metrics
            output_path (Path): Output file path
        """
        try:
            # Extract stage durations
            stages = list(performance.get('stage_durations', {}).keys())
            durations = list(performance.get('stage_durations', {}).values())
            
            # Extract throughput if available
            throughput = []
            for stage in stages:
                if stage in performance.get('throughput', {}):
                    throughput.append(performance['throughput'][stage])
                else:
                    throughput.append(0)
            
            # Create plot with two y-axes
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot durations on primary y-axis
            color = 'tab:blue'
            ax1.set_xlabel('Pipeline Stage')
            ax1.set_ylabel('Duration (seconds)', color=color)
            bars = ax1.bar(stages, durations, alpha=0.7, color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            
            # Add duration labels
            for bar, duration in zip(bars, durations):
                ax1.text(bar.get_x() + bar.get_width()/2, duration + 0.1, 
                         f"{duration:.1f}s", ha='center', va='bottom', fontsize=8, color=color)
            
            # Plot throughput on secondary y-axis if available
            if any(throughput):
                ax2 = ax1.twinx()
                color = 'tab:red'
                ax2.set_ylabel('Throughput (records/second)', color=color)
                ax2.plot(stages, throughput, 'o-', color=color)
                ax2.tick_params(axis='y', labelcolor=color)
            
            plt.title('Pipeline Performance by Stage')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(output_path)
            plt.close()
        
        except Exception as e:
            logger.error("Error generating performance comparison plot: %s", str(e))
            import traceback
            logger.error(traceback.format_exc())

    def _save_report_metadata(self, reports_dir):
        """
        Save report metadata.
        
        Args:
            reports_dir (Path): Reports directory
        """
        try:
            # Create metadata
            metadata = {
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'report_types': list(self.reports.keys()),
                'pipeline_mode': self.config['system']['mode'],
                'successful_reports': sum(1 for report in self.reports.values() if report)
            }
            
            # Save metadata
            with open(reports_dir / "report_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
        
        except Exception as e:
            logger.error("Error saving report metadata: %s", str(e))
            import traceback
            logger.error(traceback.format_exc())

    def _verify_data_consistency(self):
        """
        Verify consistency of data used for reporting.
        
        Returns:
            bool: True if data is consistent, False otherwise
        """
        output_dir = Path(self.config['system']['output_dir'])
        
        # Check for required files
        required_files = [
            "pipeline_summary.json",
            "classification_metrics.json",
            "feature_importance.json",
            "clustering_metrics.json"
        ]
        
        missing_files = []
        for file_name in required_files:
            if not (output_dir / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            logger.warning(f"Missing recommended files for reporting: {', '.join(missing_files)}")
            logger.warning("Some reports may be incomplete or unavailable")
        
        # Check data consistency
        try:
            # Check feature vectors and labels consistency
            feature_vectors_path = output_dir / "feature_vectors.npy"
            labels_path = output_dir / "labels.npy"
            
            if feature_vectors_path.exists() and labels_path.exists():
                feature_vectors = np.load(feature_vectors_path)
                labels = np.load(labels_path)
                
                if len(feature_vectors) != len(labels):
                    logger.warning(f"Feature vectors ({len(feature_vectors)}) and labels ({len(labels)}) have different lengths")
                    logger.warning("Will attempt to align arrays during report generation")
            
            # Check predictions consistency
            predictions_path = output_dir / "predictions.npy"
            if predictions_path.exists() and labels_path.exists():
                predictions = np.load(predictions_path)
                labels = np.load(labels_path)
                
                if len(predictions) != len(labels):
                    logger.warning(f"Predictions ({len(predictions)}) and labels ({len(labels)}) have different lengths")
                    logger.warning("Will attempt to align arrays during report generation")
            
            # Check timestamp alignment
            pipeline_summary_path = output_dir / "pipeline_summary.json"
            if pipeline_summary_path.exists():
                with open(pipeline_summary_path, 'r') as f:
                    pipeline_summary = json.load(f)
                
                if 'timestamp' in pipeline_summary:
                    pipeline_time = pipeline_summary['timestamp']
                    current_time = time.time()
                    
                    # Check if report is being generated more than 24 hours after pipeline execution
                    if current_time - pipeline_time > 86400:  # 24 hours in seconds
                        logger.warning(f"Report is being generated {(current_time - pipeline_time)/3600:.1f} hours after pipeline execution")
            
            return True
        
        except Exception as e:
            logger.error(f"Error verifying data consistency: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
