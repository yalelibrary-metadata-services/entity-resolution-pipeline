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
            logger.error("Data consistency check failed, aborting report generation")
            return {
                'error': 'Data consistency check failed',
                'reports_generated': 0,
                'duration': 0
            }
        
        with Timer() as timer:
            # Create reports directory
            reports_dir = Path(self.config['system']['output_dir']) / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate summary report
            self.reports['summary'] = self._generate_summary_report(reports_dir)
            
            # Generate classification report
            self.reports['classification'] = self._generate_classification_report(reports_dir)
            
            # Generate feature importance report
            self.reports['feature_importance'] = self._generate_feature_importance_report(reports_dir)
            
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
        
        return {
            'reports_generated': len(self.reports),
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
            
            return summary
        
        except Exception as e:
            logger.error("Error generating summary report: %s", str(e))
            return {}

    def _generate_classification_report(self, reports_dir):
        """
        Generate classification performance report.
        
        Args:
            reports_dir (Path): Reports directory
            
        Returns:
            dict: Classification report information
        """
        try:
            # Load classification metrics
            output_dir = Path(self.config['system']['output_dir'])
            metrics_path = output_dir / "classification_metrics.json"
            
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    classification_metrics = json.load(f)
            else:
                logger.warning("Classification metrics file not found")
                return {}
            
            # Create classification report
            report = {
                'metrics': classification_metrics,
                'confusion_matrix': {
                    'true_positives': classification_metrics.get('true_positives', 0),
                    'false_positives': classification_metrics.get('false_positives', 0),
                    'true_negatives': classification_metrics.get('true_negatives', 0),
                    'false_negatives': classification_metrics.get('false_negatives', 0)
                }
            }
            
            # Calculate derived metrics
            precision = classification_metrics.get('precision', 0)
            recall = classification_metrics.get('recall', 0)
            f1 = classification_metrics.get('f1', 0)
            accuracy = classification_metrics.get('accuracy', 0)
            
            # Write metrics to CSV
            metrics_path = reports_dir / "classification_metrics.csv"
            with open(metrics_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value'])
                writer.writerow(['Precision', f"{precision:.4f}"])
                writer.writerow(['Recall', f"{recall:.4f}"])
                writer.writerow(['F1 Score', f"{f1:.4f}"])
                writer.writerow(['Accuracy', f"{accuracy:.4f}"])
                writer.writerow(['True Positives', report['confusion_matrix']['true_positives']])
                writer.writerow(['False Positives', report['confusion_matrix']['false_positives']])
                writer.writerow(['True Negatives', report['confusion_matrix']['true_negatives']])
                writer.writerow(['False Negatives', report['confusion_matrix']['false_negatives']])
            
            # Generate confusion matrix plot
            self._plot_confusion_matrix(report['confusion_matrix'], reports_dir / "confusion_matrix.png")
            
            return report
        
        except Exception as e:
            logger.error("Error generating classification report: %s", str(e))
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
                
                for feature, importance in feature_importance.items():
                    writer.writerow([feature, f"{importance:.6f}"])
            
            return report
        
        except Exception as e:
            logger.error("Error generating feature importance report: %s", str(e))
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
            
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    clustering_metrics = json.load(f)
            else:
                logger.warning("Clustering metrics file not found")
                return {}
            
            # Create clustering report
            report = {
                'metrics': clustering_metrics
            }
            
            # Write cluster statistics to CSV
            stats_path = reports_dir / "cluster_statistics.csv"
            with open(stats_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Statistic', 'Value'])
                writer.writerow(['Total Clusters', clustering_metrics.get('cluster_count', 0)])
                writer.writerow(['Total Entities', clustering_metrics.get('total_entities', 0)])
                writer.writerow(['Singleton Clusters', clustering_metrics.get('singleton_clusters', 0)])
                writer.writerow(['Minimum Cluster Size', clustering_metrics.get('min_cluster_size', 0)])
                writer.writerow(['Maximum Cluster Size', clustering_metrics.get('max_cluster_size', 0)])
                writer.writerow(['Mean Cluster Size', f"{clustering_metrics.get('mean_cluster_size', 0):.2f}"])
                writer.writerow(['Median Cluster Size', clustering_metrics.get('median_cluster_size', 0)])
            
            # Write cluster size distribution to CSV
            if 'size_distribution' in clustering_metrics:
                dist_path = reports_dir / "cluster_size_distribution.csv"
                with open(dist_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Cluster Size', 'Count'])
                    
                    for size, count in sorted(clustering_metrics['size_distribution'].items(), key=lambda x: int(x[0])):
                        writer.writerow([size, count])
            
            return report
        
        except Exception as e:
            logger.error("Error generating clustering report: %s", str(e))
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
            
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    pipeline_summary = json.load(f)
            else:
                logger.warning("Pipeline summary file not found")
                return {}
            
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
            
            return report
        
        except Exception as e:
            logger.error("Error generating performance report: %s", str(e))
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
            
            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    field_stats = json.load(f)
            else:
                logger.warning("Field statistics file not found")
                return {}
            
            # Create data quality report
            report = {
                'field_statistics': field_stats
            }
            
            # Write field statistics to CSV
            stats_path = reports_dir / "field_statistics.csv"
            with open(stats_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Field', 'Total Occurrences', 'Unique Values'])
                
                for field, stats in field_stats.items():
                    writer.writerow([
                        field,
                        stats.get('total_occurrences', 0),
                        stats.get('unique_values', 0)
                    ])
            
            return report
        
        except Exception as e:
            logger.error("Error generating data quality report: %s", str(e))
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
                        </tr>
                    """)
                    
                    for field, stats in field_stats.items():
                        f.write(f"""
                        <tr>
                            <td>{field}</td>
                            <td>{stats.get('total_occurrences', 0)}</td>
                            <td>{stats.get('unique_values', 0)}</td>
                        </tr>
                        """)
                    
                    f.write("</table>")
                
                # Close HTML document
                f.write("""
                </body>
                </html>
                """)
            
            logger.info("Generated HTML report: %s", html_path)
        
        except Exception as e:
            logger.error("Error generating HTML report: %s", str(e))

    def _plot_stage_durations(self, stages, output_path):
        """
        Generate plot of pipeline stage durations.
        
        Args:
            stages (list): List of stage information
            output_path (Path): Output file path
        """
        try:
            # Extract stage names and durations
            stage_names = [stage['name'] for stage in stages]
            durations = [stage['duration'] for stage in stages]
            
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
                'pipeline_mode': self.config['system']['mode']
            }
            
            # Save metadata
            with open(reports_dir / "report_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
        
        except Exception as e:
            logger.error("Error saving report metadata: %s", str(e))

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
            logger.error(f"Missing required files for reporting: {', '.join(missing_files)}")
            return False
        
        # Check metric consistency
        try:
            # Load various metric files
            with open(output_dir / "pipeline_summary.json", 'r') as f:
                pipeline_summary = json.load(f)
            
            with open(output_dir / "classification_metrics.json", 'r') as f:
                classification_metrics = json.load(f)
            
            # Verify timestamp alignment
            if 'timestamp' in pipeline_summary:
                pipeline_time = pipeline_summary['timestamp']
                current_time = time.time()
                
                # Check if report is being generated more than 24 hours after pipeline execution
                if current_time - pipeline_time > 86400:  # 24 hours in seconds
                    logger.warning(f"Report is being generated {(current_time - pipeline_time)/3600:.1f} hours after pipeline execution")
            
            return True
        
        except Exception as e:
            logger.error(f"Error verifying data consistency: {e}")
            return False
