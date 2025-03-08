"""
Analysis module for entity resolution pipeline.

This module provides the Analyzer class, which handles analysis of pipeline
processes and results, identifying patterns, anomalies, and insights.
"""

import os
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import networkx as nx

from src.utils import Timer

logger = logging.getLogger(__name__)

class Analyzer:
    """
    Handles analysis of pipeline processes and results.
    
    Features:
    - Analysis of feature distribution and correlation
    - Evaluation of classification performance
    - Analysis of cluster properties and distribution
    - Identification of edge cases and anomalies
    - Visualization of entity relationships
    """
    
    def __init__(self, config):
        """
        Initialize the analyzer with configuration parameters.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        
        # Initialize analysis results
        self.analysis_results = {}
        
        logger.info("Analyzer initialized")

    def execute(self, checkpoint=None):
        """
        Execute analysis of pipeline processes and results.
        
        Args:
            checkpoint (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Analysis results
        """
        logger.info("Starting analysis")
        
        with Timer() as timer:
            # Load data
            feature_vectors, labels, feature_names = self._load_features()
            classification_metrics = self._load_classification_metrics()
            clustering_metrics = self._load_clustering_metrics()
            clusters = self._load_clusters()
            
            # Analyze feature distribution and correlation
            self.analysis_results['feature_analysis'] = self._analyze_features(
                feature_vectors, labels, feature_names
            )
            
            # Analyze classification performance
            self.analysis_results['classification_analysis'] = self._analyze_classification(
                classification_metrics
            )
            
            # Analyze clustering results
            self.analysis_results['clustering_analysis'] = self._analyze_clustering(
                clustering_metrics, clusters
            )
            
            # Find edge cases and anomalies
            self.analysis_results['anomalies'] = self._find_anomalies(
                feature_vectors, labels, feature_names, clusters
            )
            
            # Generate visualizations
            self._generate_visualizations(
                feature_vectors, labels, feature_names, clusters
            )
        
        logger.info("Analysis completed in %.2f seconds", timer.duration)
        
        # Save results
        self._save_results()
        
        return {
            'completed': True,
            'duration': timer.duration
        }

    def _load_features(self):
        """
        Load feature vectors and labels.
        
        Returns:
            tuple: (feature_vectors, labels, feature_names)
        """
        try:
            output_dir = Path(self.config['system']['output_dir'])
            
            # Load feature vectors
            feature_vectors_path = output_dir / "feature_vectors.npy"
            if feature_vectors_path.exists():
                feature_vectors = np.load(feature_vectors_path)
            else:
                logger.warning("Feature vectors file not found, using empty array")
                feature_vectors = np.array([])
            
            # Load labels
            labels_path = output_dir / "labels.npy"
            if labels_path.exists():
                labels = np.load(labels_path)
            else:
                logger.warning("Labels file not found, using empty array")
                labels = np.array([])
            
            # Load feature names
            feature_names_path = output_dir / "feature_names.json"
            if feature_names_path.exists():
                with open(feature_names_path, 'r') as f:
                    feature_names = json.load(f)
            else:
                logger.warning("Feature names file not found, using empty list")
                feature_names = []
            
            return feature_vectors, labels, feature_names
        
        except Exception as e:
            logger.error("Error loading features: %s", str(e))
            return np.array([]), np.array([]), []

    def _load_classification_metrics(self):
        """
        Load classification metrics.
        
        Returns:
            dict: Classification metrics
        """
        try:
            output_dir = Path(self.config['system']['output_dir'])
            metrics_path = output_dir / "classification_metrics.json"
            
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
            else:
                logger.warning("Classification metrics file not found, using empty dict")
                metrics = {}
            
            return metrics
        
        except Exception as e:
            logger.error("Error loading classification metrics: %s", str(e))
            return {}

    def _load_clustering_metrics(self):
        """
        Load clustering metrics.
        
        Returns:
            dict: Clustering metrics
        """
        try:
            output_dir = Path(self.config['system']['output_dir'])
            metrics_path = output_dir / "clustering_metrics.json"
            
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
            else:
                logger.warning("Clustering metrics file not found, using empty dict")
                metrics = {}
            
            return metrics
        
        except Exception as e:
            logger.error("Error loading clustering metrics: %s", str(e))
            return {}

    def _load_clusters(self):
        """
        Load entity clusters.
        
        Returns:
            list: Entity clusters
        """
        try:
            output_dir = Path(self.config['system']['output_dir'])
            clusters_path = output_dir / "entity_clusters.json"
            
            if clusters_path.exists():
                with open(clusters_path, 'r') as f:
                    clusters = json.load(f)
            else:
                logger.warning("Entity clusters file not found, using empty list")
                clusters = []
            
            return clusters
        
        except Exception as e:
            logger.error("Error loading entity clusters: %s", str(e))
            return []

    def _analyze_features(self, feature_vectors, labels, feature_names):
        """
        Analyze feature distribution and correlation with improved handling of constant features.
        """
        if len(feature_vectors) == 0 or len(feature_names) == 0:
            logger.warning("Cannot analyze features: Empty feature vectors or feature names")
            return {}
        
        # Create DataFrame for analysis
        df = pd.DataFrame(feature_vectors, columns=feature_names)
        if len(labels) == len(feature_vectors):
            df['label'] = labels
        
        # Check for constant features
        constant_features = []
        for feature in feature_names:
            if df[feature].nunique() <= 1:
                constant_features.append(feature)
                logger.warning(f"Feature '{feature}' has zero variance (constant values)")
        
        # Calculate feature statistics
        feature_stats = df.describe().to_dict()
        
        # Calculate feature importance with safe handling
        feature_importance = {}
        if 'label' in df.columns:
            for feature in feature_names:
                if feature in constant_features:
                    feature_importance[feature] = 0  # Zero importance for constant features
                else:
                    try:
                        correlation = df[feature].corr(df['label'])
                        feature_importance[feature] = abs(correlation if not pd.isna(correlation) else 0)
                    except Exception:
                        logger.warning(f"Could not calculate correlation for feature '{feature}'")
                        feature_importance[feature] = 0
        
        # Calculate correlation matrix safely
        correlation_matrix = {}
        non_constant_features = [f for f in feature_names if f not in constant_features]
        
        if non_constant_features:
            try:
                # Use only non-constant features for correlation
                corr_df = df[non_constant_features].corr().fillna(0)
                correlation_matrix = corr_df.to_dict()
                
                # Add constant features back with zero correlations
                for feature in constant_features:
                    correlation_matrix[feature] = {f: 0 for f in feature_names}
                    for f in feature_names:
                        if f in correlation_matrix:
                            correlation_matrix[f][feature] = 0
                        
            except Exception as e:
                logger.error(f"Error calculating correlation matrix: {str(e)}")
        
        # Find highly correlated features (from non-constant features only)
        highly_correlated = []
        for i, feature1 in enumerate(non_constant_features):
            for feature2 in non_constant_features[i+1:]:
                if feature1 in correlation_matrix and feature2 in correlation_matrix[feature1]:
                    correlation = correlation_matrix[feature1][feature2]
                    if abs(correlation) > 0.8:
                        highly_correlated.append({
                            'feature1': feature1,
                            'feature2': feature2,
                            'correlation': correlation
                        })
        
        return {
            'feature_stats': feature_stats,
            'feature_importance': feature_importance,
            'highly_correlated': highly_correlated,
            'constant_features': constant_features
        }

    def _analyze_classification(self, metrics):
        """
        Analyze classification performance.
        
        Args:
            metrics (dict): Classification metrics
            
        Returns:
            dict: Classification analysis results
        """
        if not metrics:
            return {}
        
        # Calculate derived metrics
        derived_metrics = {}
        
        # Precision-recall trade-off
        if 'precision' in metrics and 'recall' in metrics:
            derived_metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
        
        # False positive rate
        if 'false_positives' in metrics and 'true_negatives' in metrics:
            derived_metrics['false_positive_rate'] = metrics['false_positives'] / (metrics['false_positives'] + metrics['true_negatives']) if (metrics['false_positives'] + metrics['true_negatives']) > 0 else 0
        
        # False negative rate
        if 'false_negatives' in metrics and 'true_positives' in metrics:
            derived_metrics['false_negative_rate'] = metrics['false_negatives'] / (metrics['false_negatives'] + metrics['true_positives']) if (metrics['false_negatives'] + metrics['true_positives']) > 0 else 0
        
        return {
            'metrics': metrics,
            'derived_metrics': derived_metrics
        }

    def _analyze_clustering(self, metrics, clusters):
        """
        Analyze clustering results.
        
        Args:
            metrics (dict): Clustering metrics
            clusters (list): Entity clusters
            
        Returns:
            dict: Clustering analysis results
        """
        if not metrics or not clusters:
            return {}
        
        # Calculate cluster size statistics
        cluster_sizes = [len(cluster) for cluster in clusters]
        
        size_stats = {
            'min': min(cluster_sizes) if cluster_sizes else 0,
            'max': max(cluster_sizes) if cluster_sizes else 0,
            'mean': np.mean(cluster_sizes) if cluster_sizes else 0,
            'median': np.median(cluster_sizes) if cluster_sizes else 0,
            'std': np.std(cluster_sizes) if cluster_sizes else 0,
            'total': sum(cluster_sizes) if cluster_sizes else 0,
            'count': len(cluster_sizes)
        }
        
        # Calculate cluster size distribution
        size_counts = {}
        for size in cluster_sizes:
            size_counts[size] = size_counts.get(size, 0) + 1
        
        # Group sizes for better visualization
        grouped_sizes = {
            '1': sum(count for size, count in size_counts.items() if size == 1),
            '2-5': sum(count for size, count in size_counts.items() if 2 <= size <= 5),
            '6-10': sum(count for size, count in size_counts.items() if 6 <= size <= 10),
            '11-20': sum(count for size, count in size_counts.items() if 11 <= size <= 20),
            '21-50': sum(count for size, count in size_counts.items() if 21 <= size <= 50),
            '51+': sum(count for size, count in size_counts.items() if size > 50)
        }
        
        return {
            'metrics': metrics,
            'size_stats': size_stats,
            'size_distribution': size_counts,
            'grouped_sizes': grouped_sizes
        }

    def _find_anomalies(self, feature_vectors, labels, feature_names, clusters):
        """
        Find edge cases and anomalies in the results.
        
        Args:
            feature_vectors (numpy.ndarray): Feature vectors
            labels (numpy.ndarray): Labels
            feature_names (list): Feature names
            clusters (list): Entity clusters
            
        Returns:
            dict: Identified anomalies
        """
        anomalies = {}
        
        # Skip if data is missing
        if len(feature_vectors) == 0 or len(feature_names) == 0:
            return anomalies
        
        # Find anomalous features
        df = pd.DataFrame(feature_vectors, columns=feature_names)
        if len(labels) == len(feature_vectors):
            df['label'] = labels
        
        # Detect outliers using z-score
        z_scores = ((df[feature_names] - df[feature_names].mean()) / df[feature_names].std()).abs()
        outliers = (z_scores > 3).any(axis=1)
        
        if outliers.any():
            outlier_indices = np.where(outliers)[0]
            anomalies['feature_outliers'] = outlier_indices.tolist()
        
        # Find anomalous clusters
        large_clusters = [cluster for cluster in clusters if len(cluster) > 50]
        anomalies['large_clusters'] = large_clusters
        
        # Singleton clusters (could be valid but worth checking)
        singleton_clusters = [cluster for cluster in clusters if len(cluster) == 1]
        anomalies['singleton_clusters_count'] = len(singleton_clusters)
        
        return anomalies

    def _generate_visualizations(self, feature_vectors, labels, feature_names, clusters):
        """
        Generate visualizations for analysis results.
        
        Args:
            feature_vectors (numpy.ndarray): Feature vectors
            labels (numpy.ndarray): Labels
            feature_names (list): Feature names
            clusters (list): Entity clusters
        """
        output_dir = Path(self.config['system']['output_dir']) / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Skip if data is missing
        if len(feature_vectors) == 0 or len(feature_names) == 0:
            return
        
        # Create DataFrame for visualization
        df = pd.DataFrame(feature_vectors, columns=feature_names)
        if len(labels) == len(feature_vectors):
            df['label'] = labels
        
        # Generate feature correlation heatmap
        self._plot_correlation_heatmap(df[feature_names], output_dir / "feature_correlation.png")
        
        # Generate feature importance plot
        if 'label' in df.columns:
            self._plot_feature_importance(df, feature_names, output_dir / "feature_importance.png")
        
        # Generate feature distribution plots for top features
        if len(feature_names) > 0:
            for feature in feature_names[:min(5, len(feature_names))]:
                self._plot_feature_distribution(df, feature, output_dir / f"{feature}_distribution.png")
        
        # Generate cluster size distribution plot
        if clusters:
            self._plot_cluster_size_distribution(clusters, output_dir / "cluster_size_distribution.png")
        
        # Generate dimensionality reduction plot
        if len(feature_vectors) > 10 and len(feature_names) > 2:
            self._plot_dimensionality_reduction(feature_vectors, labels, output_dir / "feature_space.png")

    def _plot_correlation_heatmap(self, df, output_path):
        """
        Generate feature correlation heatmap.
        
        Args:
            df (pandas.DataFrame): DataFrame with features
            output_path (Path): Output file path
        """
        try:
            plt.figure(figsize=(12, 10))
            correlation = df.corr()
            mask = np.triu(np.ones_like(correlation, dtype=bool))
            sns.heatmap(correlation, mask=mask, annot=False, cmap='coolwarm', center=0, square=True)
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
        
        except Exception as e:
            logger.error("Error generating correlation heatmap: %s", str(e))

    def _plot_feature_importance(self, df, feature_names, output_path):
        """
        Generate feature importance plot.
        
        Args:
            df (pandas.DataFrame): DataFrame with features and labels
            feature_names (list): Feature names
            output_path (Path): Output file path
        """
        try:
            importance = {}
            for feature in feature_names:
                importance[feature] = abs(df[feature].corr(df['label']))
            
            # Sort by importance
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            
            # Select top 20 features
            top_features = list(importance.keys())[:min(20, len(importance))]
            top_values = [importance[feature] for feature in top_features]
            
            plt.figure(figsize=(12, 8))
            plt.barh(top_features, top_values)
            plt.xlabel('Absolute Correlation with Label')
            plt.ylabel('Feature')
            plt.title('Feature Importance (Correlation with Label)')
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
        
        except Exception as e:
            logger.error("Error generating feature importance plot: %s", str(e))

    def _plot_feature_distribution(self, df, feature, output_path):
        """
        Generate feature distribution plot.
        
        Args:
            df (pandas.DataFrame): DataFrame with features and optionally labels
            feature (str): Feature name
            output_path (Path): Output file path
        """
        try:
            plt.figure(figsize=(10, 6))
            
            if 'label' in df.columns:
                # Plot distribution by label
                sns.histplot(data=df, x=feature, hue='label', element='step', bins=30, common_norm=False, stat='density')
                plt.title(f'Distribution of {feature} by Label')
            else:
                # Plot overall distribution
                sns.histplot(data=df, x=feature, bins=30)
                plt.title(f'Distribution of {feature}')
            
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
        
        except Exception as e:
            logger.error("Error generating feature distribution plot: %s", str(e))

    def _plot_cluster_size_distribution(self, clusters, output_path):
        """
        Generate cluster size distribution plot.
        
        Args:
            clusters (list): Entity clusters
            output_path (Path): Output file path
        """
        try:
            cluster_sizes = [len(cluster) for cluster in clusters]
            
            plt.figure(figsize=(10, 6))
            plt.hist(cluster_sizes, bins=30)
            plt.xlabel('Cluster Size')
            plt.ylabel('Count')
            plt.title('Cluster Size Distribution')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
        
        except Exception as e:
            logger.error("Error generating cluster size distribution plot: %s", str(e))

    def _plot_dimensionality_reduction(self, feature_vectors, labels, output_path):
        """
        Generate dimensionality reduction plot.
        
        Args:
            feature_vectors (numpy.ndarray): Feature vectors
            labels (numpy.ndarray): Labels
            output_path (Path): Output file path
        """
        try:
            # Apply dimensionality reduction
            if len(feature_vectors) > 1000:
                # Use PCA for larger datasets
                pca = PCA(n_components=2, random_state=42)
                reduced_data = pca.fit_transform(feature_vectors)
            else:
                # Use t-SNE for smaller datasets
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(feature_vectors) // 5))
                reduced_data = tsne.fit_transform(feature_vectors)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            
            if len(labels) == len(feature_vectors):
                # Color by label
                scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
                plt.colorbar(scatter, label='Label')
            else:
                # No label information
                plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.7)
            
            plt.title('Feature Space Visualization')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
        
        except Exception as e:
            logger.error("Error generating dimensionality reduction plot: %s", str(e))

    def _save_results(self):
        """
        Save analysis results.
        """
        output_dir = Path(self.config['system']['output_dir'])
        
        # Save analysis results
        with open(output_dir / "analysis_results.json", 'w') as f:
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
            
            # Convert and save
            json.dump(convert_numpy(self.analysis_results), f, indent=2)
        
        logger.info("Analysis results saved to %s", output_dir / "analysis_results.json")
