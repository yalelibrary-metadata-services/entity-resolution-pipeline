"""
Redesigned classification module for entity resolution.

This module provides the Classifier class for training and applying a
classification model to determine entity matches, with improved data handling
and state management.
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix, 
    roc_curve, auc, precision_recall_curve
)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from src.utils import Timer
from src.data_manager import DataManager

logger = logging.getLogger(__name__)

class Classifier:
    """
    Handles training and evaluation of classification models for entity resolution
    with improved data management and error handling.
    
    Features:
    - Standardized data loading using DataManager
    - Consistent training and evaluation flow
    - Better error handling and recovery
    - Comprehensive metrics and visualization
    """
    
    def __init__(self, config):
        """
        Initialize the classifier with configuration parameters.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        
        # Initialize data manager
        self.data_manager = DataManager(config)
        
        # Classification parameters
        self.algorithm = config['classification']['algorithm']
        self.regularization = config['classification']['regularization']
        self.regularization_strength = config['classification']['regularization_strength']
        self.learning_rate = config['classification']['learning_rate']
        self.max_iterations = config['classification']['max_iterations']
        self.convergence_tolerance = config['classification']['convergence_tolerance']
        self.batch_size = config['classification']['batch_size']
        self.class_weight = config['classification']['class_weight']
        self.decision_threshold = config['classification']['decision_threshold']
        
        # Initialize model and data
        self.model = None
        self.feature_vectors = None
        self.labels = None
        self.feature_names = None
        self.weights = None
        self.selected_features = None
        
        # Train-test split data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Normalization parameters
        self.feature_means = None
        self.feature_stds = None
        
        # Results
        self.metrics = {}
        self.visualization_paths = {}
        
        self.output_dir = Path(self.config['system']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Classifier initialized with algorithm: %s", self.algorithm)
    
    def execute(self, checkpoint=None):
        """
        Execute classifier training and evaluation.
        
        Args:
            checkpoint (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Classification results
        """
        logger.info("Starting classification process")
        
        # Check if classification results already exist
        if self.data_manager.exists('classification_index'):
            logger.info("Loading existing classification results")
            return self._load_classification_results()
        
        # Load feature data
        self.feature_vectors, self.labels, self.feature_names = self.data_manager.load_feature_data()
        
        # Ensure data was loaded successfully
        if self.feature_vectors is None or self.labels is None or self.feature_names is None:
            logger.error("Failed to load feature data")
            return {
                'error': 'Failed to load feature data',
                'status': 'failed'
            }
        
        logger.info(f"Loaded {len(self.feature_vectors)} feature vectors with {len(self.feature_names)} features")
        logger.info(f"Labels distribution: {np.sum(self.labels == 1)} positive, {np.sum(self.labels == 0)} negative")
        
        # Validate data before proceeding
        if not self._validate_input_data():
            logger.error("Invalid input data, cannot proceed with classification")
            return {'error': 'Invalid input data', 'status': 'failed'}
        
        # Split data into training and testing sets with stratification
        train_ratio = self.config['data']['train_test_split']
        X_train, X_test, y_train, y_test = train_test_split(
            self.feature_vectors, self.labels,
            test_size=1 - train_ratio,
            random_state=self.config['system']['random_seed'],
            stratify=self.labels
        )
        
        # Store for later use
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # Save test indices for reporting
        test_indices = np.arange(len(self.labels))[len(X_train):]
        self.data_manager.save_numpy_array('test_indices', test_indices)
        
        # Normalize features
        X_train_norm, X_test_norm = self._normalize_features(X_train, X_test)
        
        # Save normalization parameters
        self.data_manager.save_numpy_array('feature_means', self.feature_means)
        self.data_manager.save_numpy_array('feature_stds', self.feature_stds)
        
        # Apply recursive feature elimination if enabled
        if self.config['features']['rfe_enabled']:
            logger.info("Performing recursive feature elimination")
            
            self.selected_features = self._perform_rfe(
                X_train_norm, y_train,
                step_size=self.config['features']['rfe_step_size'],
                cv_folds=self.config['features']['rfe_cv_folds']
            )
            
            logger.info(f"Selected {len(self.selected_features)} features")
            
            # Filter features
            X_train_norm = X_train_norm[:, self.selected_features]
            X_test_norm = X_test_norm[:, self.selected_features]
            
            # Update feature names
            selected_feature_names = [self.feature_names[i] for i in self.selected_features]
            
            # Save selected features
            self.data_manager.save('selected_features', {
                'indices': self.selected_features,
                'names': selected_feature_names
            })
            
            # Update feature names to selected subset
            self.feature_names = selected_feature_names
        else:
            # If not using RFE, all features are selected
            self.selected_features = list(range(len(self.feature_names)))
        
        # Train model with proper timing and error handling
        with Timer() as timer:
            try:
                logger.info("Training classifier")
                
                # Initialize model
                self.model = self._initialize_model()
                
                # Train model
                self.model.fit(X_train_norm, y_train)
                
                # Get weights
                self.weights = self.model.coef_[0]
                
                # Save trained model
                self.data_manager.save('classifier_model', self.model)
                
                logger.info(f"Classifier trained in {timer.duration:.2f} seconds")
            
            except Exception as e:
                logger.error(f"Error during model training: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                
                # Use fallback approach
                logger.warning("Using simple majority class model as fallback")
                majority_class = int(np.mean(y_train) > 0.5)
                dummy_weights = np.zeros(X_train_norm.shape[1])
                
                # Create dummy model
                self.model = LogisticRegression()
                self.model.classes_ = np.array([0, 1])
                self.model.coef_ = np.array([dummy_weights])
                self.model.intercept_ = np.array([0.0 if majority_class == 0 else 1.0])
                
                # Set weights
                self.weights = dummy_weights
                
                # Save fallback model
                self.data_manager.save('classifier_model', self.model)
        
        # Generate predictions and evaluate model
        logger.info("Evaluating classifier")
        
        try:
            # Generate predictions
            y_pred_proba = self.model.predict_proba(X_test_norm)[:, 1]
            y_pred = (y_pred_proba >= self.decision_threshold).astype(int)
            
            # Save testing data
            self.data_manager.save_numpy_array('predictions', y_pred)
            self.data_manager.save_numpy_array('probabilities', y_pred_proba)
            
            # Calculate metrics
            self.metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            
            logger.info(f"Evaluation metrics: precision={self.metrics['precision']:.4f}, "
                      f"recall={self.metrics['recall']:.4f}, "
                      f"f1={self.metrics['f1']:.4f}, "
                      f"accuracy={self.metrics['accuracy']:.4f}")
        
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Set default metrics
            self.metrics = {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'accuracy': 0.0,
                'roc_auc': 0.0,
                'true_positives': 0,
                'false_positives': 0,
                'true_negatives': 0,
                'false_negatives': 0
            }
        
        # Generate feature importance analysis
        importance_analysis = self._analyze_feature_importance()
        
        # Save feature importance
        self.data_manager.save('feature_importance', importance_analysis)
        
        # Generate and save visualizations
        self._generate_visualizations()
        
        # Save classification results
        classification_results = self.data_manager.save_classification_results(
            self.model,
            self.weights,
            self.metrics,
            y_pred if 'y_pred' in locals() else None,
            y_pred_proba if 'y_pred_proba' in locals() else None
        )
        
        # Apply classifier to full dataset if needed
        if self.config['classification'].get('classify_full_dataset', False):
            logger.info("Classifying full dataset")
            self._classify_full_dataset()
        
        # Return results
        results = {
            'vectors_processed': len(self.feature_vectors),
            'features_used': len(self.feature_names),
            'training_examples': len(X_train),
            'testing_examples': len(X_test),
            'metrics': self.metrics,
            'training_duration': timer.duration,
            'visualization_paths': self.visualization_paths,
            'feature_importance': importance_analysis.get('importance', {})
        }
        
        logger.info(f"Classification completed with {len(self.feature_vectors)} vectors, {len(self.feature_names)} features")
        
        return results
    
    def _load_classification_results(self):
        """
        Load existing classification results.
        
        Returns:
            dict: Classification results
        """
        # Load results from data manager
        results = self.data_manager.load_classification_results()
        
        if not results:
            logger.warning("Could not load classification results")
            return {'error': 'Could not load classification results', 'status': 'failed'}
        
        # Set instance variables
        self.model = results.get('model')
        self.weights = results.get('weights')
        self.metrics = results.get('metrics', {})
        
        # Load feature data to get feature count
        _, _, self.feature_names = self.data_manager.load_feature_data()
        
        # Load feature importance
        importance_analysis = self.data_manager.load('feature_importance') or {}
        
        # Return results in the expected format
        return {
            'vectors_processed': self.metrics.get('total_samples', 0),
            'features_used': len(self.feature_names) if self.feature_names else 0,
            'training_examples': self.metrics.get('training_examples', 0),
            'testing_examples': self.metrics.get('testing_examples', 0),
            'metrics': self.metrics,
            'feature_importance': importance_analysis.get('importance', {})
        }
    
    def _validate_input_data(self):
        """
        Validate input data to ensure it's usable for classification.
        
        Returns:
            bool: True if data is valid, False otherwise
        """
        # Check if feature vectors and labels are loaded
        if self.feature_vectors is None or self.labels is None:
            logger.error("Feature vectors or labels not loaded")
            return False
        
        # Check if feature vectors and labels have compatible shapes
        if len(self.feature_vectors) != len(self.labels):
            logger.error(f"Shape mismatch: {self.feature_vectors.shape} vs {self.labels.shape}")
            
            # Try to fix by truncating
            min_len = min(len(self.feature_vectors), len(self.labels))
            self.feature_vectors = self.feature_vectors[:min_len]
            self.labels = self.labels[:min_len]
            
            logger.info(f"Truncated feature vectors and labels to length {min_len}")
        
        # Check if we have at least some examples
        if len(self.feature_vectors) == 0:
            logger.error("Empty feature vectors")
            return False
        
        # Check for at least two classes
        unique_labels = np.unique(self.labels)
        if len(unique_labels) < 2:
            logger.error(f"Only one class found in labels: {unique_labels}")
            
            # Attempt to fix single-class issue by synthetic sampling
            self._balance_classes()
            
            # Recheck after balancing
            unique_labels = np.unique(self.labels)
            if len(unique_labels) < 2:
                logger.error("Still only one class after balancing attempt")
                return False
        
        # Check for NaN values
        if np.isnan(self.feature_vectors).any():
            logger.error("NaN values found in feature vectors")
            
            # Fix NaN values
            logger.info("Fixing NaN values")
            self.feature_vectors = np.nan_to_num(self.feature_vectors, nan=0.0)
        
        # Check for inf values
        if np.isinf(self.feature_vectors).any():
            logger.error("Inf values found in feature vectors")
            
            # Fix inf values
            logger.info("Fixing inf values")
            self.feature_vectors = np.nan_to_num(self.feature_vectors, posinf=1e10, neginf=-1e10)
        
        # Check for feature names
        if not self.feature_names:
            logger.warning("No feature names provided, generating default names")
            self.feature_names = [f"feature_{i}" for i in range(self.feature_vectors.shape[1])]
        
        # Check if feature names match feature vector dimensions
        if len(self.feature_names) != self.feature_vectors.shape[1]:
            logger.warning(f"Feature names count ({len(self.feature_names)}) doesn't match feature vector dimensions ({self.feature_vectors.shape[1]})")
            
            # Adjust feature names to match
            if len(self.feature_names) > self.feature_vectors.shape[1]:
                logger.info("Truncating feature names to match feature vectors")
                self.feature_names = self.feature_names[:self.feature_vectors.shape[1]]
            else:
                logger.info("Extending feature names to match feature vectors")
                self.feature_names.extend([f"feature_{i}" for i in range(len(self.feature_names), self.feature_vectors.shape[1])])
        
        return True
    
    def _balance_classes(self):
        """
        Balance classes by synthetic sampling when needed.
        """
        from sklearn.utils import resample
        
        # Count class instances
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        label_counts = dict(zip(unique_labels, counts))
        
        # If only one class, create synthetic samples for the other class
        if len(unique_labels) == 1:
            existing_class = unique_labels[0]
            new_class = 1 if existing_class == 0 else 0
            
            # Create synthetic samples for the missing class
            synthetic_size = min(len(self.feature_vectors), 100)  # Limit synthetic samples
            
            # Use existing samples as base with noise
            indices = np.random.choice(len(self.feature_vectors), synthetic_size, replace=True)
            synthetic_features = self.feature_vectors[indices] + np.random.normal(0, 0.1, (synthetic_size, self.feature_vectors.shape[1]))
            synthetic_labels = np.full(synthetic_size, new_class)
            
            # Combine original and synthetic data
            self.feature_vectors = np.vstack([self.feature_vectors, synthetic_features])
            self.labels = np.hstack([self.labels, synthetic_labels])
            
            logger.info(f"Added {synthetic_size} synthetic samples for class {new_class}")
        
        # If imbalanced, balance by upsampling minority class
        elif max(counts) / min(counts) > 10:  # If imbalance ratio > 10
            minority_class = unique_labels[np.argmin(counts)]
            majority_class = unique_labels[np.argmax(counts)]
            
            # Separate by class
            minority_indices = np.where(self.labels == minority_class)[0]
            majority_indices = np.where(self.labels == majority_class)[0]
            
            X_minority = self.feature_vectors[minority_indices]
            y_minority = self.labels[minority_indices]
            
            X_majority = self.feature_vectors[majority_indices]
            y_majority = self.labels[majority_indices]
            
            # Upsample minority class
            X_minority_upsampled, y_minority_upsampled = resample(
                X_minority, y_minority,
                replace=True,
                n_samples=len(X_majority),
                random_state=self.config['system']['random_seed']
            )
            
            # Combine upsampled minority class with majority class
            self.feature_vectors = np.vstack([X_majority, X_minority_upsampled])
            self.labels = np.hstack([y_majority, y_minority_upsampled])
            
            logger.info(f"Balanced classes by upsampling minority class {minority_class}")
    
    def _normalize_features(self, X_train, X_test):
        """
        Normalize feature vectors.
        
        Args:
            X_train (numpy.ndarray): Training feature vectors
            X_test (numpy.ndarray): Testing feature vectors
            
        Returns:
            tuple: (normalized_X_train, normalized_X_test)
        """
        # Calculate mean and standard deviation from training data
        means = np.mean(X_train, axis=0)
        stds = np.std(X_train, axis=0)
        
        # Handle zero standard deviation
        stds[stds == 0] = 1.0
        
        # Save normalization parameters
        self.feature_means = means
        self.feature_stds = stds
        
        # Normalize training and testing data
        X_train_norm = (X_train - means) / stds
        X_test_norm = (X_test - means) / stds
        
        # Check for NaN/Inf values
        if np.isnan(X_train_norm).any() or np.isinf(X_train_norm).any():
            logger.warning("NaN/Inf values found in normalized training data, fixing")
            X_train_norm = np.nan_to_num(X_train_norm, nan=0.0, posinf=1e10, neginf=-1e10)
        
        if np.isnan(X_test_norm).any() or np.isinf(X_test_norm).any():
            logger.warning("NaN/Inf values found in normalized testing data, fixing")
            X_test_norm = np.nan_to_num(X_test_norm, nan=0.0, posinf=1e10, neginf=-1e10)
        
        return X_train_norm, X_test_norm
    
    def _initialize_model(self):
        """
        Initialize classification model.
        
        Returns:
            sklearn.linear_model.LogisticRegression: Logistic regression model
        """
        if self.algorithm == 'logistic_regression':
            # Determine optimal solver based on dataset size and regularization
            if self.regularization == 'l1':
                solver = 'liblinear'  # liblinear works well with L1
            elif self.feature_vectors.shape[0] > 10000 or self.feature_vectors.shape[1] > 100:
                solver = 'saga'  # saga is faster for large datasets
            else:
                solver = 'lbfgs'  # lbfgs is generally robust
            
            # Determine class weights
            if self.class_weight == 'balanced':
                class_weight = 'balanced'
            elif self.class_weight == 'auto':
                # Calculate class distribution
                unique_labels, counts = np.unique(self.labels, return_counts=True)
                if len(unique_labels) == 2 and max(counts) / min(counts) > 3:
                    class_weight = 'balanced'
                    logger.info("Using balanced class weights due to class imbalance")
                else:
                    class_weight = None
            else:
                class_weight = None
            
            model = LogisticRegression(
                penalty=self.regularization,
                C=1.0 / self.regularization_strength,
                solver=solver,
                max_iter=self.max_iterations,
                tol=self.convergence_tolerance,
                class_weight=class_weight,
                random_state=self.config['system']['random_seed'],
                n_jobs=-1  # Use all available cores
            )
            
            logger.info(f"Initialized logistic regression with solver={solver}, class_weight={class_weight}")
            return model
        
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def _perform_rfe(self, X, y, step_size=1, cv_folds=5):
        """
        Perform recursive feature elimination.
        
        Args:
            X (numpy.ndarray): Feature vectors
            y (numpy.ndarray): Labels
            step_size (int): Number of features to remove at each step
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            list: Indices of selected features
        """
        # Initialize model for RFE with appropriate solver
        solver = 'liblinear' if self.regularization == 'l1' else 'lbfgs'
        
        base_model = LogisticRegression(
            penalty=self.regularization,
            C=1.0 / self.regularization_strength,
            solver=solver,
            max_iter=self.max_iterations,
            tol=self.convergence_tolerance,
            class_weight='balanced' if self.class_weight in ['balanced', 'auto'] else None,
            random_state=self.config['system']['random_seed']
        )
        
        # Determine minimum number of features
        n_features = X.shape[1]
        min_features = max(5, int(n_features * 0.2))  # At least 5 features or 20% of original
        
        # Initialize RFE
        rfe = RFE(
            estimator=base_model,
            n_features_to_select=min_features,
            step=step_size,
            verbose=1
        )
        
        try:
            # Fit RFE
            rfe.fit(X, y)
            
            # Get selected feature indices
            selected_features = np.where(rfe.support_)[0]
            
            return selected_features.tolist()
        
        except Exception as e:
            logger.error(f"Error during RFE: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Fall back to using all features
            logger.warning("Falling back to using all features")
            return list(range(X.shape[1]))
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate classification metrics.
        
        Args:
            y_true (numpy.ndarray): True labels
            y_pred (numpy.ndarray): Predicted labels
            y_pred_proba (numpy.ndarray, optional): Predicted probabilities
            
        Returns:
            dict: Classification metrics
        """
        try:
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Handle different shapes of confusion matrix
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                # Handle unusual confusion matrix shapes
                logger.error(f"Unexpected confusion matrix shape: {cm.shape}")
                
                # Calculate values by comparing arrays directly
                tp = np.sum((y_true == 1) & (y_pred == 1))
                tn = np.sum((y_true == 0) & (y_pred == 0))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                fn = np.sum((y_true == 1) & (y_pred == 0))
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
            # Calculate ROC AUC if probabilities are provided
            roc_auc = 0.0
            if y_pred_proba is not None:
                try:
                    # Only calculate ROC AUC if there are both positive and negative examples
                    if len(np.unique(y_true)) > 1:
                        from sklearn.metrics import roc_auc_score
                        roc_auc = roc_auc_score(y_true, y_pred_proba)
                except Exception as e:
                    logger.error(f"Error calculating ROC AUC: {e}")
            
            # Save detailed metrics for evaluation
            prediction_details = pd.DataFrame({
                'true_label': y_true,
                'predicted_label': y_pred,
                'probability': y_pred_proba if y_pred_proba is not None else np.zeros_like(y_pred)
            })
            
            self.data_manager.save_dataframe('prediction_details', prediction_details)
            
            # Gather metrics
            metrics = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'accuracy': float(accuracy),
                'roc_auc': float(roc_auc),
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn),
                'total_samples': len(y_true),
                'positive_samples': int(np.sum(y_true == 1)),
                'negative_samples': int(np.sum(y_true == 0)),
                'training_examples': len(self.X_train) if hasattr(self, 'X_train') else 0,
                'testing_examples': len(self.X_test) if hasattr(self, 'X_test') else 0
            }
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Return default metrics
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'accuracy': 0.0,
                'roc_auc': 0.0,
                'true_positives': 0,
                'false_positives': 0,
                'true_negatives': 0,
                'false_negatives': 0,
                'total_samples': len(y_true),
                'positive_samples': int(np.sum(y_true == 1)),
                'negative_samples': int(np.sum(y_true == 0))
            }
    
    def _analyze_feature_importance(self):
        """
        Analyze feature importance.
        
        Returns:
            dict: Feature importance information
        """
        if self.weights is None or not self.feature_names:
            return {'error': 'No weights or feature names available'}
        
        try:
            # Create dictionary of feature importance
            importance = {}
            
            for i, feature_name in enumerate(self.feature_names):
                if i < len(self.weights):
                    weight = self.weights[i]
                    importance[feature_name] = float(abs(weight))
            
            # Sort by importance
            sorted_importance = dict(sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True))
            
            # Normalize importance values to sum to 1.0
            total_importance = sum(sorted_importance.values())
            if total_importance > 0:
                normalized_importance = {k: v / total_importance for k, v in sorted_importance.items()}
            else:
                normalized_importance = sorted_importance
            
            # Calculate additional statistics
            importance_stats = {
                'max_feature': max(sorted_importance.items(), key=lambda x: x[1])[0] if sorted_importance else None,
                'min_feature': min(sorted_importance.items(), key=lambda x: x[1])[0] if sorted_importance else None,
                'mean_importance': float(np.mean(list(sorted_importance.values()))) if sorted_importance else 0,
                'std_importance': float(np.std(list(sorted_importance.values()))) if sorted_importance else 0
            }
            
            return {
                'importance': sorted_importance,
                'normalized_importance': normalized_importance,
                'stats': importance_stats,
                'raw_weights': self.weights.tolist()
            }
        
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    def _classify_full_dataset(self):
        """
        Apply classifier to full dataset of candidate pairs.
        """
        try:
            # Load candidate pairs
            candidate_pairs = self.data_manager.load('candidate_pairs')
            
            if not candidate_pairs:
                logger.warning("No candidate pairs found, skipping full dataset classification")
                return
            
            logger.info(f"Classifying {len(candidate_pairs)} candidate pairs")
            
            # Process candidate pairs in batches
            batch_size = self.batch_size
            classified_pairs = []
            
            for i in range(0, len(candidate_pairs), batch_size):
                batch = candidate_pairs[i:i+batch_size]
                
                # Load feature engineer to construct feature vectors
                from src.feature_engineer import FeatureEngineer
                feature_engineer = FeatureEngineer(self.config)
                
                # Build feature vectors for batch
                batch_vectors = []
                batch_ids = []
                
                for pair in batch:
                    try:
                        # Get record IDs
                        record1_id = pair.get('record1_id')
                        record2_id = pair.get('record2_id')
                        
                        if not record1_id or not record2_id:
                            continue
                        
                        # Get field hashes for records
                        record_field_hashes = feature_engineer.data_manager.load('record_field_hashes')
                        unique_strings = feature_engineer.data_manager.load('unique_strings')
                        
                        if not record_field_hashes or not unique_strings:
                            logger.error("Could not load required data for feature construction")
                            continue
                        
                        record1_fields = record_field_hashes.get(record1_id, {})
                        record2_fields = record_field_hashes.get(record2_id, {})
                        
                        # Skip if missing essential fields
                        if not record1_fields or not record2_fields:
                            continue
                        
                        # Construct feature vector
                        feature_vector = feature_engineer._construct_feature_vector(
                            record1_id, record2_id,
                            record1_fields, record2_fields,
                            unique_strings
                        )
                        
                        if feature_vector:
                            batch_vectors.append(feature_vector)
                            batch_ids.append((record1_id, record2_id))
                    except Exception as e:
                        logger.error(f"Error constructing feature vector for pair: {e}")
                
                # Normalize feature vectors
                if batch_vectors:
                    batch_vectors = np.array(batch_vectors)
                    
                    # Apply the same normalization as during training
                    batch_vectors_norm = (batch_vectors - self.feature_means) / self.feature_stds
                    
                    # Apply feature selection if needed
                    if self.selected_features:
                        batch_vectors_norm = batch_vectors_norm[:, self.selected_features]
                    
                    # Make predictions
                    batch_proba = self.model.predict_proba(batch_vectors_norm)[:, 1]
                    batch_pred = (batch_proba >= self.decision_threshold).astype(int)
                    
                    # Store classified pairs
                    for j in range(len(batch_ids)):
                        record1_id, record2_id = batch_ids[j]
                        classified_pairs.append({
                            'record1_id': record1_id,
                            'record2_id': record2_id,
                            'prediction': int(batch_pred[j]),
                            'confidence': float(batch_proba[j])
                        })
            
            # Save classified pairs
            self.data_manager.save('classified_pairs', classified_pairs)
            
            logger.info(f"Classified {len(classified_pairs)} candidate pairs")
        
        except Exception as e:
            logger.error(f"Error classifying full dataset: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _generate_visualizations(self):
        """
        Generate visualizations for classification results.
        """
        try:
            # Create visualizations directory
            viz_dir = self.output_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True, parents=True)
            
            # 1. Generate feature importance plot
            importance_path = viz_dir / "feature_importance.png"
            self._plot_feature_importance(importance_path)
            self.visualization_paths['feature_importance'] = str(importance_path)
            
            # 2. Generate confusion matrix plot
            confusion_path = viz_dir / "confusion_matrix.png"
            self._plot_confusion_matrix(confusion_path)
            self.visualization_paths['confusion_matrix'] = str(confusion_path)
            
            # 3. Generate ROC curve if probabilities are available
            if hasattr(self, 'X_test_norm') and hasattr(self, 'y_test') and self.model is not None:
                try:
                    y_pred_proba = self.model.predict_proba(self.X_test_norm)[:, 1]
                    roc_path = viz_dir / "roc_curve.png"
                    self._plot_roc_curve(self.y_test, y_pred_proba, roc_path)
                    self.visualization_paths['roc_curve'] = str(roc_path)
                except Exception as e:
                    logger.error(f"Error generating ROC curve: {e}")
            
            # 4. Generate precision-recall curve if probabilities are available
            if hasattr(self, 'X_test_norm') and hasattr(self, 'y_test') and self.model is not None:
                try:
                    y_pred_proba = self.model.predict_proba(self.X_test_norm)[:, 1]
                    pr_path = viz_dir / "precision_recall_curve.png"
                    self._plot_precision_recall_curve(self.y_test, y_pred_proba, pr_path)
                    self.visualization_paths['precision_recall_curve'] = str(pr_path)
                except Exception as e:
                    logger.error(f"Error generating precision-recall curve: {e}")
            
            # 5. Generate feature correlation heatmap if feature vectors are available
            if hasattr(self, 'feature_vectors') and self.feature_vectors.size > 0:
                try:
                    corr_path = viz_dir / "feature_correlation.png"
                    self._plot_feature_correlation(corr_path)
                    self.visualization_paths['feature_correlation'] = str(corr_path)
                except Exception as e:
                    logger.error(f"Error generating feature correlation heatmap: {e}")
            
            # Save visualization paths
            self.data_manager.save('visualization_paths', self.visualization_paths)
            
            logger.info(f"Generated {len(self.visualization_paths)} visualizations")
        
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _plot_feature_importance(self, output_path):
        """
        Generate feature importance plot.
        
        Args:
            output_path (Path): Output file path
        """
        try:
            # Get feature importance
            feature_importance = self._analyze_feature_importance()
            if not feature_importance or 'importance' not in feature_importance:
                logger.warning("No feature importance data available")
                return
            
            # Sort by importance
            sorted_importance = sorted(feature_importance['importance'].items(), key=lambda x: x[1], reverse=True)
            
            # Limit to top features for readability
            top_n = min(20, len(sorted_importance))
            top_features = sorted_importance[:top_n]
            
            feature_names = [item[0] for item in top_features]
            importance_values = [item[1] for item in top_features]
            
            # Create plot
            plt.figure(figsize=(12, 8))
            bars = plt.barh(feature_names, importance_values, color='steelblue')
            
            # Add value labels
            for bar, value in zip(bars, importance_values):
                plt.text(value + 0.01, bar.get_y() + bar.get_height()/2, f"{value:.4f}",
                       va='center', fontsize=8)
            
            plt.xlabel('Absolute Weight')
            plt.ylabel('Feature')
            plt.title('Feature Importance')
            plt.grid(True, axis='x', alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Saved feature importance plot to {output_path}")
        
        except Exception as e:
            logger.error(f"Error generating feature importance plot: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _plot_confusion_matrix(self, output_path):
        """
        Generate confusion matrix plot.
        
        Args:
            output_path (Path): Output file path
        """
        try:
            # Extract confusion matrix values
            tn = self.metrics.get('true_negatives', 0)
            fp = self.metrics.get('false_positives', 0)
            fn = self.metrics.get('false_negatives', 0)
            tp = self.metrics.get('true_positives', 0)
            
            # Create confusion matrix
            cm = np.array([[tn, fp], [fn, tp]])
            
            # Calculate metrics for display
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Create plot
            plt.figure(figsize=(10, 8))
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                      xticklabels=['Negative', 'Positive'],
                      yticklabels=['Negative', 'Positive'])
            
            # Add metrics as text
            plt.text(0.5, -0.1, f"Accuracy: {accuracy:.4f}", ha='center', transform=plt.gca().transAxes)
            plt.text(0.5, -0.15, f"Precision: {precision:.4f}", ha='center', transform=plt.gca().transAxes)
            plt.text(0.5, -0.2, f"Recall: {recall:.4f}", ha='center', transform=plt.gca().transAxes)
            plt.text(0.5, -0.25, f"F1 Score: {f1:.4f}", ha='center', transform=plt.gca().transAxes)
            
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            
            # Save plot
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0.5)
            plt.close()
            
            logger.info(f"Saved confusion matrix plot to {output_path}")
        
        except Exception as e:
            logger.error(f"Error generating confusion matrix plot: {e}")
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
            plt.figure(figsize=(10, 8))
            
            # Plot ROC curve
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
            
            # Plot random guessing line
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            
            # Plot ideal point
            plt.plot([0, 0, 1], [0, 1, 1], color='green', lw=2, linestyle=':', label='Ideal point')
            
            # Add current threshold marker
            idx = np.argmin(np.abs(thresholds - self.decision_threshold))
            if idx < len(fpr):
                plt.plot(fpr[idx], tpr[idx], 'ro', markersize=8, label=f'Threshold = {self.decision_threshold:.2f}')
            
            # Add grid and labels
            plt.grid(True, alpha=0.3)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            
            # Save plot
            plt.savefig(output_path)
            plt.close()
            
            # Save ROC data
            roc_data = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist(),
                'auc': float(roc_auc)
            }
            
            self.data_manager.save('roc_data', roc_data)
            
            logger.info(f"Saved ROC curve plot to {output_path}")
        
        except Exception as e:
            logger.error(f"Error generating ROC curve plot: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _plot_precision_recall_curve(self, y_true, y_score, output_path):
        """
        Generate precision-recall curve plot.
        
        Args:
            y_true (numpy.ndarray): True labels
            y_score (numpy.ndarray): Predicted scores/probabilities
            output_path (Path): Output file path
        """
        try:
            # Compute precision-recall curve
            precision, recall, thresholds = precision_recall_curve(y_true, y_score)
            pr_auc = auc(recall, precision)
            
            # Compute average precision
            average_precision = np.mean(precision)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            
            # Plot precision-recall curve
            plt.plot(recall, precision, color='blue', lw=2, 
                   label=f'PR curve (area = {pr_auc:.3f}, avg precision = {average_precision:.3f})')
            
            # Plot baseline
            no_skill = np.sum(y_true) / len(y_true)
            plt.plot([0, 1], [no_skill, no_skill], color='red', lw=2, linestyle='--', label=f'Baseline ({no_skill:.3f})')
            
            # Add current threshold marker if thresholds array is non-empty
            if len(thresholds) > 0:
                # We need to handle the fact that precision_recall_curve returns one more point than thresholds
                thresholds = np.append(thresholds, 0)  # Add a final threshold for the last point
                idx = np.argmin(np.abs(thresholds - self.decision_threshold))
                if idx < len(precision):
                    plt.plot(recall[idx], precision[idx], 'go', markersize=8, label=f'Threshold = {self.decision_threshold:.2f}')
            
            # Add grid and labels
            plt.grid(True, alpha=0.3)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="best")
            
            # Save plot
            plt.savefig(output_path)
            plt.close()
            
            # Save PR data
            pr_data = {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': thresholds.tolist() if len(thresholds) > 0 else [],
                'pr_auc': float(pr_auc),
                'average_precision': float(average_precision)
            }
            
            self.data_manager.save('pr_data', pr_data)
            
            logger.info(f"Saved precision-recall curve plot to {output_path}")
        
        except Exception as e:
            logger.error(f"Error generating precision-recall curve plot: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _plot_feature_correlation(self, output_path):
        """
        Generate feature correlation heatmap.
        
        Args:
            output_path (Path): Output file path
        """
        try:
            # Create feature DataFrame
            feature_df = pd.DataFrame(self.feature_vectors, columns=self.feature_names)
            
            # Check if we have too many features for a readable heatmap
            if len(self.feature_names) > 30:
                # Get top 30 features by importance
                feature_importance = self._analyze_feature_importance()
                if feature_importance and 'importance' in feature_importance:
                    top_features = sorted(feature_importance['importance'].items(), key=lambda x: x[1], reverse=True)[:30]
                    top_feature_names = [item[0] for item in top_features]
                    feature_df = feature_df[top_feature_names]
            
            # Compute correlation matrix
            corr = feature_df.corr()
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr, dtype=bool))
            
            # Create plot
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', 
                       center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5})
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            
            # Save plot
            plt.savefig(output_path)
            plt.close()
            
            # Save correlation matrix
            self.data_manager.save('feature_correlation', corr.to_dict())
            
            logger.info(f"Saved feature correlation plot to {output_path}")
        
        except Exception as e:
            logger.error(f"Error generating feature correlation plot: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def predict(self, feature_vector):
        """
        Make prediction for a single feature vector.
        
        Args:
            feature_vector (list or numpy.ndarray): Feature vector
            
        Returns:
            tuple: (prediction, probability)
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        try:
            # Ensure numpy array
            if not isinstance(feature_vector, np.ndarray):
                feature_vector = np.array(feature_vector)
            
            # Reshape if needed
            if len(feature_vector.shape) == 1:
                feature_vector = feature_vector.reshape(1, -1)
            
            # Normalize feature vector
            if hasattr(self, 'feature_means') and hasattr(self, 'feature_stds'):
                feature_vector = (feature_vector - self.feature_means) / self.feature_stds
            
            # Apply feature selection if needed
            if self.selected_features:
                feature_vector = feature_vector[:, self.selected_features]
            
            # Make prediction
            probability = self.model.predict_proba(feature_vector)[0, 1]
            prediction = 1 if probability >= self.decision_threshold else 0
            
            return prediction, probability
        
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0, 0.0
