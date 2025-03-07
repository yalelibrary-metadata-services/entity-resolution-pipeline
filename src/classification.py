"""
Classification module for entity resolution.

This module provides the Classifier class, which handles training and evaluation
of the logistic regression classifier for entity resolution.
"""

import os
import logging
import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import save_checkpoint, load_checkpoint, Timer

logger = logging.getLogger(__name__)

class Classifier:
    """
    Handles training and evaluation of the logistic regression classifier.
    
    Features:
    - Logistic regression with gradient descent
    - Hyperparameter tuning
    - Feature importance analysis
    - Recursive feature elimination
    - Evaluation metrics (precision, recall, F1-score)
    """
    
    def __init__(self, config):
        """
        Initialize the classifier with configuration parameters.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        
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
        self.metrics = {}
        self.selected_features = None
        
        logger.info("Classifier initialized with algorithm: %s", self.algorithm)

    def execute(self, checkpoint=None):
        """
        Execute classifier training and evaluation.
        
        Args:
            checkpoint (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Classification results
        """
        # Load checkpoint if provided
        if checkpoint and os.path.exists(checkpoint):
            state = load_checkpoint(checkpoint)
            self.weights = state.get('weights', None)
            self.metrics = state.get('metrics', {})
            self.selected_features = state.get('selected_features', None)
            logger.info("Resumed classification from checkpoint: %s", checkpoint)
        
        # Load feature vectors and labels
        self.feature_vectors, self.labels, self.feature_names = self._load_features()
        
        logger.info("Loaded %d feature vectors with %d features",
                   len(self.feature_vectors), len(self.feature_names))
        
        # Split data into training and testing sets
        train_ratio = self.config['data']['train_test_split']
        X_train, X_test, y_train, y_test = train_test_split(
            self.feature_vectors, self.labels,
            test_size=1 - train_ratio,
            random_state=self.config['system']['random_seed'],
            stratify=self.labels
        )
        
        logger.info("Split data: %d training, %d testing", len(X_train), len(X_test))
        
        # Normalize features
        X_train_norm, X_test_norm = self._normalize_features(X_train, X_test)
        
        # Apply recursive feature elimination if enabled
        if self.config['features']['rfe_enabled'] and self.selected_features is None:
            logger.info("Performing recursive feature elimination")
            
            self.selected_features = self._perform_rfe(
                X_train_norm, y_train,
                step_size=self.config['features']['rfe_step_size'],
                cv_folds=self.config['features']['rfe_cv_folds']
            )
            
            logger.info("Selected %d features", len(self.selected_features))
            
            # Filter features
            X_train_norm = X_train_norm[:, self.selected_features]
            X_test_norm = X_test_norm[:, self.selected_features]
            
            # Update feature names
            self.feature_names = [self.feature_names[i] for i in self.selected_features]
        
        # Train model
        with Timer() as timer:
            if self.weights is None:
                logger.info("Training classifier")
                
                # Initialize model
                self.model = self._initialize_model()
                
                # Train model
                self.model.fit(X_train_norm, y_train)
                
                # Get weights
                self.weights = self.model.coef_[0]
                
                logger.info("Classifier trained in %.2f seconds", timer.duration)
            else:
                logger.info("Using pre-trained weights")
        
        # Evaluate model
        logger.info("Evaluating classifier")
        
        # Make predictions
        y_pred_proba = self.model.predict_proba(X_test_norm)[:, 1]
        y_pred = (y_pred_proba >= self.decision_threshold).astype(int)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary'
        )
        
        confusion = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = confusion.ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except Exception:
            roc_auc = 0.0
        
        # Store metrics
        self.metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }
        
        logger.info("Evaluation metrics: precision=%.4f, recall=%.4f, f1=%.4f",
                   precision, recall, f1)
        
        # Analyze feature importance
        feature_importance = self._analyze_feature_importance()
        
        # Save results
        self._save_results()
        
        # Return results
        results = {
            'vectors_processed': len(self.feature_vectors),
            'features_used': len(self.feature_names),
            'training_examples': len(X_train),
            'testing_examples': len(X_test),
            'metrics': self.metrics,
            'training_duration': timer.duration
        }
        
        logger.info("Classification completed with %d feature vectors, %d features",
                   len(self.feature_vectors), len(self.feature_names))
        
        return results

    # Update src/classification.py to handle both storage types
    def _load_features(self):
        """
        Load feature vectors and labels with enhanced error handling.
        """
        try:
            output_dir = Path(self.config['system']['output_dir'])
            mmap_dir = Path(self.config['system']['temp_dir']) / "mmap"
            
            # Try multiple approaches to find feature vectors
            feature_vectors = None
            labels = None
            
            # Approach 1: Standard numpy files in output dir
            standard_feature_path = output_dir / "feature_vectors.npy"
            standard_labels_path = output_dir / "labels.npy"
            
            if standard_feature_path.exists() and standard_labels_path.exists():
                logger.info(f"Loading feature vectors from standard path: {standard_feature_path}")
                feature_vectors = np.load(standard_feature_path)
                labels = np.load(standard_labels_path)
            
            # Approach 2: Check feature vectors info for memory-mapped files
            info_path = output_dir / "feature_vectors_info.json"
            if feature_vectors is None and info_path.exists():
                logger.info(f"Found feature vectors info at: {info_path}")
                with open(info_path, 'r') as f:
                    info = json.load(f)
                
                feature_vectors_file = info.get('feature_vectors_file')
                labels_file = info.get('labels_file')
                
                if feature_vectors_file and Path(feature_vectors_file).exists():
                    logger.info(f"Loading feature vectors from memory-mapped file: {feature_vectors_file}")
                    feature_vectors = np.load(feature_vectors_file)
                
                if labels_file and Path(labels_file).exists():
                    logger.info(f"Loading labels from memory-mapped file: {labels_file}")
                    labels = np.load(labels_file)
            
            # Approach 3: Check memory-mapped files directly
            mmap_feature_file = mmap_dir / "feature_vectors.mmap"
            mmap_labels_file = mmap_dir / "labels.mmap"
            
            if feature_vectors is None and mmap_feature_file.exists():
                logger.info(f"Loading feature vectors from memory-mapped file: {mmap_feature_file}")
                try:
                    # Try to load as numpy memory-mapped array
                    feature_vectors = np.memmap(mmap_feature_file, mode='r')
                    # Try to reshape if necessary
                    if feature_vectors.ndim == 1:
                        # Try to guess dimensions based on feature names
                        feature_names_path = output_dir / "feature_names.json"
                        if feature_names_path.exists():
                            with open(feature_names_path, 'r') as f:
                                feature_names = json.load(f)
                            num_features = len(feature_names)
                            # Reshape assuming row-major order
                            total_elements = feature_vectors.size
                            num_vectors = total_elements // num_features
                            if num_vectors * num_features == total_elements:
                                feature_vectors = feature_vectors.reshape(num_vectors, num_features)
                except Exception as e:
                    logger.error(f"Error loading memory-mapped feature vectors: {e}")
                    feature_vectors = None
            
            # Load feature names
            feature_names_path = output_dir / "feature_names.json"
            if feature_names_path.exists():
                with open(feature_names_path, 'r') as f:
                    feature_names = json.load(f)
                logger.info(f"Loaded {len(feature_names)} feature names")
            else:
                logger.warning(f"Feature names file not found: {feature_names_path}")
                # Create default feature names if needed and feature vectors exist
                if feature_vectors is not None and feature_vectors.ndim > 1:
                    feature_names = [f"feature_{i}" for i in range(feature_vectors.shape[1])]
                else:
                    feature_names = []
            
            # Ensure we have arrays
            if feature_vectors is None:
                logger.warning("Could not find feature vectors in any location")
                feature_vectors = np.array([])
            
            if labels is None:
                logger.warning("Could not find labels in any location")
                labels = np.array([])
            
            # Log shapes
            if isinstance(feature_vectors, np.ndarray):
                logger.info(f"Loaded feature vectors with shape: {feature_vectors.shape}")
            
            if isinstance(labels, np.ndarray):
                logger.info(f"Loaded labels with shape: {labels.shape}")
            
            return feature_vectors, labels, feature_names
        
        except Exception as e:
            logger.error(f"Error loading features: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return np.array([]), np.array([]), []

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
        
        # Normalize training and testing data
        X_train_norm = (X_train - means) / stds
        X_test_norm = (X_test - means) / stds
        
        return X_train_norm, X_test_norm

    def _initialize_model(self):
        """
        Initialize classification model.
        
        Returns:
            sklearn.linear_model.LogisticRegression: Logistic regression model
        """
        if self.algorithm == 'logistic_regression':
            model = LogisticRegression(
                penalty=self.regularization,
                C=1.0 / self.regularization_strength,
                solver='liblinear',  # Changed from 'saga' to 'liblinear' for better compatibility
                max_iter=self.max_iterations,
                tol=self.convergence_tolerance,
                class_weight=self.class_weight if self.class_weight != 'None' else None,
                random_state=self.config['system']['random_seed']
            )
            
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
        # Initialize model for RFE
        base_model = LogisticRegression(
            penalty=self.regularization,
            C=1.0 / self.regularization_strength,
            solver='liblinear',
            max_iter=self.max_iterations,
            tol=self.convergence_tolerance,
            class_weight=self.class_weight if self.class_weight != 'None' else None,
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
            verbose=0
        )
        
        # Fit RFE
        rfe.fit(X, y)
        
        # Get selected feature indices
        selected_features = np.where(rfe.support_)[0]
        
        return selected_features.tolist()

    def _analyze_feature_importance(self):
        """
        Analyze feature importance.
        
        Returns:
            dict: Feature importance information
        """
        if self.weights is None or not self.feature_names:
            return {}
        
        # Create dictionary of feature importance
        importance = {}
        
        for i, feature_name in enumerate(self.feature_names):
            weight = self.weights[i] if i < len(self.weights) else 0.0
            importance[feature_name] = abs(weight)
        
        # Sort by importance
        sorted_importance = dict(sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True))
        
        return sorted_importance

    def _save_results(self):
        """
        Save classification results.
        """
        output_dir = Path(self.config['system']['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        # Save model weights
        np.save(output_dir / "model_weights.npy", self.weights)
        
        # Save metrics
        with open(output_dir / "classification_metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save feature importance
        feature_importance = self._analyze_feature_importance()
        with open(output_dir / "feature_importance.json", 'w') as f:
            json.dump(feature_importance, f, indent=2)
        
        # Save selected features
        if self.selected_features is not None:
            with open(output_dir / "selected_features.json", 'w') as f:
                json.dump({
                    'indices': self.selected_features,
                    'names': [self.feature_names[i] for i in range(len(self.feature_names))]
                }, f, indent=2)
        
        # Save final checkpoint
        checkpoint_path = Path(self.config['system']['checkpoint_dir']) / "classification_final.ckpt"
        save_checkpoint({
            'weights': self.weights.tolist() if self.weights is not None else None,
            'metrics': self.metrics,
            'selected_features': self.selected_features
        }, checkpoint_path)
        
        # Generate feature importance plot
        self._plot_feature_importance(feature_importance, output_dir / "feature_importance.png")
        
        # Generate confusion matrix plot
        self._plot_confusion_matrix(output_dir / "confusion_matrix.png")
        
        logger.info("Classification results saved to %s", output_dir)

    def _plot_feature_importance(self, feature_importance, output_path):
        """
        Generate feature importance plot.
        
        Args:
            feature_importance (dict): Feature importance values
            output_path (Path): Output file path
        """
        try:
            # Sort by importance
            features = list(feature_importance.keys())
            importance = list(feature_importance.values())
            
            # Limit to top 20 features
            if len(features) > 20:
                features = features[:20]
                importance = importance[:20]
            
            # Create plot
            plt.figure(figsize=(12, 8))
            plt.barh(features, importance)
            plt.xlabel('Absolute Weight')
            plt.ylabel('Feature')
            plt.title('Feature Importance')
            plt.tight_layout()
            
            # Save plot
            plt.savefig(output_path)
            plt.close()
        
        except Exception as e:
            logger.error("Error generating feature importance plot: %s", str(e))

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

    def predict(self, feature_vector):
        """
        Make prediction for a feature vector.
        
        Args:
            feature_vector (list or numpy.ndarray): Feature vector
            
        Returns:
            tuple: (prediction, probability)
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Ensure numpy array
        if not isinstance(feature_vector, np.ndarray):
            feature_vector = np.array(feature_vector)
        
        # Reshape if needed
        if len(feature_vector.shape) == 1:
            feature_vector = feature_vector.reshape(1, -1)
        
        # Make prediction
        probability = self.model.predict_proba(feature_vector)[0, 1]
        prediction = 1 if probability >= self.decision_threshold else 0
        
        return prediction, probability

    def batch_predict(self, feature_vectors):
        """
        Make predictions for multiple feature vectors.
        
        Args:
            feature_vectors (list or numpy.ndarray): Feature vectors
            
        Returns:
            tuple: (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Ensure numpy array
        if not isinstance(feature_vectors, np.ndarray):
            feature_vectors = np.array(feature_vectors)
        
        # Make predictions
        probabilities = self.model.predict_proba(feature_vectors)[:, 1]
        predictions = (probabilities >= self.decision_threshold).astype(int)
        
        return predictions, probabilities
