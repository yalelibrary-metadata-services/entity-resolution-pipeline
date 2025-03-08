"""
Improved classification module for entity resolution.

This module provides the Classifier class, which handles training and evaluation
of the logistic regression classifier for entity resolution with improved robustness.
"""

import os
import logging
import json
import numpy as np
import pickle
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
    - Robust data loading and error recovery
    - Detailed diagnostics for data issues
    - Improved serialization
    - Comprehensive metrics and visualization
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
        
        # Paths for data
        self.output_dir = Path(self.config['system']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directories
        self.model_dir = self.output_dir / "models"
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        self.checkpoint_dir = Path(self.config['system']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info("Classifier initialized with algorithm: %s", self.algorithm)
        
    def execute(self, checkpoint=None):
        """
        Execute classifier training and evaluation with improved robustness.
        
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
        
        # Ensure we have valid data before proceeding
        if not self._validate_input_data():
            logger.error("Invalid input data, cannot proceed with classification")
            return {'error': 'Invalid input data'}
        
        logger.info("Loaded %d feature vectors with %d features",
                len(self.feature_vectors), len(self.feature_names))
        
        # Split data into training and testing sets with explicit stratification
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
        
        # Save test indices for reporting correctly
        test_indices = np.arange(len(self.labels))[len(X_train):]
        np.save(self.output_dir / "test_indices.npy", test_indices)
        logger.info(f"Saved {len(test_indices)} test indices")
        
        # Normalize features
        X_train_norm, X_test_norm = self._normalize_features(X_train, X_test)
        
        # Save normalization parameters for future use
        self.feature_means = np.mean(X_train, axis=0)
        self.feature_stds = np.std(X_train, axis=0)
        self.feature_stds[self.feature_stds == 0] = 1.0  # Prevent division by zero
        
        # Save normalization parameters
        np.save(self.output_dir / "feature_means.npy", self.feature_means)
        np.save(self.output_dir / "feature_stds.npy", self.feature_stds)
        
        # Store normalized test data for later use
        self.X_test_norm = X_test_norm
        
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
            
            # Save selected features
            with open(self.output_dir / "selected_features.json", 'w') as f:
                json.dump({
                    'indices': self.selected_features,
                    'names': self.feature_names
                }, f, indent=2)
        
        # Train model
        with Timer() as timer:
            try:
                if self.weights is None:
                    logger.info("Training classifier")
                    
                    # Initialize model
                    self.model = self._initialize_model()
                    
                    # Train model
                    self.model.fit(X_train_norm, y_train)
                    
                    # Get weights
                    self.weights = self.model.coef_[0]
                    
                    # Save the model
                    with open(self.model_dir / "classifier_model.pkl", 'wb') as f:
                        pickle.dump(self.model, f)
                    
                    logger.info("Classifier trained in %.2f seconds", timer.duration)
                else:
                    logger.info("Using pre-trained weights")
                    
                    # Reconstruct model from weights
                    if self.model is None:
                        self.model = self._initialize_model()
                        self.model.coef_ = np.array([self.weights])
                        self.model.intercept_ = np.array([0.0])  # Default intercept
                        
                        # Save the reconstructed model
                        with open(self.model_dir / "classifier_model.pkl", 'wb') as f:
                            pickle.dump(self.model, f)
            except Exception as e:
                logger.error(f"Error during model training: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                
                if self.weights is None:
                    logger.warning("Using default weights due to training error")
                    self.weights = np.zeros(len(self.feature_names))
        
        # Evaluate model
        logger.info("Evaluating classifier")
        
        # Generate predictions with explicit verification
        try:
            y_pred_proba = self.model.predict_proba(X_test_norm)[:, 1]
            logger.info(f"Probability range: min={np.min(y_pred_proba):.4f}, max={np.max(y_pred_proba):.4f}")
            
            y_pred = (y_pred_proba >= self.decision_threshold).astype(int)
            
            # Save test labels and predictions
            np.save(self.output_dir / "test_labels.npy", y_test)
            np.save(self.output_dir / "predictions.npy", y_pred)
            np.save(self.output_dir / "probabilities.npy", y_pred_proba)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            self.metrics = metrics
            
            logger.info("Evaluation metrics: precision=%.4f, recall=%.4f, f1=%.4f, accuracy=%.4f",
                    metrics['precision'], metrics['recall'], metrics['f1'], metrics['accuracy'])
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
        
        # Save all classification results in a format that's easier to load later
        self._save_results()
        
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
            'model_path': str(self.model_dir / "classifier_model.pkl"),
            'feature_importance_path': str(self.output_dir / "feature_importance.json")
        }
        
        logger.info("Classification completed with %d feature vectors, %d features",
                len(self.feature_vectors), len(self.feature_names))
        
        return results

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
            return False
        
        # Check if we have at least some examples
        if len(self.feature_vectors) == 0:
            logger.error("Empty feature vectors")
            return False
        
        # Check for at least two classes
        unique_labels = np.unique(self.labels)
        if len(unique_labels) < 2:
            logger.error(f"Only one class found in labels: {unique_labels}")
            
            # Attempt to fix single-class issue by balancing
            if self.config['classification'].get('auto_balance_classes', False):
                logger.info("Attempting to balance classes by synthetic sampling")
                self._balance_classes()
                
                # Recheck after balancing
                unique_labels = np.unique(self.labels)
                if len(unique_labels) < 2:
                    return False
            else:
                return False
        
        # Check for NaN values
        if np.isnan(self.feature_vectors).any():
            logger.error("NaN values found in feature vectors")
            
            # Attempt to fix NaN values
            logger.info("Attempting to fix NaN values")
            self.feature_vectors = np.nan_to_num(self.feature_vectors, nan=0.0)
        
        # Check for inf values
        if np.isinf(self.feature_vectors).any():
            logger.error("Inf values found in feature vectors")
            
            # Attempt to fix inf values
            logger.info("Attempting to fix inf values")
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
        Balance classes by synthetic sampling.
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

    def _load_features(self):
        """
        Load feature vectors and labels with enhanced robustness.
        
        Returns:
            tuple: (feature_vectors, labels, feature_names)
        """
        try:
            output_dir = Path(self.config['system']['output_dir'])
            
            # Try multiple file formats and locations
            feature_vectors = None
            labels = None
            feature_names = None
            
            # Step 1: Load feature vectors - try multiple file formats
            for file_name in ["feature_vectors.npy", "feature_vectors.npz", "feature_vectors.pkl"]:
                file_path = output_dir / file_name
                if file_path.exists():
                    try:
                        if file_name.endswith('.npy'):
                            feature_vectors = np.load(file_path)
                        elif file_name.endswith('.npz'):
                            npz_file = np.load(file_path)
                            feature_vectors = npz_file['arr_0'] if 'arr_0' in npz_file else None
                        elif file_name.endswith('.pkl'):
                            with open(file_path, 'rb') as f:
                                feature_vectors = pickle.load(f)
                        
                        logger.info(f"Loaded feature vectors from {file_path}")
                        break
                    except Exception as e:
                        logger.warning(f"Error loading feature vectors from {file_path}: {e}")
            
            # Step 2: Load labels - try multiple file formats
            for file_name in ["labels.npy", "labels.npz", "labels.pkl"]:
                file_path = output_dir / file_name
                if file_path.exists():
                    try:
                        if file_name.endswith('.npy'):
                            labels = np.load(file_path)
                        elif file_name.endswith('.npz'):
                            npz_file = np.load(file_path)
                            labels = npz_file['arr_0'] if 'arr_0' in npz_file else None
                        elif file_name.endswith('.pkl'):
                            with open(file_path, 'rb') as f:
                                labels = pickle.load(f)
                        
                        logger.info(f"Loaded labels from {file_path}")
                        break
                    except Exception as e:
                        logger.warning(f"Error loading labels from {file_path}: {e}")
            
            # Step 3: Load feature names
            feature_names_path = output_dir / "feature_names.json"
            if feature_names_path.exists():
                try:
                    with open(feature_names_path, 'r') as f:
                        feature_names = json.load(f)
                    logger.info(f"Loaded {len(feature_names)} feature names")
                except Exception as e:
                    logger.warning(f"Error loading feature names: {e}")
            
            # Check if we have loaded all the necessary data
            if feature_vectors is None:
                logger.error("Failed to load feature vectors")
                feature_vectors = np.array([])
            
            if labels is None:
                logger.error("Failed to load labels")
                labels = np.array([])
            
            if feature_names is None:
                logger.warning("Failed to load feature names, using default names")
                feature_names = [f"feature_{i}" for i in range(feature_vectors.shape[1])] if feature_vectors.size > 0 else []
            
            # Ensure arrays have the right shape
            if feature_vectors.ndim == 1 and feature_vectors.size > 0:
                # Try to reshape based on feature names
                if feature_names:
                    n_features = len(feature_names)
                    if feature_vectors.size % n_features == 0:
                        n_samples = feature_vectors.size // n_features
                        feature_vectors = feature_vectors.reshape(n_samples, n_features)
                        logger.info(f"Reshaped feature vectors to {feature_vectors.shape}")
            
            # Ensure labels have the right shape
            if labels.ndim > 1 and labels.shape[1] == 1:
                labels = labels.ravel()
                logger.info(f"Flattened labels to {labels.shape}")
            
            # Log shapes for debugging
            logger.info(f"Feature vectors shape: {feature_vectors.shape if hasattr(feature_vectors, 'shape') else 'unknown'}")
            logger.info(f"Labels shape: {labels.shape if hasattr(labels, 'shape') else 'unknown'}")
            logger.info(f"Feature names count: {len(feature_names)}")
            
            # Ensure feature vectors and labels have compatible lengths
            if feature_vectors.size > 0 and labels.size > 0 and len(feature_vectors) != len(labels):
                logger.warning(f"Feature vectors and labels have different lengths: {len(feature_vectors)} vs {len(labels)}")
                
                # Truncate to shorter length
                min_len = min(len(feature_vectors), len(labels))
                feature_vectors = feature_vectors[:min_len]
                labels = labels[:min_len]
                logger.info(f"Truncated to {min_len} samples")
            
            return feature_vectors, labels, feature_names
        
        except Exception as e:
            logger.error(f"Unexpected error loading features: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return np.array([]), np.array([]), []

    def _normalize_features(self, X_train, X_test):
        """
        Normalize feature vectors with improved robustness.
        
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
        Initialize classification model with robust configuration.
        
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
        Perform recursive feature elimination with improved robustness.
        
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
            
            # Save RFE results
            rfe_results = {
                'support': rfe.support_.tolist(),
                'ranking': rfe.ranking_.tolist(),
                'selected_features': selected_features.tolist(),
                'n_features': len(selected_features)
            }
            
            with open(self.output_dir / "rfe_results.json", 'w') as f:
                json.dump(rfe_results, f, indent=2)
            
            return selected_features.tolist()
        
        except Exception as e:
            logger.error(f"Error during RFE: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Fall back to selecting all features
            logger.warning("Falling back to using all features")
            return list(range(X.shape[1]))

    def _calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate classification metrics with improved robustness.
        
        Args:
            y_true (numpy.ndarray): True labels
            y_pred (numpy.ndarray): Predicted labels
            y_pred_proba (numpy.ndarray, optional): Predicted probabilities. Defaults to None.
            
        Returns:
            dict: Classification metrics
        """
        try:
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Handle different shapes of confusion matrix
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            elif cm.shape == (1, 1):
                # Only one class in both true and predicted
                if y_true[0] == 1:  # All positive
                    if y_pred[0] == 1:  # All predicted positive
                        tp, fp, fn, tn = len(y_true), 0, 0, 0
                    else:  # All predicted negative
                        tp, fp, fn, tn = 0, 0, len(y_true), 0
                else:  # All negative
                    if y_pred[0] == 0:  # All predicted negative
                        tp, fp, fn, tn = 0, 0, 0, len(y_true)
                    else:  # All predicted positive
                        tp, fp, fn, tn = 0, len(y_true), 0, 0
            else:
                # Handle unusual confusion matrix shapes
                logger.error(f"Unexpected confusion matrix shape: {cm.shape}")
                tn, fp, fn, tp = 0, 0, 0, 0
                
                # Try to extract values if possible
                if cm.size >= 4:
                    flat_cm = cm.flatten()
                    if len(flat_cm) >= 4:
                        tn, fp, fn, tp = flat_cm[:4]
            
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
                        roc_auc = roc_auc_score(y_true, y_pred_proba)
                except Exception as e:
                    logger.error(f"Error calculating ROC AUC: {e}")
            
            # Save detailed metrics for evaluation
            with open(self.output_dir / "prediction_details.csv", 'w') as f:
                f.write("true_label,predicted_label,probability\n")
                for i in range(len(y_true)):
                    prob_val = y_pred_proba[i] if y_pred_proba is not None else ''
                    f.write(f"{y_true[i]},{y_pred[i]},{prob_val}\n")
            
            return {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'accuracy': float(accuracy),
                'roc_auc': float(roc_auc),
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn)
            }
        
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
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

    def _analyze_feature_importance(self):
        """
        Analyze feature importance with improved robustness.
        
        Returns:
            dict: Feature importance information
        """
        if self.weights is None or not self.feature_names:
            return {}
        
        try:
            # Create dictionary of feature importance
            importance = {}
            
            for i, feature_name in enumerate(self.feature_names):
                if i < len(self.weights):
                    weight = self.weights[i]
                    importance[feature_name] = abs(weight)
            
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
                'mean_importance': np.mean(list(sorted_importance.values())) if sorted_importance else 0,
                'std_importance': np.std(list(sorted_importance.values())) if sorted_importance else 0
            }
            
            return {
                'importance': sorted_importance,
                'normalized_importance': normalized_importance,
                'stats': importance_stats
            }
        
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def _classify_full_dataset(self):
        """
        Apply classifier to full dataset (all candidate pairs).
        """
        try:
            # Load candidate pairs
            output_dir = Path(self.config['system']['output_dir'])
            candidate_pairs_path = output_dir / "candidate_pairs.json"
            
            if not candidate_pairs_path.exists():
                logger.warning("Candidate pairs file not found, skipping full dataset classification")
                return
            
            logger.info("Loading candidate pairs for full classification")
            with open(candidate_pairs_path, 'r') as f:
                candidate_pairs = json.load(f)
            
            # Load feature engineering module
            from src.parallel_features import FeatureEngineer
            feature_engineer = FeatureEngineer(self.config)
            
            # Process candidate pairs in batches
            batch_size = self.config['classification']['batch_size']
            classified_pairs = []
            
            for i in range(0, len(candidate_pairs), batch_size):
                batch = candidate_pairs[i:i+batch_size]
                
                # Construct feature vectors for each pair
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
                        record1_fields = feature_engineer.record_field_hashes.get(record1_id, {})
                        record2_fields = feature_engineer.record_field_hashes.get(record2_id, {})
                        
                        # Skip if missing essential fields
                        if not record1_fields or not record2_fields:
                            continue
                        
                        # Construct feature vector
                        feature_vector = feature_engineer._construct_feature_vector(
                            record1_id, record2_id,
                            record1_fields, record2_fields,
                            feature_engineer.unique_strings, None,
                            self.feature_names
                        )
                        
                        if feature_vector:
                            batch_vectors.append(feature_vector)
                            batch_ids.append((record1_id, record2_id))
                    except Exception as e:
                        logger.error(f"Error constructing feature vector for pair: {e}")
                
                # Normalize feature vectors
                if batch_vectors:
                    batch_vectors = np.array(batch_vectors)
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
            with open(output_dir / "classified_pairs.json", 'w') as f:
                json.dump(classified_pairs, f, indent=2)
            
            logger.info(f"Classified {len(classified_pairs)} candidate pairs in full dataset")
        
        except Exception as e:
            logger.error(f"Error classifying full dataset: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _save_results(self):
        """
        Save classification results with improved organization.
        """
        try:
            # Save model weights
            np.save(self.output_dir / "model_weights.npy", self.weights)
            
            # Save metrics
            with open(self.output_dir / "classification_metrics.json", 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            # Save feature importance
            feature_importance = self._analyze_feature_importance()
            with open(self.output_dir / "feature_importance.json", 'w') as f:
                json.dump(feature_importance, f, indent=2)
            
            # Generate and save visualizations
            self._generate_visualizations()
            
            # Save metadata about the classification process
            metadata = {
                'algorithm': self.algorithm,
                'regularization': self.regularization,
                'regularization_strength': self.regularization_strength,
                'decision_threshold': self.decision_threshold,
                'feature_count': len(self.feature_names),
                'selected_feature_count': len(self.selected_features) if self.selected_features else len(self.feature_names),
                'sample_count': len(self.feature_vectors),
                'training_size': len(self.X_train) if hasattr(self, 'X_train') else 0,
                'testing_size': len(self.X_test) if hasattr(self, 'X_test') else 0,
                'class_distribution': {
                    'training': {
                        '0': int(np.sum(self.y_train == 0)) if hasattr(self, 'y_train') else 0,
                        '1': int(np.sum(self.y_train == 1)) if hasattr(self, 'y_train') else 0
                    },
                    'testing': {
                        '0': int(np.sum(self.y_test == 0)) if hasattr(self, 'y_test') else 0,
                        '1': int(np.sum(self.y_test == 1)) if hasattr(self, 'y_test') else 0
                    }
                }
            }
            
            with open(self.output_dir / "classification_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save information for downstream components
            component_info = {
                'metrics': self.metrics,
                'feature_importance': feature_importance.get('importance', {}),
                'model_path': str(self.model_dir / "classifier_model.pkl"),
                'weights_path': str(self.output_dir / "model_weights.npy"),
                'decision_threshold': self.decision_threshold
            }
            
            with open(self.output_dir / "classifier_info.json", 'w') as f:
                json.dump(component_info, f, indent=2)
            
            # Save final checkpoint
            checkpoint_path = self.checkpoint_dir / "classification_final.ckpt"
            save_checkpoint({
                'weights': self.weights.tolist() if self.weights is not None else None,
                'metrics': self.metrics,
                'selected_features': self.selected_features,
                'model_path': str(self.model_dir / "classifier_model.pkl")
            }, checkpoint_path)
            
            logger.info("Classification results saved to %s", self.output_dir)
        
        except Exception as e:
            logger.error(f"Error saving classification results: {e}")
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
            self._plot_feature_importance(viz_dir / "feature_importance.png")
            
            # 2. Generate confusion matrix plot
            self._plot_confusion_matrix(viz_dir / "confusion_matrix.png")
            
            # 3. Generate ROC curve if probabilities are available
            if hasattr(self, 'X_test_norm') and hasattr(self, 'y_test') and self.model is not None:
                try:
                    y_pred_proba = self.model.predict_proba(self.X_test_norm)[:, 1]
                    self._plot_roc_curve(self.y_test, y_pred_proba, viz_dir / "roc_curve.png")
                except Exception as e:
                    logger.error(f"Error generating ROC curve: {e}")
            
            # 4. Generate precision-recall curve if probabilities are available
            if hasattr(self, 'X_test_norm') and hasattr(self, 'y_test') and self.model is not None:
                try:
                    y_pred_proba = self.model.predict_proba(self.X_test_norm)[:, 1]
                    self._plot_precision_recall_curve(self.y_test, y_pred_proba, viz_dir / "precision_recall_curve.png")
                except Exception as e:
                    logger.error(f"Error generating precision-recall curve: {e}")
            
            # 5. Generate feature correlation heatmap if feature vectors are available
            if hasattr(self, 'feature_vectors') and self.feature_vectors.size > 0:
                try:
                    self._plot_feature_correlation(viz_dir / "feature_correlation.png")
                except Exception as e:
                    logger.error(f"Error generating feature correlation heatmap: {e}")
            
            logger.info("Generated classification visualizations in %s", viz_dir)
        
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
            
            # Save ROC data for future use
            roc_data = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist(),
                'auc': float(roc_auc)
            }
            
            with open(output_path.with_suffix('.json'), 'w') as f:
                json.dump(roc_data, f, indent=2)
        
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
            
            # Save PR data for future use
            pr_data = {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': thresholds.tolist() if len(thresholds) > 0 else [],
                'pr_auc': float(pr_auc),
                'average_precision': float(average_precision)
            }
            
            with open(output_path.with_suffix('.json'), 'w') as f:
                json.dump(pr_data, f, indent=2)
        
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
            
            # Save correlation matrix for future use
            with open(output_path.with_suffix('.json'), 'w') as f:
                # Convert to serializable format
                corr_dict = {col: {row: corr.loc[row, col] for row in corr.index} for col in corr.columns}
                json.dump(corr_dict, f)
        
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
        
        try:
            # Ensure numpy array
            if not isinstance(feature_vectors, np.ndarray):
                feature_vectors = np.array(feature_vectors)
            
            # Normalize feature vectors
            if hasattr(self, 'feature_means') and hasattr(self, 'feature_stds'):
                feature_vectors = (feature_vectors - self.feature_means) / self.feature_stds
            
            # Apply feature selection if needed
            if self.selected_features:
                feature_vectors = feature_vectors[:, self.selected_features]
            
            # Make predictions
            probabilities = self.model.predict_proba(feature_vectors)[:, 1]
            predictions = (probabilities >= self.decision_threshold).astype(int)
            
            return predictions, probabilities
        
        except Exception as e:
            logger.error(f"Error making batch predictions: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return np.array([]), np.array([])
