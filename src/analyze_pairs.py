#!/usr/bin/env python3
"""
Direct binary classification analysis with explicit manual calculations.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def analyze_classification_results(output_dir="./output"):
    output_dir = Path(output_dir)
    
    try:
        # Load raw data
        predictions = np.load(output_dir / "predictions.npy")
        labels = np.load(output_dir / "labels.npy")
        
        # Detailed inspection of data
        logger.info(f"Predictions: shape={predictions.shape}, dtype={predictions.dtype}")
        logger.info(f"Labels: shape={labels.shape}, dtype={labels.dtype}")
        
        # Handle unequal lengths
        min_len = min(len(predictions), len(labels))
        predictions = predictions[:min_len]
        labels = labels[:min_len]
        
        # Ensure binary format
        pred_binary = predictions.copy()
        if predictions.dtype == float:
            pred_binary = (predictions > 0.5).astype(int)
        
        # Show data distributions
        logger.info(f"Predictions value counts: {np.unique(pred_binary, return_counts=True)}")
        logger.info(f"Labels value counts: {np.unique(labels, return_counts=True)}")
        
        # DIRECTLY calculate confusion matrix components
        # No libraries, just basic numpy operations
        tp = np.sum((pred_binary == 1) & (labels == 1))
        fp = np.sum((pred_binary == 1) & (labels == 0))
        tn = np.sum((pred_binary == 0) & (labels == 0))
        fn = np.sum((pred_binary == 0) & (labels == 1))
        
        # Print raw counts for verification
        logger.info(f"Manually computed confusion matrix components:")
        logger.info(f"TP (pred=1, actual=1): {tp}")
        logger.info(f"FP (pred=1, actual=0): {fp}")
        logger.info(f"TN (pred=0, actual=0): {tn}")
        logger.info(f"FN (pred=0, actual=1): {fn}")
        
        # This should match the log output you shared
        cm_log = np.array([[tn, fp], [fn, tp]])
        logger.info(f"Manual confusion matrix:\n{cm_log}")
        
        # Check against your expected values
        expected_cm = np.array([[11596, 43], [46, 11575]])
        logger.info(f"Expected confusion matrix from logs:\n{expected_cm}")
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        # Find misclassified pairs directly
        misclassified = np.where(pred_binary != labels)[0]
        
        # Create and save CSV
        data = {
            'index': misclassified,
            'true_label': labels[misclassified],
            'predicted_label': pred_binary[misclassified],
            'confidence': predictions[misclassified] if predictions.dtype == float else None
        }
        
        # Try to load features
        try:
            features = np.load(output_dir / "feature_vectors.npy")[:min_len]
            feature_names = []
            
            try:
                import json
                with open(output_dir / "feature_names.json", 'r') as f:
                    feature_names = json.load(f)
            except:
                feature_names = [f"feature_{i}" for i in range(features.shape[1])]
            
            for i, name in enumerate(feature_names):
                if i < features.shape[1]:
                    data[name] = features[misclassified, i]
        except:
            logger.warning("Could not load feature data")
        
        df = pd.DataFrame(data)
        output_path = output_dir / "direct_misclassified.csv"
        df.to_csv(output_path, index=False)
        
        # Print summary
        print("\n=== DIRECT Classification Metrics ===")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        
        print("\n=== Manual Confusion Matrix ===")
        print(f"True Positives: {tp}")
        print(f"False Positives: {fp}")
        print(f"True Negatives: {tn}")
        print(f"False Negatives: {fn}")
        
        # Save to CSV
        metrics_df = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'F1', 'Accuracy', 
                       'True Positives', 'False Positives', 'True Negatives', 'False Negatives'],
            'Value': [precision, recall, f1, accuracy, tp, fp, tn, fn]
        })
        metrics_df.to_csv(output_dir / "direct_metrics.csv", index=False)
        
        print(f"\nResults saved to: {output_path}")
        
        # If all else fails, try labels and predictions swapped just to confirm
        tp_swapped = np.sum((labels == 1) & (pred_binary == 1))
        fp_swapped = np.sum((labels == 1) & (pred_binary == 0))
        tn_swapped = np.sum((labels == 0) & (pred_binary == 0))
        fn_swapped = np.sum((labels == 0) & (pred_binary == 1))
        
        print("\n=== SWAPPED Confusion Matrix (just to check) ===")
        print(f"TP (label=1, pred=1): {tp_swapped}")
        print(f"FP (label=1, pred=0): {fp_swapped}")
        print(f"TN (label=0, pred=0): {tn_swapped}")
        print(f"FN (label=0, pred=1): {fn_swapped}")
        
    except Exception as e:
        logger.error(f"Error in direct analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./output"
    analyze_classification_results(data_dir)