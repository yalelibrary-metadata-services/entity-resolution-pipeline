#!/usr/bin/env python3
"""
Data inspection script that analyzes the raw content of prediction and label files
to identify the source of the discrepancy between logs and saved data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def inspect_classification_data(output_dir="./output"):
    """Thoroughly inspect classification data files to identify discrepancies."""
    output_dir = Path(output_dir)
    
    # 1. LOAD RAW DATA
    logger.info("Loading raw prediction and label data...")
    try:
        predictions_path = output_dir / "predictions.npy"
        labels_path = output_dir / "labels.npy"
        
        if not predictions_path.exists() or not labels_path.exists():
            logger.error("Prediction or label files not found")
            return
            
        predictions = np.load(predictions_path)
        labels = np.load(labels_path)
        
        # 2. DETAILED INSPECTION
        pred_type = predictions.dtype
        label_type = labels.dtype
        pred_shape = predictions.shape
        label_shape = labels.shape
        
        logger.info(f"Predictions file: {predictions_path}")
        logger.info(f"  - Shape: {pred_shape}")
        logger.info(f"  - Data type: {pred_type}")
        logger.info(f"  - Min value: {predictions.min()}")
        logger.info(f"  - Max value: {predictions.max()}")
        
        logger.info(f"Labels file: {labels_path}")
        logger.info(f"  - Shape: {label_shape}")
        logger.info(f"  - Data type: {label_type}")
        logger.info(f"  - Min value: {labels.min()}")
        logger.info(f"  - Max value: {labels.max()}")
        
        # 3. ALIGNMENT CHECK
        if pred_shape != label_shape:
            logger.warning(f"Shape mismatch: predictions {pred_shape}, labels {label_shape}")
            min_len = min(len(predictions), len(labels))
            predictions = predictions[:min_len]
            labels = labels[:min_len]
            logger.info(f"Truncated both arrays to length {min_len}")
        
        # 4. DISTRIBUTION ANALYSIS
        pred_binary = predictions.copy()
        if pred_type == float:
            pred_binary = (predictions > 0.5).astype(int)
            
        # Value counts
        pred_values, pred_counts = np.unique(pred_binary, return_counts=True)
        label_values, label_counts = np.unique(labels, return_counts=True)
        
        logger.info("Prediction value distribution:")
        for val, count in zip(pred_values, pred_counts):
            logger.info(f"  {val}: {count} ({count/len(pred_binary)*100:.2f}%)")
            
        logger.info("Label value distribution:")
        for val, count in zip(label_values, label_counts):
            logger.info(f"  {val}: {count} ({count/len(labels)*100:.2f}%)")
        
        # 5. DATA CONSISTENCY CHECK
        # Expected counts from logs
        expected_tn = 11596
        expected_fp = 43
        expected_fn = 46
        expected_tp = 11575
        expected_total = expected_tn + expected_fp + expected_fn + expected_tp
        
        # Actual counts
        actual_tn = np.sum((pred_binary == 0) & (labels == 0))
        actual_fp = np.sum((pred_binary == 1) & (labels == 0))
        actual_fn = np.sum((pred_binary == 0) & (labels == 1))
        actual_tp = np.sum((pred_binary == 1) & (labels == 1))
        actual_total = len(predictions)
        
        # Compare expected vs actual
        logger.info("\nExpected vs Actual Confusion Matrix:")
        logger.info(f"                Expected  Actual  Difference")
        logger.info(f"True Negatives:  {expected_tn:7d}  {actual_tn:6d}  {actual_tn-expected_tn:+10d}")
        logger.info(f"False Positives: {expected_fp:7d}  {actual_fp:6d}  {actual_fp-expected_fp:+10d}")
        logger.info(f"False Negatives: {expected_fn:7d}  {actual_fn:6d}  {actual_fn-expected_fn:+10d}")
        logger.info(f"True Positives:  {expected_tp:7d}  {actual_tp:6d}  {actual_tp-expected_tp:+10d}")
        logger.info(f"Total:           {expected_total:7d}  {actual_total:6d}  {actual_total-expected_total:+10d}")
        
        # 6. DATA VISUALIZATION
        # Look at first 20 samples
        sample_size = min(20, len(predictions))
        sample_indices = np.arange(sample_size)
        
        logger.info(f"\nFirst {sample_size} samples:")
        logger.info(f"Index  Prediction  Binary Pred  Label  Match?")
        for i in sample_indices:
            binary_pred = pred_binary[i]
            match = "✓" if binary_pred == labels[i] else "✗"
            logger.info(f"{i:5d}  {predictions[i]:10.4f}  {binary_pred:11d}  {labels[i]:5d}  {match}")
        
        # 7. SAVE FINDINGS TO FILE
        # Create a sample DataFrame with raw data
        sample_df = pd.DataFrame({
            'index': np.arange(min(1000, len(predictions))),
            'prediction_raw': predictions[:1000],
            'prediction_binary': pred_binary[:1000],
            'label': labels[:1000],
            'match': pred_binary[:1000] == labels[:1000]
        })
        
        sample_csv_path = output_dir / "raw_data_sample.csv"
        sample_df.to_csv(sample_csv_path, index=False)
        logger.info(f"\nSaved raw data sample to {sample_csv_path}")
        
        # 8. ANALYZE MISMATCHES
        mismatch_indices = np.where(pred_binary != labels)[0]
        logger.info(f"\nFound {len(mismatch_indices)} mismatched predictions")
        
        if len(mismatch_indices) > 0:
            # Store mismatches
            mismatch_df = pd.DataFrame({
                'index': mismatch_indices,
                'prediction_raw': predictions[mismatch_indices],
                'prediction_binary': pred_binary[mismatch_indices],
                'label': labels[mismatch_indices]
            })
            
            mismatch_csv_path = output_dir / "prediction_label_mismatches.csv"
            mismatch_df.to_csv(mismatch_csv_path, index=False)
            logger.info(f"Saved mismatches to {mismatch_csv_path}")
            
            # Print mismatch stats
            mismatch_0_to_1 = np.sum((pred_binary == 0) & (labels == 1))
            mismatch_1_to_0 = np.sum((pred_binary == 1) & (labels == 0))
            
            logger.info(f"Mismatch types:")
            logger.info(f"  Predicted 0, Actual 1: {mismatch_0_to_1} ({mismatch_0_to_1/len(mismatch_indices)*100:.2f}%)")
            logger.info(f"  Predicted 1, Actual 0: {mismatch_1_to_0} ({mismatch_1_to_0/len(mismatch_indices)*100:.2f}%)")
        
        # 9. ATTEMPT LABEL CORRECTION (if applicable)
        if actual_tn == 0 and actual_fp == 0 and labels.max() == 1 and labels.min() == 1:
            logger.warning("\nPOTENTIAL DATA ISSUE DETECTED: All labels appear to be 1")
            logger.warning("This could explain why we see 0 true negatives and 0 false positives")
            
            # This could be an issue with the dataset, might need to fix the labels based on expected results
            suggestion = """
SUGGESTION: Your label file appears to contain only 1's, which doesn't match the expected distribution.
This could be due to:
1. A data generation error
2. The wrong file being loaded 
3. Data corruption during saving

You may need to check how the labels were saved, or reconstruct them from the original data.
"""
            print(suggestion)
            
        # 10. CONCLUSION
        conclusion = """
DATA ANALYSIS SUMMARY:
1. The data in your prediction and label files does NOT match the confusion matrix in your logs
2. Check for any code that might have modified these files after initial evaluation
3. Consider regenerating your data files from the original classification step
"""
        print(conclusion)
        
    except Exception as e:
        logger.error(f"Error inspecting data: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./output"
    inspect_classification_data(data_dir)