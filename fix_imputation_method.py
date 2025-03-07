#!/usr/bin/env python3
"""
Fix missing _save_results method in imputation.py.

This script adds the missing _save_results method to the Imputator class.
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fix_imputation_method")

def fix_imputation_save_results():
    """
    Add the missing _save_results method to the Imputator class.
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Find imputation.py
    imputation_path = Path("src") / "imputation.py"
    
    if not imputation_path.exists():
        logger.error(f"Imputation file not found at {imputation_path}")
        return False
    
    # Read the file
    with open(imputation_path, 'r') as f:
        content = f.read()
    
    # Check if _save_results is called but not defined
    if "_save_results(" not in content and "self._save_results()" in content:
        logger.info("Found call to _save_results() but method is not defined")
        
        # Define the missing method
        missing_method = """
    def _save_results(self):
        \"\"\"
        Save imputation results.
        \"\"\"
        output_dir = Path(self.config['system']['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        # Save imputed values
        with open(output_dir / "imputed_values.json", 'w') as f:
            json.dump(self.imputed_values, f, indent=2)
        
        # Save sample of imputed values
        sample_size = min(100, len(self.imputed_values))
        sample_records = list(self.imputed_values.keys())[:sample_size]
        sample_values = {r: self.imputed_values[r] for r in sample_records}
        
        with open(output_dir / "imputed_values_sample.json", 'w') as f:
            json.dump(sample_values, f, indent=2)
        
        # Save statistics
        field_stats = {}
        for record_id, fields in self.imputed_values.items():
            for field, data in fields.items():
                if field not in field_stats:
                    field_stats[field] = 0
                
                field_stats[field] += 1
        
        with open(output_dir / "imputation_statistics.json", 'w') as f:
            json.dump(field_stats, f, indent=2)
        
        # Save final checkpoint
        checkpoint_path = Path(self.config['system']['checkpoint_dir']) / "imputation_final.ckpt"
        save_checkpoint({
            'imputed_values': self.imputed_values,
            'processed_records': list(self.imputed_values.keys())
        }, checkpoint_path)
        
        logger.info("Imputation results saved to %s", output_dir)
"""
        
        # Add the method before the get_imputed_value method or at the end of the class
        if "get_imputed_value" in content:
            # Insert before get_imputed_value
            new_content = content.replace("    def get_imputed_value", missing_method + "\n    def get_imputed_value")
        else:
            # Add at the end of the class before any closing comments
            import re
            class_end = re.search(r'def __del__\(self\):.*?pass', content, re.DOTALL)
            if class_end:
                position = class_end.end()
                new_content = content[:position] + "\n" + missing_method + "\n" + content[position:]
            else:
                # Add at the very end
                new_content = content + "\n" + missing_method
        
        # Backup the original file
        backup_path = str(imputation_path) + ".bak"
        import shutil
        shutil.copy(imputation_path, backup_path)
        logger.info(f"Backed up original file to {backup_path}")
        
        # Write the updated file
        with open(imputation_path, 'w') as f:
            f.write(new_content)
        
        logger.info(f"Added _save_results method to {imputation_path}")
        return True
    
    else:
        logger.error("Could not locate where to add _save_results method")
        return False

def main():
    """Main entry point."""
    # Fix imputation save results
    if fix_imputation_save_results():
        logger.info("Imputation method fixed!")
        logger.info("\nNext steps:")
        logger.info("1. Run the imputation stage again: python main.py --component impute")
    else:
        logger.error("Failed to fix imputation method")

if __name__ == "__main__":
    main()
