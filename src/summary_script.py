import json
import time
from pathlib import Path

# Create a minimal pipeline summary
pipeline_summary = {
    "duration": 0,
    "stages": {
        "preprocess": {"duration": 0, "records_processed": 0},
        "embed": {"duration": 0, "records_processed": 0},
        "index": {"duration": 0, "records_processed": 0},
        "impute": {"duration": 0, "records_processed": 0},
        "query": {"duration": 0, "records_processed": 0},
        "features": {"duration": 0, "records_processed": 0},
        "classify": {"duration": 0, "records_processed": 0},
        "cluster": {"duration": 0, "records_processed": 0},
        "analyze": {"duration": 0, "records_processed": 0}
    },
    "mode": "dev",
    "timestamp": time.time()
}

# Save to output directory
output_dir = Path("output")  # Update this path if your output dir is different
output_dir.mkdir(exist_ok=True)
with open(output_dir / "pipeline_summary.json", 'w') as f:
    json.dump(pipeline_summary, f, indent=2)

print("Created minimal pipeline_summary.json file")