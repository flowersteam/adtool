import json
from datetime import datetime
from pathlib import Path


def create_run_dir(output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir).resolve() / f"analysis_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_summary(summary):
    with (summary.run_dir / "summary.json").open("w") as handle:
        json.dump(summary.to_payload(), handle, indent=2)
