from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geochemad.catalog import scan_datasets
from geochemad.settings import ProjectPaths
from geochemad.training import ExperimentConfig, benchmark_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark matrix across datasets and models")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--context-epochs", type=int, default=10)
    parser.add_argument("--anomaly-epochs", type=int, default=20)
    parser.add_argument("--feature-strategy", default="none")
    args = parser.parse_args()

    paths = ProjectPaths()
    records = scan_datasets(paths.data_dir)
    if args.datasets:
        wanted = set(args.datasets)
        records = [record for record in records if record.subset_id in wanted]
    config = ExperimentConfig(
        dataset_id=records[0].subset_id,
        device=args.device,
        context_epochs=args.context_epochs,
        anomaly_epochs=args.anomaly_epochs,
    )
    config.preprocess.feature_selection.strategy = args.feature_strategy
    result = benchmark_models(records, config, args.models)
    output_path = paths.artifact_dir / "benchmark_latest.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "rows": len(result["rows"])}, indent=2))


if __name__ == "__main__":
    main()
