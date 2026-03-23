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
from geochemad.training import ExperimentConfig, save_run, train_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GeoChemAD experiment")
    parser.add_argument("--dataset", required=True, help="Dataset id, e.g. area1_sediment_au")
    parser.add_argument("--model", default="geochemformer")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--context-epochs", type=int, default=20)
    parser.add_argument("--anomaly-epochs", type=int, default=30)
    parser.add_argument("--k-neighbors", type=int, default=128)
    parser.add_argument("--feature-strategy", default="none")
    parser.add_argument("--feature-top-k", type=int, default=24)
    parser.add_argument("--pca-components", type=int, default=24)
    parser.add_argument("--interpolation-method", default="idw")
    args = parser.parse_args()

    paths = ProjectPaths()
    datasets = {record.subset_id: record for record in scan_datasets(paths.data_dir)}
    if args.dataset not in datasets:
        raise SystemExit(f"Dataset not found: {args.dataset}")
    config = ExperimentConfig(
        dataset_id=args.dataset,
        model_name=args.model,
        context_epochs=args.context_epochs,
        anomaly_epochs=args.anomaly_epochs,
        k_neighbors=args.k_neighbors,
        device=args.device,
    )
    config.preprocess.feature_selection.strategy = args.feature_strategy
    config.preprocess.feature_selection.top_k = args.feature_top_k
    config.preprocess.feature_selection.pca_components = args.pca_components
    config.interpolation_method = args.interpolation_method
    _, metrics, payload = train_experiment(datasets[args.dataset], config)
    run_dir = paths.artifact_dir / "runs" / f"cli_{args.dataset}_{args.model}"
    save_run(run_dir, config, metrics, payload)
    print(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
