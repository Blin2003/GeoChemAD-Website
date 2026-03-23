from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .baselines import BaselineName, BaselineTrainConfig, run_baseline
from .catalog import DatasetRecord
from .dataset import SpatialContextDataset, load_dataset_bundle
from .interpolation import generate_anomaly_map
from .metrics import repeated_auc, spatial_targeting_metrics
from .models import GeoChemFormer, ModelConfig
from .preprocessing import PreprocessingConfig


@dataclass
class ExperimentConfig:
    dataset_id: str
    model_name: str = "geochemformer"
    preprocess: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    k_neighbors: int = 128
    context_epochs: int = 40
    anomaly_epochs: int = 60
    batch_size: int = 128
    learning_rate: float = 1e-3
    repeats: int = 20
    background_size: int = 256
    seed: int = 42
    device: str = "cpu"
    interpolation_method: str = "idw"
    interpolation_grid_size: int = 96
    freeze_context_after_pretrain: bool = True


def _device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_geochemformer(
    record: DatasetRecord,
    config: ExperimentConfig,
    should_stop: callable | None = None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    bundle = load_dataset_bundle(record, config.preprocess, config.k_neighbors)
    device = _device(config.device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    dataset = SpatialContextDataset(bundle)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = GeoChemFormer(
        feature_dim=bundle.prepared.features.shape[1],
        commodity_count=6,
        k_neighbors=config.k_neighbors,
        config=ModelConfig(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
        ),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    context_history: list[float] = []
    for _ in range(config.context_epochs):
        if should_stop and should_stop():
            raise RuntimeError("Run cancelled by user.")
        model.train()
        losses: list[float] = []
        for batch in loader:
            if should_stop and should_stop():
                raise RuntimeError("Run cancelled by user.")
            coords = batch["coords"].to(device)
            commodity_ids = batch["commodity_id"].to(device)
            neighbor_offsets = batch["neighbor_offsets"].to(device)
            neighbor_features = batch["neighbor_features"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad()
            pred, _ = model.predict_target(coords, commodity_ids, neighbor_offsets, neighbor_features)
            loss = torch.mean((pred - target) ** 2)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        context_history.append(float(np.mean(losses)))
    anomaly_history: list[float] = []
    if config.freeze_context_after_pretrain:
        for module in [model.target_embedding, model.query_proj, model.neighbor_proj, model.context_encoder]:
            for parameter in module.parameters():
                parameter.requires_grad = False
        anomaly_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.Adam(anomaly_parameters, lr=config.learning_rate)
    for _ in range(config.anomaly_epochs):
        if should_stop and should_stop():
            raise RuntimeError("Run cancelled by user.")
        model.train()
        losses = []
        for batch in loader:
            if should_stop and should_stop():
                raise RuntimeError("Run cancelled by user.")
            coords = batch["coords"].to(device)
            commodity_ids = batch["commodity_id"].to(device)
            features = batch["feature"].to(device)
            neighbor_offsets = batch["neighbor_offsets"].to(device)
            neighbor_features = batch["neighbor_features"].to(device)
            optimizer.zero_grad()
            if config.freeze_context_after_pretrain:
                with torch.no_grad():
                    context = model.encode_context(coords, commodity_ids, neighbor_offsets, neighbor_features)
                recon = model.reconstruct_from_context(context.detach(), features)
            else:
                _, recon, _ = model(coords, commodity_ids, neighbor_offsets, neighbor_features, features)
            loss = torch.mean((recon - features) ** 2)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        anomaly_history.append(float(np.mean(losses)))
    model.eval()
    sample_scores: list[np.ndarray] = []
    latent_context: list[np.ndarray] = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=config.batch_size):
            coords = batch["coords"].to(device)
            commodity_ids = batch["commodity_id"].to(device)
            features = batch["feature"].to(device)
            neighbor_offsets = batch["neighbor_offsets"].to(device)
            neighbor_features = batch["neighbor_features"].to(device)
            if config.freeze_context_after_pretrain:
                context = model.encode_context(coords, commodity_ids, neighbor_offsets, neighbor_features)
                recon = model.reconstruct_from_context(context, features)
            else:
                _, recon, context = model(coords, commodity_ids, neighbor_offsets, neighbor_features, features)
            score = torch.mean((recon - features) ** 2, dim=1)
            sample_scores.append(score.cpu().numpy())
            latent_context.append(context.cpu().numpy())
    scores = np.concatenate(sample_scores)
    diagnostics = {
        "context_loss_history": context_history,
        "anomaly_loss_history": anomaly_history,
        "average_spacing": bundle.average_spacing,
        "feature_count": int(bundle.prepared.features.shape[1]),
        "feature_diagnostics": bundle.prepared.feature_diagnostics,
    }
    anomaly_map = generate_anomaly_map(
        bundle.prepared.coords,
        scores,
        method=config.interpolation_method,
        grid_size=config.interpolation_grid_size,
    )
    payload = {
        "coords": bundle.prepared.coords.tolist(),
        "scores": scores.tolist(),
        "sample_ids": bundle.prepared.sample_ids,
        "sites": bundle.site_coords().tolist(),
        "latent_context": np.concatenate(latent_context).tolist(),
        "anomaly_map": anomaly_map,
    }
    metrics = repeated_auc(
        sample_scores=scores,
        sample_coords=bundle.prepared.coords,
        site_coords=bundle.site_coords(),
        repeats=config.repeats,
        background_size=config.background_size,
        exclusion_radius=max(bundle.average_spacing * 2.0, 1e-3),
        seed=config.seed,
    )
    metrics.update(spatial_targeting_metrics(scores, bundle.prepared.coords, bundle.site_coords()))
    return scores, {**metrics, **diagnostics}, payload


def train_experiment(
    record: DatasetRecord,
    config: ExperimentConfig,
    should_stop: callable | None = None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    bundle = load_dataset_bundle(record, config.preprocess, config.k_neighbors)
    if config.model_name == "geochemformer":
        return train_geochemformer(record, config, should_stop=should_stop)
    scores = run_baseline(
        name=config.model_name,  # type: ignore[arg-type]
        features=bundle.prepared.features,
        config=BaselineTrainConfig(
            epochs=config.anomaly_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            device=config.device,
        ),
        seed=config.seed,
    )
    metrics = repeated_auc(
        sample_scores=scores,
        sample_coords=bundle.prepared.coords,
        site_coords=bundle.site_coords(),
        repeats=config.repeats,
        background_size=config.background_size,
        exclusion_radius=max(bundle.average_spacing * 2.0, 1e-3),
        seed=config.seed,
    )
    metrics.update(spatial_targeting_metrics(scores, bundle.prepared.coords, bundle.site_coords()))
    payload = {
        "coords": bundle.prepared.coords.tolist(),
        "scores": scores.tolist(),
        "sample_ids": bundle.prepared.sample_ids,
        "sites": bundle.site_coords().tolist(),
        "anomaly_map": generate_anomaly_map(
            bundle.prepared.coords,
            scores,
            method=config.interpolation_method,
            grid_size=config.interpolation_grid_size,
        ),
    }
    metrics["feature_diagnostics"] = bundle.prepared.feature_diagnostics
    return scores, metrics, payload


def save_run(run_dir: Path, config: ExperimentConfig, metrics: dict[str, Any], payload: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "payload.json").write_text(json.dumps(payload), encoding="utf-8")


def benchmark_models(
    records: list[DatasetRecord],
    base_config: ExperimentConfig,
    models: list[str],
) -> dict[str, Any]:
    table: list[dict[str, Any]] = []
    for record in records:
        for model in models:
            config = deepcopy(base_config)
            config.dataset_id = record.subset_id
            config.model_name = model
            _, metrics, _ = train_experiment(record, config)
            table.append(
                {
                    "dataset_id": record.subset_id,
                    "model_name": model,
                    "auc_mean": metrics["auc_mean"],
                    "auc_std": metrics["auc_std"],
                    "ap_mean": metrics["ap_mean"],
                    "dtd_top_mean": metrics["dtd_top_mean"],
                }
            )
    grouped: dict[str, dict[str, float]] = {}
    for row in table:
        grouped.setdefault(row["model_name"], {})
        grouped[row["model_name"]][row["dataset_id"]] = row["auc_mean"]
    return {"rows": table, "auc_matrix": grouped}
