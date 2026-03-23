from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .catalog import DatasetRecord
from .runs import RunRegistry
from .settings import ProjectPaths
from .training import ExperimentConfig, benchmark_models
from .uploads import load_uploaded_datasets, save_uploaded_dataset, slugify


class ExperimentPayload(BaseModel):
    dataset_id: str
    model_name: str = Field(default="geochemformer")
    compositional_transform: str = Field(default="ilr")
    hidden_dim: int = Field(default=128, ge=32, le=512)
    num_heads: int = Field(default=4, ge=1, le=8)
    num_layers: int = Field(default=3, ge=1, le=8)
    k_neighbors: int = Field(default=128, ge=4, le=512)
    context_epochs: int = Field(default=20, ge=1, le=400)
    anomaly_epochs: int = Field(default=30, ge=1, le=400)
    batch_size: int = Field(default=128, ge=8, le=2048)
    learning_rate: float = Field(default=1e-3, gt=0.0, lt=1.0)
    repeats: int = Field(default=20, ge=1, le=100)
    background_size: int = Field(default=256, ge=8, le=10000)
    seed: int = Field(default=42)
    device: str = Field(default="cpu")
    feature_strategy: str = Field(default="none")
    feature_top_k: int = Field(default=24, ge=2, le=128)
    pca_components: int = Field(default=24, ge=2, le=128)
    interpolation_method: str = Field(default="idw")
    interpolation_grid_size: int = Field(default=96, ge=24, le=256)

    def to_config(self) -> ExperimentConfig:
        config = ExperimentConfig(dataset_id=self.dataset_id)
        config.model_name = self.model_name
        config.preprocess.compositional_transform = self.compositional_transform  # type: ignore[misc]
        config.hidden_dim = self.hidden_dim
        config.num_heads = self.num_heads
        config.num_layers = self.num_layers
        config.k_neighbors = self.k_neighbors
        config.context_epochs = self.context_epochs
        config.anomaly_epochs = self.anomaly_epochs
        config.batch_size = self.batch_size
        config.learning_rate = self.learning_rate
        config.repeats = self.repeats
        config.background_size = self.background_size
        config.seed = self.seed
        config.device = self.device
        config.preprocess.feature_selection.strategy = self.feature_strategy
        config.preprocess.feature_selection.top_k = self.feature_top_k
        config.preprocess.feature_selection.pca_components = self.pca_components
        config.interpolation_method = self.interpolation_method
        config.interpolation_grid_size = self.interpolation_grid_size
        return config


def build_config(
    dataset_id: str,
    model_name: str,
    compositional_transform: str,
    hidden_dim: int,
    num_heads: int,
    num_layers: int,
    k_neighbors: int,
    context_epochs: int,
    anomaly_epochs: int,
    batch_size: int,
    learning_rate: float,
    repeats: int,
    background_size: int,
    seed: int,
    device: str,
    feature_strategy: str,
    feature_top_k: int,
    pca_components: int,
    interpolation_method: str,
    interpolation_grid_size: int,
) -> ExperimentConfig:
    config = ExperimentConfig(dataset_id=dataset_id)
    config.model_name = model_name
    config.preprocess.compositional_transform = compositional_transform  # type: ignore[misc]
    config.hidden_dim = hidden_dim
    config.num_heads = num_heads
    config.num_layers = num_layers
    config.k_neighbors = k_neighbors
    config.context_epochs = context_epochs
    config.anomaly_epochs = anomaly_epochs
    config.batch_size = batch_size
    config.learning_rate = learning_rate
    config.repeats = repeats
    config.background_size = background_size
    config.seed = seed
    config.device = device
    config.preprocess.feature_selection.strategy = feature_strategy
    config.preprocess.feature_selection.top_k = feature_top_k
    config.preprocess.feature_selection.pca_components = pca_components
    config.interpolation_method = interpolation_method
    config.interpolation_grid_size = interpolation_grid_size
    return config


class BenchmarkPayload(BaseModel):
    dataset_ids: list[str]
    models: list[str]
    compositional_transform: str = Field(default="ilr")
    feature_strategy: str = Field(default="none")
    feature_top_k: int = Field(default=24, ge=2, le=128)
    pca_components: int = Field(default=24, ge=2, le=128)
    context_epochs: int = Field(default=10, ge=1, le=400)
    anomaly_epochs: int = Field(default=20, ge=1, le=400)
    k_neighbors: int = Field(default=128, ge=4, le=512)
    batch_size: int = Field(default=128, ge=8, le=2048)
    repeats: int = Field(default=10, ge=1, le=100)
    background_size: int = Field(default=256, ge=8, le=10000)
    device: str = Field(default="cpu")

    def to_config(self, dataset_id: str) -> ExperimentConfig:
        config = ExperimentConfig(dataset_id=dataset_id)
        config.preprocess.compositional_transform = self.compositional_transform  # type: ignore[misc]
        config.preprocess.feature_selection.strategy = self.feature_strategy
        config.preprocess.feature_selection.top_k = self.feature_top_k
        config.preprocess.feature_selection.pca_components = self.pca_components
        config.context_epochs = self.context_epochs
        config.anomaly_epochs = self.anomaly_epochs
        config.k_neighbors = self.k_neighbors
        config.batch_size = self.batch_size
        config.repeats = self.repeats
        config.background_size = self.background_size
        config.device = self.device
        return config


def create_app(paths: ProjectPaths | None = None) -> FastAPI:
    paths = paths or ProjectPaths()
    paths.ensure()
    app = FastAPI(title="GeoChemAD")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    datasets = load_uploaded_datasets(paths.upload_dir)
    dataset_index: dict[str, DatasetRecord] = {record.subset_id: record for record in datasets}
    registry = RunRegistry(paths.artifact_dir / "runs", dataset_index)
    app.state.paths = paths
    app.state.datasets = datasets
    app.state.registry = registry

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/datasets")
    def list_datasets() -> list[dict[str, Any]]:
        return [record.to_dict() for record in dataset_index.values()]

    @app.get("/api/datasets/{dataset_id}")
    def dataset_detail(dataset_id: str) -> dict[str, Any]:
        record = dataset_index.get(dataset_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return record.to_dict()

    @app.get("/api/datasets/{dataset_id}/preview")
    def dataset_preview(dataset_id: str, limit: int = 5) -> dict[str, Any]:
        record = dataset_index.get(dataset_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        import pandas as pd

        sample = pd.read_csv(record.sample_path, encoding="utf-8-sig", nrows=limit)
        sites = pd.read_csv(record.site_path, encoding="utf-8-sig", nrows=limit)
        return {
            "sample_columns": sample.columns.tolist(),
            "sample_rows": sample.fillna("").to_dict(orient="records"),
            "site_columns": sites.columns.tolist(),
            "site_rows": sites.fillna("").to_dict(orient="records"),
        }

    @app.post("/api/runs")
    def create_run(payload: ExperimentPayload) -> dict[str, Any]:
        state = registry.start(payload.to_config())
        return asdict(state)

    @app.post("/api/runs/upload")
    async def create_run_from_upload(
        dataset_name: str = Form(...),
        target_element: str = Form(...),
        sample_file: UploadFile = File(...),
        site_file: UploadFile = File(...),
        model_name: str = Form("geochemformer"),
        compositional_transform: str = Form("ilr"),
        hidden_dim: int = Form(128),
        num_heads: int = Form(4),
        num_layers: int = Form(3),
        k_neighbors: int = Form(128),
        context_epochs: int = Form(20),
        anomaly_epochs: int = Form(30),
        batch_size: int = Form(128),
        learning_rate: float = Form(1e-3),
        repeats: int = Form(20),
        background_size: int = Form(256),
        seed: int = Form(42),
        device: str = Form("cpu"),
        feature_strategy: str = Form("none"),
        feature_top_k: int = Form(24),
        pca_components: int = Form(24),
        interpolation_method: str = Form("idw"),
        interpolation_grid_size: int = Form(96),
    ) -> dict[str, Any]:
        record = save_uploaded_dataset(
            paths.upload_dir,
            dataset_name=dataset_name,
            target_element=target_element,
            sample_file=sample_file,
            site_file=site_file,
        )
        meta_path = paths.upload_dir / record.subset_id / "meta.json"
        meta_path.write_text(
            json.dumps({"dataset_name": dataset_name, "target_element": target_element}, indent=2),
            encoding="utf-8",
        )
        config = build_config(
            dataset_id=record.subset_id,
            model_name=model_name,
            compositional_transform=compositional_transform,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            k_neighbors=k_neighbors,
            context_epochs=context_epochs,
            anomaly_epochs=anomaly_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            repeats=repeats,
            background_size=background_size,
            seed=seed,
            device=device,
            feature_strategy=feature_strategy,
            feature_top_k=feature_top_k,
            pca_components=pca_components,
            interpolation_method=interpolation_method,
            interpolation_grid_size=interpolation_grid_size,
        )
        state = registry.start_with_record(record, config)
        return asdict(state)

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str) -> dict[str, Any]:
        state = registry.cancel(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return asdict(state)

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        return [asdict(item) for item in registry.list_runs()]

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        state = registry.get_run(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Run not found")
        data = asdict(state)
        run_dir = paths.artifact_dir / "runs" / run_id
        payload_path = run_dir / "payload.json"
        config_path = run_dir / "config.json"
        if payload_path.exists():
            data["payload"] = json.loads(payload_path.read_text(encoding="utf-8"))
        if config_path.exists():
            data["config"] = json.loads(config_path.read_text(encoding="utf-8"))
        return data

    @app.post("/api/benchmarks")
    def create_benchmark(payload: BenchmarkPayload) -> dict[str, Any]:
        records = []
        for dataset_id in payload.dataset_ids:
            record = dataset_index.get(dataset_id)
            if record is None:
                raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
            records.append(record)
        result = benchmark_models(records, payload.to_config(records[0].subset_id), payload.models)
        output_path = paths.artifact_dir / "benchmark_latest.json"
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    @app.get("/api/benchmarks/latest")
    def latest_benchmark() -> dict[str, Any]:
        output_path = paths.artifact_dir / "benchmark_latest.json"
        if not output_path.exists():
            raise HTTPException(status_code=404, detail="No benchmark result found")
        return json.loads(output_path.read_text(encoding="utf-8"))

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(paths.web_dir / "index.html")

    app.mount("/web", StaticFiles(directory=paths.web_dir), name="web")
    return app
