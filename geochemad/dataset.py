from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from .catalog import DatasetRecord
from .preprocessing import PreparedFrame, PreprocessingConfig, prepare_samples
from .vocab import commodity_id


@dataclass
class DatasetBundle:
    record: DatasetRecord
    prepared: PreparedFrame
    site_frame: pd.DataFrame
    neighbor_indices: np.ndarray
    neighbor_distances: np.ndarray
    average_spacing: float
    commodity_id: int

    def site_coords(self) -> np.ndarray:
        return self.site_frame[["X", "Y"]].astype(float).to_numpy(dtype=np.float32)

    def to_summary(self) -> dict[str, Any]:
        return {
            **self.record.to_dict(),
            "feature_count": int(self.prepared.features.shape[1]),
            "average_spacing": float(self.average_spacing),
        }


def build_neighbors(coords: np.ndarray, k_neighbors: int) -> tuple[np.ndarray, np.ndarray, float]:
    tree = cKDTree(coords)
    distances, indices = tree.query(coords, k=min(k_neighbors + 1, len(coords)))
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    if distances.shape[1] > 1:
        distances = distances[:, 1:]
        indices = indices[:, 1:]
    average_spacing = float(np.mean(distances[:, 0])) if distances.size else 0.0
    return indices.astype(np.int64), distances.astype(np.float32), average_spacing


def load_dataset_bundle(
    record: DatasetRecord,
    preprocess: PreprocessingConfig,
    k_neighbors: int,
) -> DatasetBundle:
    sample_frame = pd.read_csv(record.sample_path, encoding="utf-8-sig")
    site_frame = pd.read_csv(record.site_path, encoding="utf-8-sig")
    prepared = prepare_samples(
        sample_frame,
        target_element=record.target,
        config=preprocess,
        preserve_element_identity=True,
    )
    neighbor_indices, neighbor_distances, average_spacing = build_neighbors(prepared.coords, k_neighbors)
    return DatasetBundle(
        record=record,
        prepared=prepared,
        site_frame=site_frame,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
        average_spacing=average_spacing,
        commodity_id=commodity_id(record.target),
    )


class SpatialContextDataset(Dataset):
    def __init__(self, bundle: DatasetBundle):
        self.bundle = bundle

    def __len__(self) -> int:
        return len(self.bundle.prepared.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        coords = self.bundle.prepared.coords_normalized[index]
        feature = self.bundle.prepared.features[index]
        neighbors = self.bundle.neighbor_indices[index]
        neighbor_coords = self.bundle.prepared.coords_normalized[neighbors]
        rel = neighbor_coords - coords[None, :]
        spacing = max(self.bundle.average_spacing, 1e-6)
        rel = rel / spacing
        neighbor_features = self.bundle.prepared.features[neighbors]
        return {
            "coords": torch.from_numpy(coords),
            "feature": torch.from_numpy(feature),
            "neighbor_features": torch.from_numpy(neighbor_features),
            "neighbor_offsets": torch.from_numpy(rel.astype(np.float32)),
            "target": torch.tensor(self.bundle.prepared.target_values[index], dtype=torch.float32),
            "commodity_id": torch.tensor(self.bundle.commodity_id, dtype=torch.long),
        }
