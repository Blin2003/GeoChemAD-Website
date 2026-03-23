from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .models import AutoEncoder, ElementTransformer, VariationalAutoEncoder


BaselineName = Literal[
    "zscore",
    "mahalanobis",
    "knn",
    "iforest",
    "ocsvm",
    "ae",
    "vae",
    "transformer",
]


@dataclass
class BaselineTrainConfig:
    epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 1e-3
    device: str = "cpu"


def zscore_scores(features: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(features), axis=1)


def mahalanobis_scores(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    cov = np.cov(features.T) + np.eye(features.shape[1]) * 1e-5
    inv = np.linalg.pinv(cov)
    diff = features - mean
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, inv, diff))


def knn_scores(features: np.ndarray, n_neighbors: int = 10) -> np.ndarray:
    model = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(features)))
    model.fit(features)
    distances, _ = model.kneighbors(features)
    if distances.shape[1] > 1:
        distances = distances[:, 1:]
    return distances.mean(axis=1)


def iforest_scores(features: np.ndarray, seed: int) -> np.ndarray:
    model = IsolationForest(random_state=seed, contamination="auto")
    model.fit(features)
    return -model.score_samples(features)


def ocsvm_scores(features: np.ndarray) -> np.ndarray:
    model = OneClassSVM(gamma="scale", nu=0.05)
    model.fit(features)
    return -model.score_samples(features)


def _train_reconstruction_model(
    model: nn.Module,
    features: np.ndarray,
    config: BaselineTrainConfig,
    vae: bool = False,
) -> np.ndarray:
    dataset = TensorDataset(torch.from_numpy(features.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    device = torch.device(config.device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    for _ in range(config.epochs):
        model.train()
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            if vae:
                recon, mu, logvar = model(batch)
                recon_loss = torch.mean((recon - batch) ** 2)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + 1e-3 * kl
            else:
                recon = model(batch)
                loss = torch.mean((recon - batch) ** 2)
            loss.backward()
            optimizer.step()
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in DataLoader(dataset, batch_size=config.batch_size):
            batch = batch.to(device)
            if vae:
                recon, _, _ = model(batch)
            else:
                recon = model(batch)
            score = torch.mean((recon - batch) ** 2, dim=1)
            scores.append(score.cpu().numpy())
    return np.concatenate(scores)


def run_baseline(
    name: BaselineName,
    features: np.ndarray,
    config: BaselineTrainConfig,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if name == "zscore":
        return zscore_scores(features)
    if name == "mahalanobis":
        return mahalanobis_scores(features)
    if name == "knn":
        return knn_scores(features)
    if name == "iforest":
        return iforest_scores(features, seed=seed)
    if name == "ocsvm":
        return ocsvm_scores(features)
    if name == "ae":
        model = AutoEncoder(features.shape[1])
        return _train_reconstruction_model(model, features, config)
    if name == "vae":
        model = VariationalAutoEncoder(features.shape[1])
        return _train_reconstruction_model(model, features, config, vae=True)
    if name == "transformer":
        model = ElementTransformer(features.shape[1])
        return _train_reconstruction_model(model, features, config)
    raise ValueError(f"Unsupported baseline: {name}")
