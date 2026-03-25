from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .models import (
    AutoEncoder,
    CascadeGenerator,
    ElementTransformer,
    TabularDiscriminator,
    VariationalAutoEncoder,
)


BaselineName = Literal[
    "zscore",
    "mahalanobis",
    "knn",
    "iforest",
    "ocsvm",
    "ae",
    "vae",
    "vaegan",
    "cascade_gan",
    "transformer",
]


@dataclass
class BaselineTrainConfig:
    epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 1e-3
    device: str = "cpu"


def _device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    device = _device(config.device)
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


def _train_vaegan(features: np.ndarray, config: BaselineTrainConfig, latent_dim: int = 32) -> np.ndarray:
    dataset = TensorDataset(torch.from_numpy(features.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    device = _device(config.device)
    generator = VariationalAutoEncoder(features.shape[1], latent_dim=latent_dim).to(device)
    discriminator = TabularDiscriminator(features.shape[1]).to(device)
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=config.learning_rate)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.learning_rate)
    for _ in range(config.epochs):
        generator.train()
        discriminator.train()
        for (batch,) in loader:
            batch = batch.to(device)
            with torch.no_grad():
                recon_detached, _, _ = generator(batch)
            disc_optimizer.zero_grad()
            real_logits, _ = discriminator(batch)
            fake_logits, _ = discriminator(recon_detached)
            real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
            fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
            disc_loss = 0.5 * (real_loss + fake_loss)
            disc_loss.backward()
            disc_optimizer.step()

            gen_optimizer.zero_grad()
            recon, mu, logvar = generator(batch)
            fake_logits, fake_features = discriminator(recon)
            _, real_features = discriminator(batch)
            recon_loss = torch.mean((recon - batch) ** 2)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            adv_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
            feature_match = torch.mean((fake_features - real_features.detach()) ** 2)
            gen_loss = recon_loss + 1e-3 * kl + 0.05 * adv_loss + 0.1 * feature_match
            gen_loss.backward()
            gen_optimizer.step()
    generator.eval()
    discriminator.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in DataLoader(dataset, batch_size=config.batch_size):
            batch = batch.to(device)
            recon, _, _ = generator(batch)
            fake_logits, fake_features = discriminator(recon)
            _, real_features = discriminator(batch)
            recon_error = torch.mean((recon - batch) ** 2, dim=1)
            feature_error = torch.mean((fake_features - real_features) ** 2, dim=1)
            adv_score = torch.sigmoid(-fake_logits.squeeze(-1))
            score = recon_error + 0.2 * feature_error + 0.05 * adv_score
            scores.append(score.cpu().numpy())
    return np.concatenate(scores)


def _train_cascade_gan(features: np.ndarray, config: BaselineTrainConfig, latent_dim: int = 32) -> np.ndarray:
    dataset = TensorDataset(torch.from_numpy(features.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    device = _device(config.device)
    generator = CascadeGenerator(features.shape[1], latent_dim=latent_dim).to(device)
    discriminator = TabularDiscriminator(features.shape[1]).to(device)
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=config.learning_rate)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.learning_rate)
    for _ in range(config.epochs):
        generator.train()
        discriminator.train()
        for (batch,) in loader:
            batch = batch.to(device)
            with torch.no_grad():
                _, recon_detached = generator(batch)
            disc_optimizer.zero_grad()
            real_logits, _ = discriminator(batch)
            fake_logits, _ = discriminator(recon_detached)
            real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
            fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
            disc_loss = 0.5 * (real_loss + fake_loss)
            disc_loss.backward()
            disc_optimizer.step()

            gen_optimizer.zero_grad()
            coarse, refined = generator(batch)
            fake_logits, fake_features = discriminator(refined)
            _, real_features = discriminator(batch)
            coarse_loss = torch.mean((coarse - batch) ** 2)
            refined_loss = torch.mean((refined - batch) ** 2)
            consistency_loss = torch.mean((refined - coarse.detach()) ** 2)
            adv_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
            feature_match = torch.mean((fake_features - real_features.detach()) ** 2)
            gen_loss = coarse_loss + refined_loss + 0.25 * consistency_loss + 0.05 * adv_loss + 0.1 * feature_match
            gen_loss.backward()
            gen_optimizer.step()
    generator.eval()
    discriminator.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in DataLoader(dataset, batch_size=config.batch_size):
            batch = batch.to(device)
            coarse, refined = generator(batch)
            fake_logits, fake_features = discriminator(refined)
            _, real_features = discriminator(batch)
            coarse_error = torch.mean((coarse - batch) ** 2, dim=1)
            refined_error = torch.mean((refined - batch) ** 2, dim=1)
            refinement_gap = torch.mean((refined - coarse) ** 2, dim=1)
            feature_error = torch.mean((fake_features - real_features) ** 2, dim=1)
            adv_score = torch.sigmoid(-fake_logits.squeeze(-1))
            score = 0.35 * coarse_error + 0.45 * refined_error + 0.1 * refinement_gap + 0.08 * feature_error + 0.02 * adv_score
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
    if name == "vaegan":
        return _train_vaegan(features, config)
    if name == "cascade_gan":
        return _train_cascade_gan(features, config)
    if name == "transformer":
        model = ElementTransformer(features.shape[1])
        return _train_reconstruction_model(model, features, config)
    raise ValueError(f"Unsupported baseline: {name}")
