from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1
    mlp_dim: int = 256


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GeoChemFormer(nn.Module):
    def __init__(self, feature_dim: int, commodity_count: int, k_neighbors: int, config: ModelConfig):
        super().__init__()
        self.feature_dim = feature_dim
        self.k_neighbors = k_neighbors
        self.config = config
        self.target_embedding = nn.Embedding(commodity_count, config.hidden_dim)
        self.query_proj = MLP(2, config.hidden_dim, config.hidden_dim)
        self.neighbor_proj = MLP(feature_dim + 2, config.hidden_dim, config.hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_dim,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.context_encoder = nn.TransformerEncoder(enc_layer, num_layers=config.num_layers)
        self.context_predictor = nn.Linear(config.hidden_dim, 1)
        self.context_to_geo = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.element_value_proj = nn.Linear(1, config.hidden_dim)
        self.element_embedding = nn.Embedding(feature_dim, config.hidden_dim)
        self.dependency_encoder = nn.TransformerEncoder(enc_layer, num_layers=config.num_layers)
        self.reconstruction_head = nn.Linear(config.hidden_dim, 1)

    def encode_context(
        self,
        coords: torch.Tensor,
        commodity_ids: torch.Tensor,
        neighbor_offsets: torch.Tensor,
        neighbor_features: torch.Tensor,
    ) -> torch.Tensor:
        target_token = self.target_embedding(commodity_ids).unsqueeze(1)
        query_token = self.query_proj(coords).unsqueeze(1)
        neighbor_token = self.neighbor_proj(torch.cat([neighbor_offsets, neighbor_features], dim=-1))
        tokens = torch.cat([target_token, query_token, neighbor_token], dim=1)
        encoded = self.context_encoder(tokens)
        return encoded[:, 1]

    def predict_target(
        self,
        coords: torch.Tensor,
        commodity_ids: torch.Tensor,
        neighbor_offsets: torch.Tensor,
        neighbor_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.encode_context(coords, commodity_ids, neighbor_offsets, neighbor_features)
        prediction = self.context_predictor(context).squeeze(-1)
        return prediction, context

    def reconstruct_from_context(self, context: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        batch = features.shape[0]
        geo_token = self.context_to_geo(context).unsqueeze(1)
        elem_ids = torch.arange(self.feature_dim, device=features.device).unsqueeze(0).expand(batch, -1)
        elem_tokens = self.element_embedding(elem_ids) + self.element_value_proj(features.unsqueeze(-1))
        encoded = self.dependency_encoder(torch.cat([geo_token, elem_tokens], dim=1))
        recon = self.reconstruction_head(encoded[:, 1:]).squeeze(-1)
        return recon

    def forward(
        self,
        coords: torch.Tensor,
        commodity_ids: torch.Tensor,
        neighbor_offsets: torch.Tensor,
        neighbor_features: torch.Tensor,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.encode_context(coords, commodity_ids, neighbor_offsets, neighbor_features)
        target_pred = self.context_predictor(context).squeeze(-1)
        recon = self.reconstruct_from_context(context, features)
        return target_pred, recon, context


class AutoEncoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 128, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class VariationalAutoEncoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 128, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        mu = self.mu(hidden)
        logvar = self.logvar(hidden)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


class TabularDiscriminator(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_net(x)
        logits = self.head(features)
        return logits, features


class CascadeGenerator(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 128, latent_dim: int = 32):
        super().__init__()
        self.stage_one = AutoEncoder(feature_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.stage_two = AutoEncoder(feature_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coarse = self.stage_one(x)
        refined = self.stage_two(coarse)
        return coarse, refined


class ElementTransformer(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 128, num_heads: int = 4, num_layers: int = 3):
        super().__init__()
        self.feature_dim = feature_dim
        self.value_proj = nn.Linear(1, hidden_dim)
        self.embedding = nn.Embedding(feature_dim, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        elem_ids = torch.arange(self.feature_dim, device=x.device).unsqueeze(0).expand(batch, -1)
        tokens = self.embedding(elem_ids) + self.value_proj(x.unsqueeze(-1))
        encoded = self.encoder(tokens)
        return self.head(encoded).squeeze(-1)
