import torch

from geochemad.models import GeoChemFormer, ModelConfig


def test_geochemformer_forward_shapes():
    model = GeoChemFormer(feature_dim=12, target_index=2, k_neighbors=8, config=ModelConfig(hidden_dim=32, num_heads=4, num_layers=2))
    coords = torch.randn(5, 2)
    neighbor_offsets = torch.randn(5, 8, 2)
    neighbor_features = torch.randn(5, 8, 12)
    features = torch.randn(5, 12)
    target_pred, recon, context = model(coords, neighbor_offsets, neighbor_features, features)
    assert target_pred.shape == (5,)
    assert recon.shape == (5, 12)
    assert context.shape == (5, 32)
