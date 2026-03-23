from __future__ import annotations

from typing import Any

import numpy as np
from pykrige.ok import OrdinaryKriging


def _grid_axes(coords: np.ndarray, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    pad_x = max((x_max - x_min) * 0.03, 1e-6)
    pad_y = max((y_max - y_min) * 0.03, 1e-6)
    grid_x = np.linspace(x_min - pad_x, x_max + pad_x, grid_size)
    grid_y = np.linspace(y_min - pad_y, y_max + pad_y, grid_size)
    return grid_x, grid_y


def idw_interpolation(coords: np.ndarray, values: np.ndarray, grid_size: int = 96, power: float = 2.0) -> dict[str, Any]:
    grid_x, grid_y = _grid_axes(coords, grid_size)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    dx = mesh_x[..., None] - coords[:, 0]
    dy = mesh_y[..., None] - coords[:, 1]
    distances = np.sqrt(dx * dx + dy * dy)
    distances[distances < 1e-10] = 1e-10
    weights = 1.0 / np.power(distances, power)
    grid = np.sum(weights * values[None, None, :], axis=2) / np.sum(weights, axis=2)
    return {"grid_x": grid_x.tolist(), "grid_y": grid_y.tolist(), "grid": grid.tolist(), "method": "idw"}


def kriging_interpolation(
    coords: np.ndarray,
    values: np.ndarray,
    grid_size: int = 96,
    sample_cap: int = 4000,
) -> dict[str, Any]:
    if len(coords) > sample_cap:
        choice = np.linspace(0, len(coords) - 1, sample_cap, dtype=int)
        coords = coords[choice]
        values = values[choice]
    grid_x, grid_y = _grid_axes(coords, grid_size)
    ok = OrdinaryKriging(
        coords[:, 0],
        coords[:, 1],
        values,
        variogram_model="gaussian",
        enable_plotting=False,
        verbose=False,
    )
    grid, _ = ok.execute("grid", grid_x, grid_y)
    return {"grid_x": grid_x.tolist(), "grid_y": grid_y.tolist(), "grid": np.asarray(grid).tolist(), "method": "kriging"}


def generate_anomaly_map(
    coords: np.ndarray,
    scores: np.ndarray,
    method: str = "idw",
    grid_size: int = 96,
) -> dict[str, Any]:
    method = method.lower()
    if method == "kriging":
        try:
            return kriging_interpolation(coords, scores, grid_size=grid_size)
        except Exception:
            return idw_interpolation(coords, scores, grid_size=grid_size)
    return idw_interpolation(coords, scores, grid_size=grid_size)
