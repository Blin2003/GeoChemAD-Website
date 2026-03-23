from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .features import FeatureSelectionConfig, select_features


META_COLUMNS = {
    "SAMPLEID",
    "SAMPLETYPE",
    "WAMEX_A_NO",
    "COMPSAMPID",
    "OBJECTID_x",
    "OBJECTID_y",
    "SITE_CODE",
    "SITE_TITLE",
    "SHORT_NAME",
    "SITE_COMMO",
    "SITE_TYPE_",
    "SITE_SUB_T",
    "SITE_STAGE",
    "TARGET_COM",
    "COMMODITY_",
    "PROJ_CODE",
    "PROJ_TITLE",
    "WEB_LINK",
    "EXTRACT_DA",
    "Au_SITES",
}


@dataclass
class PreprocessingConfig:
    missing_value_floor: float = -999.0
    replacement_strategy: Literal["half_min_positive"] = "half_min_positive"
    compositional_transform: Literal["none", "clr", "ilr"] = "ilr"
    scale: bool = True
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)


@dataclass
class PreparedFrame:
    coords: np.ndarray
    coords_normalized: np.ndarray
    features: np.ndarray
    feature_names: list[str]
    target_index: int
    target_values: np.ndarray
    sample_ids: list[str]
    raw_frame: pd.DataFrame
    feature_diagnostics: dict[str, object]


def _coerce_coords(frame: pd.DataFrame) -> np.ndarray:
    if {"X", "Y"}.issubset(frame.columns):
        return frame[["X", "Y"]].astype(float).to_numpy(dtype=np.float32)
    raise ValueError("Expected X/Y coordinate columns in input frame.")


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    numeric_cols: list[str] = []
    for column in frame.columns:
        if column in META_COLUMNS or column in {"X", "Y"}:
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.notna().any():
            numeric_cols.append(column)
    return numeric_cols


def _replace_abnormal_values(values: np.ndarray, floor: float) -> np.ndarray:
    arr = values.astype(np.float64, copy=True)
    invalid_mask = ~np.isfinite(arr) | (arr <= 0.0) | (arr <= floor)
    for idx in range(arr.shape[1]):
        col = arr[:, idx]
        valid = col[~invalid_mask[:, idx]]
        substitute = 1e-6
        if valid.size:
            substitute = max(float(np.nanmin(valid)) * 0.5, 1e-6)
        col[invalid_mask[:, idx]] = substitute
        arr[:, idx] = col
    return arr


def _closure(values: np.ndarray) -> np.ndarray:
    denom = values.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return values / denom


def clr_transform(values: np.ndarray) -> np.ndarray:
    closed = _closure(values)
    logged = np.log(closed)
    return logged - logged.mean(axis=1, keepdims=True)


def _helmert_basis(dim: int) -> np.ndarray:
    basis = np.zeros((dim, dim - 1), dtype=np.float64)
    for j in range(1, dim):
        basis[:j, j - 1] = 1.0 / np.sqrt(j * (j + 1))
        basis[j, j - 1] = -j / np.sqrt(j * (j + 1))
    return basis


def ilr_transform(values: np.ndarray) -> np.ndarray:
    clr = clr_transform(values)
    basis = _helmert_basis(values.shape[1])
    return clr @ basis


def zscore_scale(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (values - mean) / std


def zscore_vector(values: np.ndarray) -> np.ndarray:
    mean = values.mean()
    std = values.std()
    if std < 1e-8:
        std = 1.0
    return (values - mean) / std


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    mean = coords.mean(axis=0, keepdims=True)
    std = coords.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (coords - mean) / std


def prepare_samples(
    frame: pd.DataFrame,
    target_element: str,
    config: PreprocessingConfig,
    preserve_element_identity: bool = False,
) -> PreparedFrame:
    coords = _coerce_coords(frame)
    feature_names = _feature_columns(frame)
    if not feature_names:
        raise ValueError("No numeric geochemical feature columns were detected.")
    numeric = frame[feature_names].apply(pd.to_numeric, errors="coerce").to_numpy()
    clean = _replace_abnormal_values(numeric, config.missing_value_floor)
    target_candidates = [
        idx for idx, name in enumerate(feature_names) if name.lower().startswith(target_element.lower())
    ]
    if not target_candidates:
        raise ValueError(f"Target element {target_element} not found in feature columns.")
    original_target_index = target_candidates[0]
    target_values = np.log1p(clean[:, original_target_index])
    if config.scale:
        target_values = zscore_vector(target_values)
    selected, selected_names, target_index, feature_diagnostics = select_features(
        values=clean,
        feature_names=feature_names,
        target_element=target_element,
        target_index=original_target_index,
        target_values=target_values,
        config=config.feature_selection,
    )
    transformed = selected
    transformed_names = selected_names
    applied_transform = config.compositional_transform
    if preserve_element_identity and config.compositional_transform == "ilr":
        applied_transform = "clr"
        feature_diagnostics["transform_adjustment"] = (
            "GeoChemFormer requires element-identity-preserving tokens; using CLR instead of ILR."
        )
    if applied_transform == "clr":
        transformed = clr_transform(selected)
    elif applied_transform == "ilr":
        transformed = ilr_transform(selected)
        transformed_names = [f"ILR_{idx:03d}" for idx in range(transformed.shape[1])]
        target_index = min(target_index, transformed.shape[1] - 1)
    elif applied_transform == "none":
        transformed = selected
        transformed_names = selected_names
    if config.scale:
        transformed = zscore_scale(transformed)
    sample_ids = (
        frame["SAMPLEID"].astype(str).tolist() if "SAMPLEID" in frame.columns else [str(i) for i in range(len(frame))]
    )
    return PreparedFrame(
        coords=coords,
        coords_normalized=normalize_coords(coords).astype(np.float32),
        features=transformed.astype(np.float32),
        feature_names=transformed_names,
        target_index=target_index,
        target_values=target_values.astype(np.float32),
        sample_ids=sample_ids,
        raw_frame=frame,
        feature_diagnostics=feature_diagnostics,
    )
