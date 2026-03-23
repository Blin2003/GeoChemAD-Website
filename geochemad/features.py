from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression


@dataclass
class FeatureSelectionConfig:
    strategy: str = "none"
    top_k: int = 24
    pca_components: int = 24


TARGET_PRIORS = {
    "AU": ["Ag", "As", "Bi", "Cu", "Mo", "Pb", "Sb", "Te", "W", "Zn", "S"],
    "CU": ["Ag", "Au", "Co", "Fe", "Mn", "Mo", "Ni", "Pb", "S", "Zn"],
    "NI": ["Co", "Cr", "Cu", "Fe", "Mg", "Mn", "S"],
    "W": ["As", "Bi", "Cu", "Fe", "Mo", "Sn"],
    "LI": ["B", "Be", "Cs", "K", "Na", "Rb", "Ta", "Sn"],
}


def _prefix(name: str) -> str:
    return name.split("_", 1)[0].upper()


def _match_columns(feature_names: list[str], tokens: list[str]) -> list[int]:
    wanted = {token.upper() for token in tokens}
    indices = [idx for idx, name in enumerate(feature_names) if _prefix(name) in wanted]
    return sorted(set(indices))


def _ensure_target(indices: list[int], target_index: int) -> list[int]:
    if target_index not in indices:
        indices = [target_index] + indices
    return sorted(set(indices))


def select_features(
    values: np.ndarray,
    feature_names: list[str],
    target_element: str,
    target_index: int,
    target_values: np.ndarray,
    config: FeatureSelectionConfig,
) -> tuple[np.ndarray, list[str], int, dict[str, object]]:
    strategy = config.strategy.lower()
    diagnostics: dict[str, object] = {"feature_strategy": strategy}
    if strategy == "none":
        return values, feature_names, target_index, diagnostics
    if strategy == "manual":
        indices = _ensure_target(_match_columns(feature_names, TARGET_PRIORS.get(target_element.upper(), [])), target_index)
        selected = values[:, indices]
        names = [feature_names[idx] for idx in indices]
        diagnostics["selected_features"] = names
        return selected, names, names.index(feature_names[target_index]), diagnostics
    if strategy == "causal":
        scores = mutual_info_regression(values, target_values, random_state=42)
        order = np.argsort(scores)[::-1]
        picked = order[: max(2, min(config.top_k, len(order)))].tolist()
        picked = _ensure_target(picked, target_index)
        picked = picked[: max(2, min(config.top_k, len(picked)))]
        names = [feature_names[idx] for idx in picked]
        diagnostics["selected_features"] = names
        diagnostics["feature_scores"] = {feature_names[idx]: float(scores[idx]) for idx in picked}
        return values[:, picked], names, names.index(feature_names[target_index]), diagnostics
    if strategy == "llm":
        tokens = [target_element.upper()] + TARGET_PRIORS.get(target_element.upper(), [])
        indices = _ensure_target(_match_columns(feature_names, tokens), target_index)
        if len(indices) < min(config.top_k, len(feature_names)):
            scores = mutual_info_regression(values, target_values, random_state=42)
            extras = [idx for idx in np.argsort(scores)[::-1].tolist() if idx not in indices]
            indices.extend(extras[: max(0, config.top_k - len(indices))])
        indices = indices[: max(2, min(config.top_k, len(indices)))]
        names = [feature_names[idx] for idx in indices]
        diagnostics["selected_features"] = names
        diagnostics["selector_note"] = "LLM-style geology prior plus mutual-information fallback."
        return values[:, indices], names, names.index(feature_names[target_index]), diagnostics
    if strategy == "pca":
        components = max(2, min(config.pca_components, values.shape[0] - 1, values.shape[1]))
        pca = PCA(n_components=components, random_state=42)
        projected = pca.fit_transform(values)
        names = [f"PCA_{idx + 1:02d}" for idx in range(projected.shape[1])]
        diagnostics["explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
        diagnostics["selected_features"] = names
        return projected.astype(np.float64), names, 0, diagnostics
    raise ValueError(f"Unsupported feature-selection strategy: {config.strategy}")
