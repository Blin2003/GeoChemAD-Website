from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def interpolate_scores_at_locations(
    sample_scores: np.ndarray,
    sample_coords: np.ndarray,
    query_coords: np.ndarray,
    k_neighbors: int = 8,
) -> np.ndarray:
    k = min(k_neighbors, len(sample_scores))
    distances = np.linalg.norm(query_coords[:, None, :] - sample_coords[None, :, :], axis=2)
    nearest_idx = np.argpartition(distances, kth=max(k - 1, 0), axis=1)[:, :k]
    nearest_dist = np.take_along_axis(distances, nearest_idx, axis=1)
    nearest_scores = np.take_along_axis(sample_scores[None, :].repeat(len(query_coords), axis=0), nearest_idx, axis=1)
    nearest_dist[nearest_dist < 1e-8] = 1e-8
    weights = 1.0 / nearest_dist
    return np.sum(weights * nearest_scores, axis=1) / np.sum(weights, axis=1)


def repeated_auc(
    sample_scores: np.ndarray,
    sample_coords: np.ndarray,
    site_coords: np.ndarray,
    repeats: int,
    background_size: int,
    exclusion_radius: float,
    seed: int,
) -> dict[str, object]:
    if len(site_coords) == 0:
        raise ValueError("No site coordinates available for evaluation.")
    distances = np.linalg.norm(sample_coords[:, None, :] - site_coords[None, :, :], axis=2)
    positive_scores = interpolate_scores_at_locations(sample_scores, sample_coords, site_coords)
    min_site_distance = distances.min(axis=1)
    candidate_background = np.where(min_site_distance > exclusion_radius)[0]
    if len(candidate_background) == 0:
        nearest_sample_ids = np.unique(np.argmin(distances, axis=0))
        candidate_background = np.setdiff1d(np.arange(len(sample_scores)), nearest_sample_ids)
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    aps: list[float] = []
    curves: dict[str, np.ndarray] | None = None
    bg_size = min(background_size, len(candidate_background))
    for _ in range(repeats):
        bg = rng.choice(candidate_background, size=bg_size, replace=False)
        labels = np.concatenate([np.ones(len(positive_scores)), np.zeros(len(bg))])
        scores = np.concatenate([positive_scores, sample_scores[bg]])
        aucs.append(float(roc_auc_score(labels, scores)))
        aps.append(float(average_precision_score(labels, scores)))
        if curves is None:
            fpr, tpr, _ = roc_curve(labels, scores)
            precision, recall, _ = precision_recall_curve(labels, scores)
            curves = {
                "roc_fpr": fpr,
                "roc_tpr": tpr,
                "pr_precision": precision,
                "pr_recall": recall,
            }
    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "ap_mean": float(np.mean(aps)),
        "ap_std": float(np.std(aps)),
        "positive_count": int(len(positive_scores)),
        "background_count": int(bg_size),
        "curves": {
            key: value.tolist() for key, value in (curves or {}).items()
        },
    }


def spatial_targeting_metrics(
    sample_scores: np.ndarray,
    sample_coords: np.ndarray,
    site_coords: np.ndarray,
    top_fraction: float = 0.05,
) -> dict[str, float]:
    distances = np.linalg.norm(sample_coords[:, None, :] - site_coords[None, :, :], axis=2)
    nearest = distances.min(axis=1)
    cutoff = max(1, int(len(sample_scores) * top_fraction))
    top_idx = np.argsort(sample_scores)[::-1][:cutoff]
    all_mean = float(np.mean(nearest))
    top_mean = float(np.mean(nearest[top_idx]))
    radius = float(np.percentile(nearest, 25))
    hit_rate = float(np.mean(nearest[top_idx] <= radius))
    return {
        "dtd_all_mean": all_mean,
        "dtd_top_mean": top_mean,
        "hit_rate_at_q1": hit_rate,
        "top_fraction": float(top_fraction),
    }
