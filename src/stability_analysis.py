"""Reusable statistical helpers for post-hoc robustness analysis."""
import numpy as np


def stratified_bootstrap_indices(labels, rng):
    """Sample with replacement inside each observed class."""
    labels = np.asarray(labels)
    sampled = []
    for label in np.unique(labels):
        positions = np.flatnonzero(labels == label)
        sampled.append(rng.choice(positions, size=len(positions), replace=True))
    indices = np.concatenate(sampled)
    rng.shuffle(indices)
    return indices


def percentile_interval(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    alpha = 1 - confidence
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return {
        "CI Low": float(low),
        "CI High": float(high),
        "CI Width": float(high - low),
        "Bootstrap SE": float(values.std(ddof=1)),
    }
