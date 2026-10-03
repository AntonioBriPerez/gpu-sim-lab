"""Brute-force O(N^2) NumPy reference used to validate the grid-based GPU kernel."""

from __future__ import annotations

import numpy as np


def force_profile(r: np.ndarray, attraction: np.ndarray, beta: float) -> np.ndarray:
    """Classic Particle Life force vs normalised distance r in [0, inf)."""
    repel = r / beta - 1.0
    attract = attraction * (1.0 - np.abs(2.0 * r - 1.0 - beta) / (1.0 - beta))
    return np.where(r < beta, repel, np.where(r < 1.0, attract, 0.0))


def accelerations_bruteforce(
    pos: np.ndarray,
    ptype: np.ndarray,
    matrix: np.ndarray,
    r_max: float,
    beta: float,
    force_scale: float,
) -> np.ndarray:
    d = pos[None, :, :] - pos[:, None, :]  # d[i, j] = pos[j] - pos[i]
    d -= np.round(d)  # minimum image on the periodic unit square
    dist = np.linalg.norm(d, axis=-1)
    np.fill_diagonal(dist, np.inf)
    f = force_profile(dist / r_max, matrix[ptype[:, None], ptype[None, :]], beta)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = np.where(np.isfinite(dist)[..., None], d / dist[..., None], 0.0)
    return (f[..., None] * unit).sum(axis=1) * r_max * force_scale
