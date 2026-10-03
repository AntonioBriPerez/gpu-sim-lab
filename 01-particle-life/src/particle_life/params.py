"""Simulation parameters and preset (de)serialisation. Pure Python, no Taichi."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

R_MIN = 0.004  # smallest interaction radius the grid is sized for (world is the unit square)
R_MAX_LIMIT = 1.0 / 3.0  # grid needs >= 3 cells per side so periodic neighbours never repeat


def random_matrix(n_types: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=(n_types, n_types)).astype(np.float32)


def mutate_matrix(matrix: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian jitter of every entry, clipped to [-1, 1]: explore variations of a good matrix."""
    noisy = matrix + rng.normal(0.0, amount, size=matrix.shape)
    return np.clip(noisy, -1.0, 1.0).astype(np.float32)


@dataclass
class Params:
    n_particles: int = 30_000
    n_types: int = 6
    seed: int = 0
    r_max: float = 0.03  # interaction radius, fraction of the world size
    beta: float = 0.3  # normalised radius below which every pair repels
    force_scale: float = 10.0
    friction_half_life: float = 0.04  # seconds for velocity to halve
    dt: float = 0.02
    matrix: list[list[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.matrix:
            rng = np.random.default_rng(self.seed)
            self.matrix = random_matrix(self.n_types, rng).tolist()
        self.validate()

    def validate(self) -> None:
        if not R_MIN <= self.r_max <= R_MAX_LIMIT:
            raise ValueError(f"r_max must be in [{R_MIN}, {R_MAX_LIMIT:.3f}], got {self.r_max}")
        if not 0.0 < self.beta < 1.0:
            raise ValueError(f"beta must be in (0, 1), got {self.beta}")
        if self.n_types < 1 or self.n_particles < 1:
            raise ValueError("n_types and n_particles must be >= 1")
        m = np.asarray(self.matrix)
        if m.shape != (self.n_types, self.n_types):
            raise ValueError(f"matrix shape {m.shape} does not match n_types={self.n_types}")

    def matrix_array(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=np.float32)

    def friction_factor(self, dt: float | None = None) -> float:
        return float(0.5 ** ((self.dt if dt is None else dt) / self.friction_half_life))


def save_preset(params: Params, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(params), indent=2))


def load_preset(path: Path) -> Params:
    return Params(**json.loads(path.read_text()))
