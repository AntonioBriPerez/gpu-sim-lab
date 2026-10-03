"""Particle type colours shared by the renderer and the control panel."""

import colorsys

import numpy as np


def type_colors(n_types: int) -> np.ndarray:
    return np.array(
        [colorsys.hsv_to_rgb(i / n_types, 0.85, 1.0) for i in range(n_types)], dtype=np.float32
    )


def type_hex(n_types: int) -> list[str]:
    return [
        f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
        for r, g, b in type_colors(n_types)
    ]
