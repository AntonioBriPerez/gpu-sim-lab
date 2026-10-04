"""Orbit camera over the periodic unit cube. Pure NumPy (no wgpu).

Conventions: right-handed, y up. The camera always looks at its target, which is
the centre of the rendered 'view cell' (positions are wrapped to target +- 0.5), so
in view-cell coordinates the target is the origin. WebGPU clip space: z in [0, 1].
Matrices are row-major NumPy (M @ v); upload M.T so WGSL (column-major) sees M.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

DIST_MIN, DIST_MAX = 0.05, 6.0
PITCH_LIMIT = math.radians(89.0)


def perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(fov_y / 2.0)
    return np.array(
        [
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, far / (near - far), near * far / (near - far)],
            [0, 0, -1, 0],
        ],
        dtype=np.float64,
    )


def look_at(eye: np.ndarray, right: np.ndarray, up: np.ndarray, fwd: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = right, up, -fwd
    m[:3, 3] = -m[:3, :3] @ eye
    return m


@dataclass
class OrbitCamera:
    yaw: float = math.radians(35.0)
    pitch: float = math.radians(25.0)
    dist: float = 2.0
    target: np.ndarray = field(default_factory=lambda: np.full(3, 0.5))
    fov_y: float = math.radians(50.0)

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(right, up, fwd) unit vectors; fwd points from the eye to the target."""
        cp = math.cos(self.pitch)
        to_eye = np.array([cp * math.sin(self.yaw), math.sin(self.pitch), cp * math.cos(self.yaw)])
        fwd = -to_eye
        right = np.cross(fwd, [0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        return right, up, fwd

    def eye(self) -> np.ndarray:
        """Eye position in view-cell coordinates (target at the origin)."""
        return -self.basis()[2] * self.dist

    def near_far(self) -> tuple[float, float]:
        return 0.005, self.dist + 2.0

    def view_proj(self, aspect: float) -> np.ndarray:
        right, up, fwd = self.basis()
        near, far = self.near_far()
        return perspective(self.fov_y, aspect, near, far) @ look_at(self.eye(), right, up, fwd)

    def orbit(self, d_yaw: float, d_pitch: float) -> None:
        self.yaw = (self.yaw + d_yaw) % (2.0 * math.pi)
        self.pitch = min(PITCH_LIMIT, max(-PITCH_LIMIT, self.pitch + d_pitch))

    def zoom_by(self, factor: float) -> None:
        self.dist = min(DIST_MAX, max(DIST_MIN, self.dist * factor))

    def pan(self, dx: float, dy: float) -> None:
        """Move the target in the screen plane; dx, dy in world units. Wraps."""
        right, up, _ = self.basis()
        self.target = (self.target + right * dx + up * dy) % 1.0

    def ray(self, sx: float, sy: float, aspect: float) -> tuple[np.ndarray, np.ndarray]:
        """Ray through screen point (sx, sy) in [0,1]^2 (y down, like GLFW), view-cell coords."""
        right, up, fwd = self.basis()
        t = math.tan(self.fov_y / 2.0)
        x = (2.0 * sx - 1.0) * t * aspect
        y = (1.0 - 2.0 * sy) * t
        d = fwd + right * x + up * y
        return self.eye(), d / np.linalg.norm(d)

    def focal_point_rel(self, sx: float, sy: float, aspect: float) -> np.ndarray | None:
        """Where the cursor ray meets the plane through the target, in view-cell coords.

        None when that point is outside the drawn cube (|coord| > 0.5): there the
        periodic wrap would act on particles drawn somewhere else on screen.
        """
        origin, d = self.ray(sx, sy, aspect)
        fwd = self.basis()[2]
        rel = origin + d * (self.dist / float(d @ fwd))
        return rel if np.abs(rel).max() <= 0.5 else None

    def focal_point(self, sx: float, sy: float, aspect: float) -> np.ndarray | None:
        """World position (in [0,1)^3) under the cursor on the focal plane, or None."""
        rel = self.focal_point_rel(sx, sy, aspect)
        return None if rel is None else (self.target + rel) % 1.0

    def project(self, world: np.ndarray, width: int, height: int) -> np.ndarray:
        """World points -> pixel coordinates (x right, y down). For tests and overlays."""
        rel = world - self.target
        rel -= np.round(rel)
        h = np.c_[rel, np.ones(len(rel))] @ self.view_proj(width / height).T
        ndc = h[:, :2] / h[:, 3:4]
        return np.c_[(ndc[:, 0] + 1) * 0.5 * width, (1 - ndc[:, 1]) * 0.5 * height]
