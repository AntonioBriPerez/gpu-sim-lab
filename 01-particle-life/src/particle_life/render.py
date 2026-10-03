# pyright: reportInvalidTypeForm=false, reportArgumentType=false, reportOptionalCall=false
"""Additive glow renderer: Gaussian splats into an accumulation buffer.

One pass gives glow (overlapping particles brighten), trails (the buffer decays
instead of being cleared), per-type highlighting and zoom/pan (camera transform).
Call ``taichi.init(...)`` first. Annotations must stay real objects for Taichi, so no
``from __future__ import annotations`` here.
"""

import math
from typing import Any

import numpy as np
import taichi as ti

from .camera import Camera
from .colors import type_colors

DIM_GAIN = 0.12  # brightness of non-highlighted types


@ti.data_oriented
class GlowRenderer:
    def __init__(self, size: int, n_types: int) -> None:
        self.size = size
        self.accum = ti.Vector.field(3, ti.f32, (size, size))
        self.image = ti.Vector.field(3, ti.f32, (size, size))
        self.type_color = ti.Vector.field(3, ti.f32, n_types)
        self.type_gain = ti.field(ti.f32, n_types)
        self.n_types = n_types
        self.type_color.from_numpy(type_colors(n_types))
        self.set_highlight(None)

    def set_highlight(self, type_id: int | None) -> None:
        gains = np.ones(self.n_types, dtype=np.float32)
        if type_id is not None:
            gains[:] = DIM_GAIN
            gains[type_id] = 1.0
        self.type_gain.from_numpy(gains)

    def draw(
        self,
        pos: Any,
        ptype: Any,
        cam: Camera,
        sigma: float,
        exposure: float,
        trail: float,
    ) -> None:
        """Render into ``self.image``. ``trail`` in [0, 1): 0 clears every frame."""
        sigma_eff = sigma * math.sqrt(cam.zoom)  # dots grow a little when zooming in
        rad = max(1, math.ceil(3.0 * sigma_eff))
        self.fade(trail)
        self.splat(pos, ptype, cam.cx, cam.cy, cam.zoom, sigma_eff, rad)
        self.tonemap(exposure)

    @ti.kernel
    def fade(self, decay: ti.f32):
        for i, j in self.accum:
            self.accum[i, j] *= decay

    @ti.kernel
    def splat(
        self,
        pos: ti.template(),
        ptype: ti.template(),
        cx: ti.f32,
        cy: ti.f32,
        zoom: ti.f32,
        sigma: ti.f32,
        rad: ti.i32,
    ):
        size = ti.cast(self.size, ti.f32)
        inv = 1.0 / (2.0 * sigma * sigma)
        for i in pos:
            d = pos[i] - ti.Vector([cx, cy])
            d -= ti.round(d)
            s = (0.5 + d * zoom) * size
            if -rad < s.x < size + rad and -rad < s.y < size + rad:
                t = ptype[i]
                col = self.type_color[t] * self.type_gain[t]
                ix = ti.cast(ti.floor(s.x), ti.i32)
                iy = ti.cast(ti.floor(s.y), ti.i32)
                for ox in range(-rad, rad + 1):
                    for oy in range(-rad, rad + 1):
                        px = ix + ox
                        py = iy + oy
                        if 0 <= px < self.size and 0 <= py < self.size:
                            dd = (px + 0.5 - s.x) ** 2 + (py + 0.5 - s.y) ** 2
                            self.accum[px, py] += col * ti.exp(-dd * inv)

    @ti.kernel
    def tonemap(self, exposure: ti.f32):
        for i, j in self.accum:
            c = self.accum[i, j] * exposure
            self.image[i, j] = ti.Vector(
                [1.0 - ti.exp(-c.x), 1.0 - ti.exp(-c.y), 1.0 - ti.exp(-c.z)]
            )
