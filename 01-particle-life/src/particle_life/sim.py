# pyright: reportInvalidTypeForm=false, reportArgumentType=false, reportOptionalCall=false
# (Taichi kernels use dynamic annotations/types that static checkers cannot model.)
"""Particle Life simulation on a uniform spatial grid (Taichi).

The caller must run ``taichi.init(...)`` before creating a ``ParticleLife``.
The world is the periodic unit square. (No ``from __future__ import annotations``
here: Taichi needs real annotation objects.) Neighbour search uses a counting-sort grid
whose cell size is >= r_max, so each particle only visits its 3x3 neighbourhood.
"""

import numpy as np
import taichi as ti

from .params import R_MIN, Params

MAX_CELLS_PER_SIDE = int(1.0 / R_MIN)


@ti.func
def _force(r: ti.f32, a: ti.f32, beta: ti.f32) -> ti.f32:
    f = 0.0
    if r < beta:
        f = r / beta - 1.0
    elif r < 1.0:
        f = a * (1.0 - ti.abs(2.0 * r - 1.0 - beta) / (1.0 - beta))
    return f


@ti.data_oriented
class ParticleLife:
    def __init__(self, params: Params) -> None:
        self.params = params
        n, t = params.n_particles, params.n_types
        self.pos = ti.Vector.field(2, ti.f32, n)
        self.vel = ti.Vector.field(2, ti.f32, n)
        self.acc = ti.Vector.field(2, ti.f32, n)
        self.ptype = ti.field(ti.i32, n)
        self.matrix = ti.field(ti.f32, (t, t))
        max_cells = MAX_CELLS_PER_SIDE**2
        self.cell_count = ti.field(ti.i32, max_cells)
        self.cell_start = ti.field(ti.i32, max_cells + 1)
        self.cell_cursor = ti.field(ti.i32, max_cells)
        self.sorted_idx = ti.field(ti.i32, n)
        self.reset(params.seed)
        self.set_matrix(params.matrix_array())

    # ---- host-side helpers -------------------------------------------------
    def reset(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        n = self.params.n_particles
        self.pos.from_numpy(rng.random((n, 2), dtype=np.float32))
        self.vel.fill(0)
        self.ptype.from_numpy(rng.integers(0, self.params.n_types, n, dtype=np.int32))

    def set_matrix(self, matrix: np.ndarray) -> None:
        self.params.matrix = matrix.tolist()
        self.matrix.from_numpy(matrix.astype(np.float32))

    def grid_cells_per_side(self) -> int:
        return max(3, int(1.0 / self.params.r_max))

    def step(self) -> None:
        p = self.params
        nc = self.grid_cells_per_side()
        self.build_grid(nc)
        self.compute_acc(nc, p.r_max, p.beta, p.force_scale)
        self.integrate(p.dt, p.friction_factor())

    # ---- kernels -----------------------------------------------------------
    @ti.func
    def _cell(self, p: ti.template(), nc: ti.i32):
        c = ti.min(ti.cast(p * nc, ti.i32), nc - 1)
        return c

    @ti.kernel
    def build_grid(self, nc: ti.i32):
        n_cells = nc * nc
        for c in range(n_cells):
            self.cell_count[c] = 0
        for i in self.pos:
            c = self._cell(self.pos[i], nc)
            ti.atomic_add(self.cell_count[c.x * nc + c.y], 1)
        for _ in range(1):  # single thread: exclusive prefix sum over the (few) cells
            s = 0
            for c in range(n_cells):
                self.cell_start[c] = s
                self.cell_cursor[c] = s
                s += self.cell_count[c]
            self.cell_start[n_cells] = s
        for i in self.pos:
            c = self._cell(self.pos[i], nc)
            slot = ti.atomic_add(self.cell_cursor[c.x * nc + c.y], 1)
            self.sorted_idx[slot] = i

    @ti.kernel
    def compute_acc(self, nc: ti.i32, r_max: ti.f32, beta: ti.f32, force_scale: ti.f32):
        for i in self.pos:
            p = self.pos[i]
            ti_type = self.ptype[i]
            c = self._cell(p, nc)
            a = ti.Vector([0.0, 0.0])
            for dx in ti.static(range(-1, 2)):
                for dy in ti.static(range(-1, 2)):
                    cx = (c.x + dx + nc) % nc
                    cy = (c.y + dy + nc) % nc
                    cell = cx * nc + cy
                    for k in range(self.cell_start[cell], self.cell_start[cell + 1]):
                        j = self.sorted_idx[k]
                        if j != i:
                            d = self.pos[j] - p
                            d -= ti.round(d)
                            dist = d.norm()
                            if 0.0 < dist < r_max:
                                f = _force(dist / r_max, self.matrix[ti_type, self.ptype[j]], beta)
                                a += d / dist * f
            self.acc[i] = a * r_max * force_scale

    @ti.kernel
    def integrate(self, dt: ti.f32, friction: ti.f32):
        for i in self.pos:
            self.vel[i] = self.vel[i] * friction + self.acc[i] * dt
            p = self.pos[i] + self.vel[i] * dt
            self.pos[i] = p - ti.floor(p)
