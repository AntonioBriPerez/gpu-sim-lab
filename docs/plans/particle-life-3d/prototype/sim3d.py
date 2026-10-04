"""Prototype of the planned ParticleLife3D (wgpu compute)."""

from __future__ import annotations

import math

import numpy as np
import wgpu
from gpu import bind, compute_pipeline, load_shader

WG = 256
BLOCK = 1024
NC_MAX = 100  # 100^3 = 1e6 cells <= BLOCK^2, so the two-level scan always suffices
MAX_CELLS = NC_MAX**3
MAX_TYPES = 16

U = wgpu.BufferUsage
STORAGE = U.STORAGE | U.COPY_SRC | U.COPY_DST


def pack_sim_params(n, nc, n_types, r_max, beta, force_scale, dt, friction) -> bytes:
    ints = np.array([n, nc, n_types, 0], dtype=np.uint32)
    floats = np.array([r_max, beta, force_scale, dt, friction, 0, 0, 0], dtype=np.float32)
    return ints.tobytes() + floats.tobytes()  # 48 bytes


def pack_brush_params(center, radius, strength, dt, n) -> bytes:
    f = np.array([*center, radius, strength, dt], dtype=np.float32)
    i = np.array([n, 0], dtype=np.uint32)
    return f.tobytes() + i.tobytes()  # 32 bytes


def pack_scan_params(n) -> bytes:
    return np.array([n, 0, 0, 0], dtype=np.uint32).tobytes()


def groups(n: int, size: int = WG) -> int:
    return (n + size - 1) // size


class Sim:
    def __init__(
        self,
        device,
        n,
        n_types,
        r_max,
        beta=0.3,
        force_scale=10.0,
        dt=0.02,
        friction_half_life=0.04,
        seed=0,
    ):
        self.device = d = device
        self.n, self.n_types = n, n_types
        self.r_max, self.beta, self.force_scale = r_max, beta, force_scale
        self.dt, self.half_life = dt, friction_half_life
        mk = lambda size, label: d.create_buffer(label=label, size=size, usage=STORAGE)
        self.pos_a, self.pos_b = mk(16 * n, "pos_a"), mk(16 * n, "pos_b")
        self.vel_a, self.vel_b = mk(16 * n, "vel_a"), mk(16 * n, "vel_b")
        self.cell_of, self.rank = mk(4 * n, "cell_of"), mk(4 * n, "rank")
        self.cell_count = mk(4 * MAX_CELLS, "cell_count")
        self.cell_start = mk(4 * MAX_CELLS, "cell_start")
        self.block_sums = mk(4 * BLOCK, "block_sums")
        self.block_offsets = mk(4 * BLOCK, "block_offsets")
        self.scan_total = mk(16, "scan_total")
        self.matrix = mk(4 * MAX_TYPES * MAX_TYPES, "matrix")
        uni = U.UNIFORM | U.COPY_DST
        self.u_sim = d.create_buffer(label="u_sim", size=48, usage=uni)
        self.u_scan_cells = d.create_buffer(label="u_scan_cells", size=16, usage=uni)
        self.u_scan_blocks = d.create_buffer(label="u_scan_blocks", size=16, usage=uni)
        self.u_brush = d.create_buffer(label="u_brush", size=32, usage=uni)

        grid = load_shader(d, "common", "grid")
        scatter = load_shader(d, "common", "scatter")
        forces = load_shader(d, "common", "forces")
        scan = load_shader(d, "scan")
        brush = load_shader(d, "brush")
        self.p_count, l_count = compute_pipeline(
            d, grid, "count", {0: "uniform", 1: "read", 2: "rw", 3: "rw", 4: "rw"}
        )
        self.p_scatter, l_scatter = compute_pipeline(
            d,
            scatter,
            "scatter",
            {0: "uniform", 1: "read", 2: "read", 3: "read", 4: "read", 5: "read", 6: "rw", 7: "rw"},
        )
        self.p_scan, l_scan = compute_pipeline(
            d, scan, "scan_blocks", {0: "uniform", 1: "read", 2: "rw", 3: "rw"}
        )
        self.p_add, l_add = compute_pipeline(
            d, scan, "add_offsets", {0: "uniform", 2: "rw", 3: "rw"}
        )
        self.p_step, l_step = compute_pipeline(
            d,
            forces,
            "step",
            {0: "uniform", 1: "read", 2: "read", 3: "read", 4: "read", 5: "read", 6: "rw", 7: "rw"},
        )
        self.p_brush, l_brush = compute_pipeline(
            d, brush, "brush", {0: "uniform", 1: "read", 2: "rw"}
        )

        self.g_count = bind(
            d,
            l_count,
            {0: self.u_sim, 1: self.pos_a, 2: self.cell_count, 3: self.cell_of, 4: self.rank},
        )
        self.g_scan1 = bind(
            d,
            l_scan,
            {0: self.u_scan_cells, 1: self.cell_count, 2: self.cell_start, 3: self.block_sums},
        )
        self.g_scan2 = bind(
            d,
            l_scan,
            {0: self.u_scan_blocks, 1: self.block_sums, 2: self.block_offsets, 3: self.scan_total},
        )
        self.g_add = bind(
            d, l_add, {0: self.u_scan_cells, 2: self.cell_start, 3: self.block_offsets}
        )
        self.g_scatter = bind(
            d,
            l_scatter,
            {
                0: self.u_sim,
                1: self.pos_a,
                2: self.vel_a,
                3: self.cell_of,
                4: self.rank,
                5: self.cell_start,
                6: self.pos_b,
                7: self.vel_b,
            },
        )
        self.g_step = bind(
            d,
            l_step,
            {
                0: self.u_sim,
                1: self.pos_b,
                2: self.vel_b,
                3: self.cell_start,
                4: self.cell_count,
                5: self.matrix,
                6: self.pos_a,
                7: self.vel_a,
            },
        )
        self.g_brush = bind(d, l_brush, {0: self.u_brush, 1: self.pos_a, 2: self.vel_a})
        self.reset(seed)
        rng = np.random.default_rng(seed)
        self.set_matrix(rng.uniform(-1, 1, (n_types, n_types)).astype(np.float32))
        self._written_nc = None

    def reset(self, seed):
        rng = np.random.default_rng(seed)
        pos = np.zeros((self.n, 4), np.float32)
        pos[:, :3] = rng.random((self.n, 3), dtype=np.float32)
        pos[:, 3] = rng.integers(0, self.n_types, self.n)
        self.write_state(pos, np.zeros((self.n, 4), np.float32))

    def write_state(self, pos4, vel4):
        q = self.device.queue
        q.write_buffer(self.pos_a, 0, np.ascontiguousarray(pos4, np.float32))
        q.write_buffer(self.vel_a, 0, np.ascontiguousarray(vel4, np.float32))

    def set_matrix(self, m):
        self.matrix_np = m.astype(np.float32)
        self.device.queue.write_buffer(self.matrix, 0, self.matrix_np.ravel())

    def nc(self):
        return min(NC_MAX, max(3, int(1.0 / self.r_max)))

    def _write_uniforms(self, dt_scale):
        nc = self.nc()
        dt = self.dt * dt_scale
        friction = 0.5 ** (dt / self.half_life)
        q = self.device.queue
        q.write_buffer(
            self.u_sim,
            0,
            pack_sim_params(
                self.n, nc, self.n_types, self.r_max, self.beta, self.force_scale, dt, friction
            ),
        )
        n_cells = nc**3
        q.write_buffer(self.u_scan_cells, 0, pack_scan_params(n_cells))
        q.write_buffer(self.u_scan_blocks, 0, pack_scan_params(groups(n_cells, BLOCK)))
        return n_cells

    def encode_grid(self, enc, n_cells, timestamps=None):
        enc.clear_buffer(self.cell_count, 0, 4 * n_cells)
        cp = enc.begin_compute_pass(label="grid", timestamp_writes=timestamps)
        cp.set_pipeline(self.p_count)
        cp.set_bind_group(0, self.g_count)
        cp.dispatch_workgroups(groups(self.n))
        n_blocks = groups(n_cells, BLOCK)
        cp.set_pipeline(self.p_scan)
        cp.set_bind_group(0, self.g_scan1)
        cp.dispatch_workgroups(n_blocks)
        cp.set_bind_group(0, self.g_scan2)
        cp.dispatch_workgroups(1)
        cp.set_pipeline(self.p_add)
        cp.set_bind_group(0, self.g_add)
        cp.dispatch_workgroups(groups(n_cells))
        cp.set_pipeline(self.p_scatter)
        cp.set_bind_group(0, self.g_scatter)
        cp.dispatch_workgroups(groups(self.n))
        cp.end()

    def encode_forces(self, enc, timestamps=None):
        cp = enc.begin_compute_pass(label="forces", timestamp_writes=timestamps)
        cp.set_pipeline(self.p_step)
        cp.set_bind_group(0, self.g_step)
        cp.dispatch_workgroups(groups(self.n))
        cp.end()

    def step(self, steps=1, dt_scale=1.0):
        n_cells = self._write_uniforms(dt_scale)
        enc = self.device.create_command_encoder()
        for _ in range(steps):
            self.encode_grid(enc, n_cells)
            self.encode_forces(enc)
        self.device.queue.submit([enc.finish()])

    def brush(self, center, radius, strength, dt_scale=1.0):
        self.device.queue.write_buffer(
            self.u_brush, 0, pack_brush_params(center, radius, strength, self.dt * dt_scale, self.n)
        )
        enc = self.device.create_command_encoder()
        cp = enc.begin_compute_pass(label="brush")
        cp.set_pipeline(self.p_brush)
        cp.set_bind_group(0, self.g_brush)
        cp.dispatch_workgroups(groups(self.n))
        cp.end()
        self.device.queue.submit([enc.finish()])

    def read(self, buf, dtype, shape):
        return np.frombuffer(self.device.queue.read_buffer(buf), dtype=dtype).reshape(shape)

    def state(self):
        return self.read(self.pos_a, np.float32, (self.n, 4)), self.read(
            self.vel_a, np.float32, (self.n, 4)
        )


def neighbours(n, r):
    return n * 4.0 / 3.0 * math.pi * r**3
