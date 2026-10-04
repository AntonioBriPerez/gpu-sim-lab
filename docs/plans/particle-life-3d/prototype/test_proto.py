import numpy as np
import pytest
import wgpu
from gpu import bind, compute_pipeline, create_device, load_shader
from sim3d import BLOCK, STORAGE, Sim, groups, pack_scan_params


def force_profile(r, a, beta):
    repel = r / beta - 1.0
    attract = a * (1.0 - np.abs(2.0 * r - 1.0 - beta) / (1.0 - beta))
    return np.where(r < beta, repel, np.where(r < 1.0, attract, 0.0))


def acc_bruteforce(pos, ptype, matrix, r_max, beta, force_scale):
    d = pos[None, :, :] - pos[:, None, :]
    d -= np.round(d)
    dist = np.linalg.norm(d, axis=-1)
    np.fill_diagonal(dist, np.inf)
    f = force_profile(dist / r_max, matrix[ptype[:, None], ptype[None, :]], beta)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = np.where(np.isfinite(dist)[..., None], d / dist[..., None], 0.0)
    return (f[..., None] * unit).sum(axis=1) * r_max * force_scale


@pytest.fixture(scope="session")
def device():
    return create_device()


@pytest.mark.parametrize("n", [1, 5, 1023, 1024, 1025, 4097, 70_000, 1_000_000])
def test_exclusive_scan(device, n):
    d = device
    rng = np.random.default_rng(n)
    x = rng.integers(0, 50, n).astype(np.uint32)
    src = d.create_buffer_with_data(data=x, usage=STORAGE)
    dst = d.create_buffer(size=4 * n, usage=STORAGE)
    sums = d.create_buffer(size=4 * BLOCK, usage=STORAGE)
    offs = d.create_buffer(size=4 * BLOCK, usage=STORAGE)
    total = d.create_buffer(size=16, usage=STORAGE)
    u = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
    u1 = d.create_buffer_with_data(data=pack_scan_params(n), usage=u)
    u2 = d.create_buffer_with_data(data=pack_scan_params(groups(n, BLOCK)), usage=u)
    m = load_shader(d, "scan")
    p_scan, l_scan = compute_pipeline(
        d, m, "scan_blocks", {0: "uniform", 1: "read", 2: "rw", 3: "rw"}
    )
    p_add, l_add = compute_pipeline(d, m, "add_offsets", {0: "uniform", 2: "rw", 3: "rw"})
    enc = d.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(p_scan)
    cp.set_bind_group(0, bind(d, l_scan, {0: u1, 1: src, 2: dst, 3: sums}))
    cp.dispatch_workgroups(groups(n, BLOCK))
    cp.set_bind_group(0, bind(d, l_scan, {0: u2, 1: sums, 2: offs, 3: total}))
    cp.dispatch_workgroups(1)
    cp.set_pipeline(p_add)
    cp.set_bind_group(0, bind(d, l_add, {0: u1, 2: dst, 3: offs}))
    cp.dispatch_workgroups(groups(n))
    cp.end()
    d.queue.submit([enc.finish()])
    got = np.frombuffer(d.queue.read_buffer(dst), np.uint32)
    expected = np.concatenate([[0], np.cumsum(x, dtype=np.uint64)[:-1]]).astype(np.uint32)
    np.testing.assert_array_equal(got, expected)
    tot = np.frombuffer(d.queue.read_buffer(total), np.uint32)[0]
    assert tot == x.sum()


@pytest.mark.parametrize("r_max", [0.05, 0.1, 0.33])
def test_grid_sort(device, r_max):
    sim = Sim(device, 5000, 4, r_max, seed=1)
    pos0, _ = sim.state()
    n_cells = sim._write_uniforms(1.0)
    enc = device.create_command_encoder()
    sim.encode_grid(enc, n_cells)
    device.queue.submit([enc.finish()])
    nc = sim.nc()
    c3 = np.minimum((pos0[:, :3] * nc).astype(np.int64), nc - 1)
    cells = (c3[:, 0] * nc + c3[:, 1]) * nc + c3[:, 2]
    counts = sim.read(sim.cell_count, np.uint32, (-1,))[:n_cells]
    starts = sim.read(sim.cell_start, np.uint32, (-1,))[:n_cells]
    np.testing.assert_array_equal(counts, np.bincount(cells, minlength=n_cells))
    np.testing.assert_array_equal(starts[1:], np.cumsum(counts)[:-1])
    pos_b = sim.read(sim.pos_b, np.float32, (sim.n, 4))
    cb3 = np.minimum((pos_b[:, :3] * nc).astype(np.int64), nc - 1)
    cells_b = (cb3[:, 0] * nc + cb3[:, 1]) * nc + cb3[:, 2]
    assert (np.diff(cells_b) >= 0).all()  # sorted by cell
    a = pos0[np.lexsort(pos0.T)]
    b = pos_b[np.lexsort(pos_b.T)]
    np.testing.assert_array_equal(a, b)  # same particles, each exactly once


@pytest.mark.parametrize("r_max", [0.1, 0.2, 0.33])
def test_forces_match_bruteforce(device, r_max):
    sim = Sim(device, 700, 5, r_max, seed=2, dt=1.0, friction_half_life=1e-9)
    sim.step()  # dt=1, friction=0 -> new velocity == acceleration of the sorted state
    pos_b = sim.read(sim.pos_b, np.float32, (sim.n, 4))
    _, vel = sim.state()
    expected = acc_bruteforce(
        pos_b[:, :3].astype(np.float64),
        pos_b[:, 3].astype(np.int64),
        sim.matrix_np.astype(np.float64),
        r_max,
        sim.beta,
        sim.force_scale,
    )
    assert np.abs(expected).max() > 0.1
    np.testing.assert_allclose(vel[:, :3], expected, rtol=1e-3, atol=1e-3)


def test_long_run_stays_in_cube_and_keeps_types(device):
    sim = Sim(device, 3000, 6, 0.1, seed=3)
    pos0, _ = sim.state()
    for _ in range(50):
        sim.step()
    pos, vel = sim.state()
    assert np.isfinite(pos).all() and np.isfinite(vel).all()
    assert (pos[:, :3] >= 0).all() and (pos[:, :3] < 1).all()
    assert np.array_equal(np.bincount(pos0[:, 3].astype(int)), np.bincount(pos[:, 3].astype(int)))


def _one(device, xyz):
    sim = Sim(device, 1, 1, 0.1)
    p = np.zeros((1, 4), np.float32)
    p[0, :3] = xyz
    sim.write_state(p, np.zeros((1, 4), np.float32))
    return sim


def test_brush(device):
    sim = _one(device, (0.5, 0.5, 0.5))
    sim.brush((0.55, 0.5, 0.5), 0.2, 10.0)
    assert sim.state()[1][0, 0] > 0
    sim = _one(device, (0.5, 0.5, 0.5))
    sim.brush((0.55, 0.5, 0.5), 0.2, -10.0)
    assert sim.state()[1][0, 0] < 0
    sim = _one(device, (0.5, 0.5, 0.5))
    sim.brush((0.9, 0.5, 0.5), 0.1, 10.0)
    assert sim.state()[1][0, 0] == 0
    sim = _one(device, (0.5, 0.5, 0.98))
    sim.brush((0.5, 0.5, 0.02), 0.2, 10.0)  # across the periodic border
    assert sim.state()[1][0, 2] > 0
