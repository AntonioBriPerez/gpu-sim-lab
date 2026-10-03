import numpy as np
import pytest

from particle_life.params import Params
from particle_life.reference import accelerations_bruteforce
from particle_life.sim import ParticleLife


@pytest.mark.parametrize("r_max", [0.05, 0.12, 0.3])
def test_grid_matches_bruteforce(r_max):
    p = Params(n_particles=400, n_types=5, seed=1, r_max=r_max)
    sim = ParticleLife(p)
    sim.build_grid(sim.grid_cells_per_side())
    sim.compute_acc(sim.grid_cells_per_side(), p.r_max, p.beta, p.force_scale)
    expected = accelerations_bruteforce(
        sim.pos.to_numpy().astype(np.float64),
        sim.ptype.to_numpy(),
        p.matrix_array().astype(np.float64),
        p.r_max,
        p.beta,
        p.force_scale,
    )
    np.testing.assert_allclose(sim.acc.to_numpy(), expected, rtol=1e-3, atol=1e-3)


def test_every_particle_is_in_the_grid_once():
    sim = ParticleLife(Params(n_particles=500, r_max=0.07, seed=2))
    nc = sim.grid_cells_per_side()
    sim.build_grid(nc)
    assert sim.cell_start.to_numpy()[nc * nc] == 500
    assert sorted(sim.sorted_idx.to_numpy().tolist()) == list(range(500))


def test_same_seed_is_reproducible():
    def run():
        sim = ParticleLife(Params(n_particles=300, r_max=0.1, seed=5))
        for _ in range(10):
            sim.step()
        return sim.pos.to_numpy()

    np.testing.assert_array_equal(run(), run())


def test_positions_stay_in_unit_square_and_finite():
    sim = ParticleLife(Params(n_particles=300, r_max=0.1, seed=6))
    for _ in range(50):
        sim.step()
    pos = sim.pos.to_numpy()
    assert np.isfinite(pos).all()
    assert (pos >= 0).all() and (pos < 1.0 + 1e-6).all()


def test_set_matrix_updates_params():
    sim = ParticleLife(Params(n_particles=50, n_types=3, seed=0))
    sim.set_matrix(np.ones((3, 3), dtype=np.float32))
    assert sim.params.matrix == [[1.0] * 3] * 3
