import pytest

from particle_life.params import Params, load_preset, save_preset


def test_preset_roundtrip(tmp_path):
    p = Params(n_particles=123, n_types=4, seed=7, r_max=0.05)
    f = tmp_path / "p.json"
    save_preset(p, f)
    assert load_preset(f) == p


def test_same_seed_same_matrix():
    assert Params(seed=3).matrix == Params(seed=3).matrix
    assert Params(seed=3).matrix != Params(seed=4).matrix


@pytest.mark.parametrize("kwargs", [{"r_max": 0.9}, {"r_max": 0.0001}, {"beta": 1.2}])
def test_invalid_params_rejected(kwargs):
    with pytest.raises(ValueError):
        Params(**kwargs)


def test_matrix_shape_checked():
    with pytest.raises(ValueError):
        Params(n_types=3, matrix=[[0.0, 0.0], [0.0, 0.0]])


def test_mutate_matrix_is_bounded_and_close():
    import numpy as np

    from particle_life.params import mutate_matrix

    rng = np.random.default_rng(0)
    m = np.full((4, 4), 0.95, dtype=np.float32)
    out = mutate_matrix(m, 0.1, rng)
    assert out.shape == m.shape and out.max() <= 1.0 and out.min() >= -1.0
    assert not np.array_equal(out, m)
    assert np.abs(out - m).max() < 0.6


def test_friction_scales_with_dt():
    p = Params(dt=0.02, friction_half_life=0.04)
    assert p.friction_factor() == p.friction_factor(0.02)
    assert abs(p.friction_factor(0.04) - 0.5) < 1e-9
