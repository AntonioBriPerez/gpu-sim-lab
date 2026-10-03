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
