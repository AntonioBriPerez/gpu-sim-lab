import numpy as np
import pytest
import taichi as ti

from particle_life.camera import Camera
from particle_life.render import GlowRenderer


@pytest.fixture
def scene():
    pos = ti.Vector.field(2, ti.f32, 1)
    ptype = ti.field(ti.i32, 1)
    pos.from_numpy(np.array([[0.5, 0.25]], dtype=np.float32))
    ptype.from_numpy(np.array([0], dtype=np.int32))
    return pos, ptype


def test_particle_lands_at_expected_pixel(scene):
    pos, ptype = scene
    r = GlowRenderer(64, 3)
    r.draw(pos, ptype, Camera(), sigma=1.5, exposure=1.0, trail=0.0)
    img = r.image.to_numpy().sum(axis=-1)
    x, y = np.unravel_index(img.argmax(), img.shape)
    assert (x, y) == (32, 16) or abs(x - 32) <= 1 and abs(y - 16) <= 1
    assert img.max() > 0.5 and img[0, 0] == 0


def test_zoom_moves_particle_towards_edges(scene):
    pos, ptype = scene
    r = GlowRenderer(64, 3)
    r.draw(pos, ptype, Camera(zoom=2.0), sigma=1.0, exposure=1.0, trail=0.0)
    img = r.image.to_numpy().sum(axis=-1)
    _, y = np.unravel_index(img.argmax(), img.shape)
    assert abs(y - 0) <= 1 or y == 0  # y=0.25 -> 0.5 + (-0.25)*2 = 0 -> bottom edge


def test_highlight_dims_other_types(scene):
    pos, ptype = scene
    r = GlowRenderer(64, 3)
    r.draw(pos, ptype, Camera(), 1.5, 1.0, 0.0)
    full = r.image.to_numpy().max()
    r.set_highlight(1)  # particle is type 0 -> dimmed
    r.draw(pos, ptype, Camera(), 1.5, 1.0, 0.0)
    assert r.image.to_numpy().max() < full * 0.5


def test_trail_persists_and_zero_clears(scene):
    pos, ptype = scene
    r = GlowRenderer(64, 3)
    r.draw(pos, ptype, Camera(), 1.5, 1.0, 0.0)
    pos.from_numpy(np.array([[0.9, 0.9]], dtype=np.float32))
    r.draw(pos, ptype, Camera(), 1.5, 1.0, 0.9)
    assert r.image.to_numpy()[32, 16].sum() > 0  # old position still glows
    r.draw(pos, ptype, Camera(), 1.5, 1.0, 0.0)
    assert r.image.to_numpy()[32, 16].sum() == 0
