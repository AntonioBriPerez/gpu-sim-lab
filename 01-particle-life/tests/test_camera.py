import pytest

from particle_life.camera import ZOOM_MAX, Camera


def test_identity_view_maps_screen_to_world():
    assert Camera().screen_to_world(0.25, 0.75) == pytest.approx((0.25, 0.75))


def test_zoom_and_centre():
    cam = Camera(cx=0.2, cy=0.2, zoom=4.0)
    assert cam.screen_to_world(0.5, 0.5) == pytest.approx((0.2, 0.2))
    assert cam.screen_to_world(1.0, 0.5) == pytest.approx((0.2 + 0.125, 0.2))


def test_world_wraps():
    cam = Camera(cx=0.05, cy=0.5, zoom=2.0)
    assert cam.screen_to_world(0.0, 0.5)[0] == pytest.approx(0.8)


def test_zoom_is_clamped_and_pan_wraps():
    cam = Camera()
    cam.zoom_by(1000)
    assert cam.zoom == ZOOM_MAX
    cam.zoom_by(1e-9)
    assert cam.zoom == 1.0
    cam.pan(0.7, -0.7)
    assert cam.cx == pytest.approx(0.2) and cam.cy == pytest.approx(0.8)
