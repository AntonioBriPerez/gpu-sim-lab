import math

import numpy as np
import pytest
import wgpu
from camera3d import OrbitCamera
from gpu import create_device
from render3d import Renderer

W, H = 160, 120
U = wgpu.BufferUsage


@pytest.fixture(scope="session")
def device():
    return create_device()


def make(device, n_types=3):
    r = Renderer(device, "rgba8unorm", n_types)
    r.resize(W, H)
    out = device.create_texture(
        size=(W, H, 1),
        format="rgba8unorm",
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
    )
    return r, out


def particles(device, pts, types):
    a = np.zeros((len(pts), 4), np.float32)
    a[:, :3] = pts
    a[:, 3] = types
    return device.create_buffer_with_data(data=a, usage=U.STORAGE | U.COPY_DST)


def render(device, r, out, buf, n, cam, **kw):
    enc = device.create_command_encoder()
    r.encode(enc, out.create_view(), buf, n, cam, show_box=False, **kw)
    device.queue.submit([enc.finish()])
    img = device.queue.read_texture(
        {"texture": out, "origin": (0, 0, 0)},
        {"bytes_per_row": 4 * W, "rows_per_image": H},
        (W, H, 1),
    )
    return np.frombuffer(img, np.uint8).reshape(H, W, 4)[..., :3].astype(np.float32).sum(-1)


@pytest.mark.parametrize("yaw,pitch", [(0.0, 0.0), (0.6, 0.3), (2.5, -0.8)])
def test_particle_lands_where_camera_projects_it(device, yaw, pitch):
    r, out = make(device)
    cam = OrbitCamera(yaw=yaw, pitch=pitch, dist=1.6)
    pt = np.array([[0.62, 0.41, 0.55]])
    img = render(device, r, out, particles(device, pt, [0]), 1, cam, point_size=0.03)
    y, x = np.unravel_index(img.argmax(), img.shape)
    ex, ey = cam.project(pt, W, H)[0]
    assert abs(x + 0.5 - ex) <= 1.5 and abs(y + 0.5 - ey) <= 1.5
    assert img.max() > 100 and img[0, 0] == 0


def test_periodic_wrap_draws_nearest_copy(device):
    r, out = make(device)
    cam = OrbitCamera(yaw=0.0, pitch=0.0, dist=1.6, target=np.array([0.95, 0.5, 0.5]))
    pt = np.array([[0.05, 0.5, 0.5]])  # 0.1 to the right of the target across the border
    img = render(device, r, out, particles(device, pt, [0]), 1, cam, point_size=0.01)
    y, x = np.unravel_index(img.argmax(), img.shape)
    assert x > W / 2 + 3 and abs(y - H / 2) <= 2


def test_highlight_dims_other_types(device):
    r, out = make(device)
    cam = OrbitCamera(dist=1.6)
    buf = particles(device, np.array([[0.5, 0.5, 0.5]]), [0])
    full = render(device, r, out, buf, 1, cam, point_size=0.01).max()
    r.set_highlight(1)
    assert render(device, r, out, buf, 1, cam, point_size=0.01).max() < 0.5 * full


def test_trail_persists_and_zero_clears(device):
    r, out = make(device)
    cam = OrbitCamera(yaw=0.0, pitch=0.0, dist=1.6)
    a = np.array([[0.4, 0.5, 0.5]])
    ax, ay = (int(v) for v in cam.project(a, W, H)[0])
    render(device, r, out, particles(device, a, [0]), 1, cam, point_size=0.01)
    b = particles(device, np.array([[0.6, 0.5, 0.5]]), [0])
    assert render(device, r, out, b, 1, cam, point_size=0.01, trail=0.9)[ay, ax] > 0
    assert render(device, r, out, b, 1, cam, point_size=0.01, trail=0.0)[ay, ax] == 0


def test_slice_hides_particles_off_the_focal_plane(device):
    r, out = make(device)
    cam = OrbitCamera(yaw=0.0, pitch=0.0, dist=1.6)  # looking along -z
    near_plane = particles(device, np.array([[0.5, 0.5, 0.5]]), [0])
    far_off = particles(device, np.array([[0.5, 0.5, 0.2]]), [0])  # 0.3 behind the focal plane
    assert render(device, r, out, near_plane, 1, cam, point_size=0.03, slice_half=0.1).max() > 0
    assert render(device, r, out, far_off, 1, cam, point_size=0.03, slice_half=0.1).max() == 0
    assert render(device, r, out, far_off, 1, cam, point_size=0.03, slice_half=0.5).max() > 0


def test_tiny_far_particles_still_visible(device):
    r, out = make(device)
    cam = OrbitCamera(dist=5.0)
    # about 0.3 px: without the 0.75 px floor this would often hit no pixel centre at all
    img = render(
        device,
        r,
        out,
        particles(device, np.array([[0.5, 0.5, 0.5]]), [0]),
        1,
        cam,
        point_size=0.012,
        min_px=0.75,
    )
    assert img.max() > 0


def test_focal_point_is_under_the_cursor():
    cam = OrbitCamera(yaw=0.7, pitch=0.4, dist=1.0, target=np.array([0.2, 0.9, 0.5]))
    for sx, sy in [(0.5, 0.5), (0.3, 0.6), (0.7, 0.35)]:
        p = cam.focal_point(sx, sy, W / H)
        assert p is not None and (p >= 0).all() and (p < 1).all()
        px, py = cam.project(p[None], W, H)[0]
        assert math.isclose(px / W, sx, abs_tol=1e-6) and math.isclose(py / H, sy, abs_tol=1e-6)


def test_focal_point_outside_the_cube_is_none():
    cam = OrbitCamera(yaw=0.0, pitch=0.0, dist=1.8)
    assert cam.focal_point(0.02, 0.5, W / H) is None


def test_fog_dims_far_particles(device):
    r, out = make(device)
    cam = OrbitCamera(yaw=0.0, pitch=0.0, dist=1.6)  # looking along -z: smaller z is farther
    near = particles(device, np.array([[0.5, 0.5, 0.8]]), [0])
    far = particles(device, np.array([[0.5, 0.5, 0.2]]), [0])
    near_px = render(device, r, out, near, 1, cam, point_size=0.03, fog=2.0).max()
    far_px = render(device, r, out, far, 1, cam, point_size=0.03, fog=2.0).max()
    assert near_px > 0 and far_px < 0.5 * near_px
