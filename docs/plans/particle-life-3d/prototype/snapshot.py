# /// script
# requires-python = ">=3.11"
# dependencies = ["wgpu==0.32.0", "numpy>=2.0"]
# ///
"""Offscreen render to PNG (visual check of the look)."""

import math
import struct
import sys
import time
import zlib

import numpy as np
import wgpu
from camera3d import OrbitCamera
from gpu import create_device
from render3d import Renderer
from sim3d import Sim, neighbours


def write_png(path, rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


n, r_max, steps = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 7
W, H = 1000, 750
device = create_device()
sim = Sim(device, n, 6, r_max, seed=seed)
print(f"N={n} r={r_max} neighbours/particle={neighbours(n, r_max):.0f}")
t0 = time.perf_counter()
for k in range(steps // 10):
    sim.step(10)
device.queue.read_buffer(
    sim.scan_total
)  # blocks until the queue is done (on_submitted_work_done_sync is broken in 0.32.0)
print(f"{steps} steps in {time.perf_counter() - t0:.1f}s")
out = device.create_texture(
    size=(W, H, 1),
    format="rgba8unorm",
    usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
)
r = Renderer(device, "rgba8unorm", 6)
r.resize(W, H)
for name, kw in {
    "full": {"point_size": 0.004, "fog": 1.0, "exposure": 1.2},
    "slice": {"point_size": 0.004, "fog": 0.0, "exposure": 1.5, "slice_half": 0.06},
}.items():
    cam = OrbitCamera(yaw=math.radians(35), pitch=math.radians(25), dist=1.9)
    enc = device.create_command_encoder()
    r.encode(enc, out.create_view(), sim.pos_a, sim.n, cam, **kw)
    device.queue.submit([enc.finish()])
    img = np.frombuffer(
        device.queue.read_texture(
            {"texture": out}, {"bytes_per_row": 4 * W, "rows_per_image": H}, (W, H, 1)
        ),
        np.uint8,
    ).reshape(H, W, 4)
    write_png(f"snap_{name}.png", np.ascontiguousarray(img[..., :3]))
    print("wrote", name)
