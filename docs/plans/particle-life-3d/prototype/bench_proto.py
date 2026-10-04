# /// script
# requires-python = ">=3.11"
# dependencies = ["wgpu==0.32.0", "numpy>=2.0"]
# ///
"""Headless benchmark of the prototype simulation: steps/s and GPU time per phase.

uv run bench_proto.py                       # default sweep
uv run bench_proto.py --counts 200000 1000000 --r-max 0.04
"""

import argparse
import time

import numpy as np
import wgpu
from gpu import create_device
from sim3d import Sim, neighbours


def measure(device, n, r_max, steps, warmup, timestamps):
    sim = Sim(device, n, 6, r_max, seed=0)
    for _ in range(warmup):
        sim.step()
    device.queue.read_buffer(sim.scan_total)  # sync (on_submitted_work_done_sync is broken)
    t0 = time.perf_counter()
    sim.step(steps)
    device.queue.read_buffer(sim.scan_total)
    sps = steps / (time.perf_counter() - t0)
    grid_ms = forces_ms = float("nan")
    if timestamps:
        qs = device.create_query_set(type="timestamp", count=4)
        qbuf = device.create_buffer(
            size=32, usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC
        )
        n_cells = sim._write_uniforms(1.0)
        enc = device.create_command_encoder()
        sim.encode_grid(
            enc,
            n_cells,
            {"query_set": qs, "beginning_of_pass_write_index": 0, "end_of_pass_write_index": 1},
        )
        sim.encode_forces(
            enc, {"query_set": qs, "beginning_of_pass_write_index": 2, "end_of_pass_write_index": 3}
        )
        enc.resolve_query_set(
            query_set=qs, first_query=0, query_count=4, destination=qbuf, destination_offset=0
        )
        device.queue.submit([enc.finish()])
        t = np.frombuffer(device.queue.read_buffer(qbuf), np.uint64).astype(np.float64)
        grid_ms, forces_ms = (t[1] - t[0]) / 1e6, (t[3] - t[2]) / 1e6
    return sps, grid_ms, forces_ms


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--counts", type=int, nargs="+", default=[100_000, 250_000, 500_000, 1_000_000, 2_000_000]
    )
    ap.add_argument("--r-max", type=float, nargs="+", default=[0.03, 0.045, 0.06])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()
    device = create_device(features=("timestamp-query",))
    ts = "timestamp-query" in device.features
    print(f"adapter: {device.adapter.info['device']} ({device.adapter.info['backend_type']})")
    print("| particles | r_max | neighbours/particle | steps/s | grid ms | forces ms |")
    print("|---|---|---|---|---|---|")
    for r in args.r_max:
        for n in args.counts:
            sps, g, f = measure(device, n, r, args.steps, args.warmup, ts)
            print(
                f"| {n:,} | {r} | {neighbours(n, r):.0f} | {sps:.0f} | {g:.2f} | {f:.2f} |",
                flush=True,
            )


if __name__ == "__main__":
    main()
