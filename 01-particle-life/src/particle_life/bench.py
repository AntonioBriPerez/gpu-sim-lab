"""Headless simulation benchmark (GPU): steps/s for several particle counts."""

import argparse
import time

import taichi as ti

from .params import Params
from .sim import ParticleLife


def measure(n: int, r_max: float, steps: int, warmup: int = 20) -> float:
    sim = ParticleLife(Params(n_particles=n, r_max=r_max, seed=0))
    for _ in range(warmup):
        sim.step()
    ti.sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        sim.step()
    ti.sync()
    return steps / (time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--counts", type=int, nargs="+", default=[50_000, 100_000, 200_000, 500_000])
    ap.add_argument("--r-max", type=float, nargs="+", default=[0.01, 0.02])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    ti.init(arch=ti.cpu if args.cpu else ti.gpu)
    print("| particles | r_max | neighbours/particle | steps/s |")
    print("|---|---|---|---|")
    import math

    for r in args.r_max:
        for n in args.counts:
            sps = measure(n, r, args.steps)
            print(f"| {n:,} | {r} | {n * math.pi * r * r:.0f} | {sps:.0f} |", flush=True)


if __name__ == "__main__":
    main()
