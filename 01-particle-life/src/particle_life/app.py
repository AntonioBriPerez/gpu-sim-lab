"""Interactive front-end: GGUI window for rendering + Tk panel for controls."""

import argparse
import time
from pathlib import Path

import numpy as np
import taichi as ti

from .colors import type_colors
from .params import Params, load_preset, random_matrix, save_preset
from .sim import ParticleLife
from .ui import ControlPanel

PRESET_DIR = Path("presets")
USER_PRESET_DIR = PRESET_DIR / "user"


class App:
    def __init__(self, params: Params, vsync: bool, size: int, font_size: int) -> None:
        self.window = ti.ui.Window("Particle Life", (size, size), vsync=vsync)
        self.canvas = self.window.get_canvas()
        self.rng = np.random.default_rng(params.seed + 1)
        self.point_radius = 0.0015
        self.preset_idx = -1
        self._build(params)
        self.panel = ControlPanel(
            params,
            font_size,
            on_matrix=self.sim.set_matrix,
            on_random_matrix=self._random_matrix,
            on_reset=lambda: self.sim.reset(int(self.rng.integers(0, 2**31 - 1))),
            on_save=self._save,
            on_load_next=self._load_next,
        )

    def _build(self, params: Params) -> None:
        self.sim = ParticleLife(params)
        self.colors = ti.Vector.field(3, ti.f32, params.n_particles)
        types = self.sim.ptype.to_numpy().astype(np.intp)
        self.colors.from_numpy(type_colors(params.n_types)[types])

    def _random_matrix(self) -> None:
        m = random_matrix(self.sim.params.n_types, self.rng)
        self.sim.set_matrix(m)
        self.panel.set_matrix(m)

    def _save(self) -> None:
        name = USER_PRESET_DIR / f"preset_{time.strftime('%Y%m%d_%H%M%S')}.json"
        save_preset(self.sim.params, name)
        print(f"saved {name}")

    def _load_next(self) -> None:
        presets = sorted(PRESET_DIR.glob("*.json")) + sorted(USER_PRESET_DIR.glob("*.json"))
        if not presets:
            print("no presets found")
            return
        self.preset_idx = (self.preset_idx + 1) % len(presets)
        loaded = load_preset(presets[self.preset_idx])
        loaded.n_particles = self.sim.params.n_particles  # keep the current particle budget
        self._build(loaded)
        self.panel._on_matrix = self.sim.set_matrix
        self.panel.load_params(loaded)

    def run(self, bench_frames: int | None = None) -> None:
        t0, frames = time.perf_counter(), 0
        while self.window.running and not self.panel.closed:
            self.panel.update()
            self.point_radius = self.panel.apply_to(self.sim.params)
            if not self.panel.paused.get():
                self.sim.step()
            self.canvas.set_background_color((0.02, 0.02, 0.04))
            self.canvas.circles(self.sim.pos, self.point_radius, per_vertex_color=self.colors)
            self.window.show()
            frames += 1
            if bench_frames and frames >= bench_frames:
                ti.sync()
                dt = time.perf_counter() - t0
                p = self.sim.params
                print(f"{frames / dt:.1f} FPS over {frames} frames ")
                print(f"(N={p.n_particles}, r_max={p.r_max:.3f})")
                break
        self.panel.destroy()


def main() -> None:
    ap = argparse.ArgumentParser(description="Particle Life (Taichi + Tk controls)")
    ap.add_argument("--particles", type=int, default=30_000)
    ap.add_argument("--types", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--r-max", type=float, default=0.03)
    ap.add_argument("--preset", type=Path, help="load a preset JSON file")
    ap.add_argument("--size", type=int, default=1400, help="render window size in pixels (square)")
    ap.add_argument("--ui-font", type=int, default=14, help="control panel font size (points)")
    ap.add_argument("--no-vsync", action="store_true", help="uncap FPS (for benchmarking)")
    ap.add_argument("--bench-frames", type=int, help="run N frames, print mean FPS and exit")
    args = ap.parse_args()
    ti.init(arch=ti.gpu)
    if args.preset:
        params = load_preset(args.preset)
        params.n_particles = args.particles
    else:
        params = Params(args.particles, args.types, args.seed, r_max=args.r_max)
    App(params, not args.no_vsync, args.size, args.ui_font).run(args.bench_frames)


if __name__ == "__main__":
    main()
