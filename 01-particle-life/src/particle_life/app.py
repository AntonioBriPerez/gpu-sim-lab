"""Interactive GGUI front-end: sliders, random matrix, presets. No simulation logic here."""

import argparse
import time
from pathlib import Path

import numpy as np
import taichi as ti

from .params import R_MAX_LIMIT, R_MIN, Params, load_preset, random_matrix, save_preset
from .sim import ParticleLife

PRESET_DIR = Path("presets")
USER_PRESET_DIR = PRESET_DIR / "user"


def type_colors(n_types: int) -> np.ndarray:
    import colorsys

    return np.array(
        [colorsys.hsv_to_rgb(i / n_types, 0.85, 1.0) for i in range(n_types)], dtype=np.float32
    )


class App:
    def __init__(self, params: Params, vsync: bool) -> None:
        self.window = ti.ui.Window("Particle Life", (1000, 1000), vsync=vsync)
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()
        self.rng = np.random.default_rng(params.seed + 1)
        self.paused = False
        self.show_matrix = True
        self.point_radius = 0.0015
        self.preset_idx = -1
        self._build(params)

    def _build(self, params: Params) -> None:
        self.sim = ParticleLife(params)
        self.colors = ti.Vector.field(3, ti.f32, params.n_particles)
        types = self.sim.ptype.to_numpy().astype(np.intp)
        self.colors.from_numpy(type_colors(params.n_types)[types])

    def _presets(self) -> list[Path]:
        return sorted(PRESET_DIR.glob("*.json")) + sorted(USER_PRESET_DIR.glob("*.json"))

    def _load(self, path: Path) -> None:
        loaded = load_preset(path)
        loaded.n_particles = self.sim.params.n_particles  # keep the current particle budget
        self._build(loaded)

    def _ui(self) -> None:
        p = self.sim.params
        g = self.gui
        g.begin("Controls", 0.01, 0.01, 0.30, 0.34)
        p.r_max = g.slider_float("radius", p.r_max, R_MIN, R_MAX_LIMIT)
        p.beta = g.slider_float("repulsion core (beta)", p.beta, 0.05, 0.95)
        p.force_scale = g.slider_float("force", p.force_scale, 0.0, 30.0)
        p.friction_half_life = g.slider_float(
            "friction half-life", p.friction_half_life, 0.005, 0.5
        )
        self.point_radius = g.slider_float("point size", self.point_radius, 0.0005, 0.005)
        self.paused = g.checkbox("paused", self.paused)
        self.show_matrix = g.checkbox("show matrix", self.show_matrix)
        if g.button("random matrix"):
            self.sim.set_matrix(random_matrix(p.n_types, self.rng))
        if g.button("randomise positions"):
            self.sim.reset(int(self.rng.integers(0, 2**31 - 1)))
        if g.button("save preset"):
            name = USER_PRESET_DIR / f"preset_{time.strftime('%Y%m%d_%H%M%S')}.json"
            save_preset(p, name)
        presets = self._presets()
        if presets and g.button("load next preset"):
            self.preset_idx = (self.preset_idx + 1) % len(presets)
            self._load(presets[self.preset_idx])
        g.end()

        if self.show_matrix:
            n = p.n_types
            g.begin("Attraction matrix (row attracted to column)", 0.01, 0.37, 0.30, 0.60)
            m = p.matrix_array()
            changed = False
            for i in range(n):
                for j in range(n):
                    v = g.slider_float(f"{i}->{j}", float(m[i, j]), -1.0, 1.0)
                    if v != m[i, j]:
                        m[i, j] = v
                        changed = True
            if changed:
                self.sim.set_matrix(m)
            g.end()

    def run(self, bench_frames: int | None = None) -> None:
        t0, frames = time.perf_counter(), 0
        while self.window.running:
            if not self.paused:
                self.sim.step()
            self.canvas.set_background_color((0.02, 0.02, 0.04))
            self.canvas.circles(self.sim.pos, self.point_radius, per_vertex_color=self.colors)
            self._ui()
            self.window.show()
            frames += 1
            if bench_frames and frames >= bench_frames:
                ti.sync()
                dt = time.perf_counter() - t0
                print(
                    f"{frames / dt:.1f} FPS over {frames} frames "
                    f"(N={self.sim.params.n_particles}, r_max={self.sim.params.r_max})"
                )
                break


def main() -> None:
    ap = argparse.ArgumentParser(description="Particle Life (Taichi GGUI)")
    ap.add_argument("--particles", type=int, default=30_000)
    ap.add_argument("--types", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--r-max", type=float, default=0.03)
    ap.add_argument("--preset", type=Path, help="load a preset JSON file")
    ap.add_argument("--no-vsync", action="store_true", help="uncap FPS (for benchmarking)")
    ap.add_argument("--bench-frames", type=int, help="run N frames, print mean FPS and exit")
    args = ap.parse_args()
    ti.init(arch=ti.gpu)
    if args.preset:
        params = load_preset(args.preset)
        params.n_particles = args.particles
    else:
        params = Params(args.particles, args.types, args.seed, r_max=args.r_max)
    App(params, vsync=not args.no_vsync).run(args.bench_frames)


if __name__ == "__main__":
    main()
