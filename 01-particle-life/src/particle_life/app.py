"""Interactive front-end: GGUI window for rendering + Tk panel for controls."""

import argparse
import time
from pathlib import Path

import numpy as np
import taichi as ti

from .camera import Camera
from .params import Params, load_preset, mutate_matrix, random_matrix, save_preset
from .render import GlowRenderer
from .sim import ParticleLife
from .ui import Callbacks, ControlPanel

PRESET_DIR = Path("presets")
USER_PRESET_DIR = PRESET_DIR / "user"
RING_SEGMENTS = 48
WARMUP_FRAMES = 30


class App:
    def __init__(self, params: Params, vsync: bool, size: int, font_size: int) -> None:
        self.size = size
        self.window = ti.ui.Window("Particle Life", (size, size), vsync=vsync)
        self.canvas = self.window.get_canvas()
        self.rng = np.random.default_rng(params.seed + 1)
        self.cam = Camera()
        self.preset_idx = -1
        self.ring = ti.Vector.field(2, ti.f32, 2 * RING_SEGMENTS)
        self.prev_cursor: tuple[float, float] | None = None
        self.sim = ParticleLife(params)
        self.renderer = GlowRenderer(size, params.n_types)
        self.panel = ControlPanel(
            params,
            font_size,
            Callbacks(
                on_matrix=lambda m: self.sim.set_matrix(m),
                on_random_matrix=self._random_matrix,
                on_mutate=self._mutate,
                on_reset=lambda: self.sim.reset(int(self.rng.integers(0, 2**31 - 1))),
                on_save=self._save,
                on_load_next=self._load_next,
                on_highlight=self.renderer.set_highlight,
                on_reset_view=self._reset_view,
            ),
        )

    # ---- panel callbacks ---------------------------------------------------
    def _set_matrix(self, m: np.ndarray) -> None:
        self.sim.set_matrix(m)
        self.panel.set_matrix(m)

    def _random_matrix(self) -> None:
        self._set_matrix(random_matrix(self.sim.params.n_types, self.rng))

    def _mutate(self) -> None:
        amount = self.panel.value("mutation")
        self._set_matrix(mutate_matrix(self.sim.params.matrix_array(), amount, self.rng))

    def _reset_view(self) -> None:
        for key, v in (("zoom", 1.0), ("center_x", 0.5), ("center_y", 0.5)):
            self.panel.set_value(key, v)

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
        self.sim = ParticleLife(loaded)
        self.renderer = GlowRenderer(self.size, loaded.n_types)
        self.panel.load_params(loaded)

    # ---- input -------------------------------------------------------------
    def _keys_and_mouse(self, brush_dt_scale: float) -> None:
        w, pn = self.window, self.panel
        zoom = pn.value("zoom")
        pan = 0.012 / zoom
        for key, dx, dy in (("a", -pan, 0), ("d", pan, 0), ("s", 0, -pan), ("w", 0, pan)):
            if w.is_pressed(key):
                pn.nudge("center_x" if dx else "center_y", dx or dy, wrap=True)
        if w.is_pressed("q"):
            pn.nudge("zoom", -zoom * 0.03)
        if w.is_pressed("e"):
            pn.nudge("zoom", zoom * 0.03)
        if w.is_pressed("r"):
            self._reset_view()

        cursor = w.get_cursor_pos()
        if w.is_pressed(ti.ui.MMB) and self.prev_cursor is not None:
            ddx = (cursor[0] - self.prev_cursor[0]) / zoom
            ddy = (cursor[1] - self.prev_cursor[1]) / zoom
            pn.nudge("center_x", -ddx, wrap=True)
            pn.nudge("center_y", -ddy, wrap=True)
        self.prev_cursor = cursor

        self.cam = Camera(pn.value("center_x"), pn.value("center_y"), pn.value("zoom"))
        wx, wy = self.cam.screen_to_world(*cursor)
        radius, strength = pn.value("brush_radius"), pn.value("brush_strength")
        if w.is_pressed(ti.ui.LMB):
            self.sim.apply_brush(wx, wy, radius, strength, brush_dt_scale)
        if w.is_pressed(ti.ui.RMB):
            self.sim.apply_brush(wx, wy, radius, -strength, brush_dt_scale)
        self._draw_ring(cursor, radius * self.cam.zoom)

    def _draw_ring(self, cursor: tuple[float, float], radius: float) -> None:
        a = np.linspace(0, 2 * np.pi, RING_SEGMENTS + 1)
        pts = np.stack([cursor[0] + radius * np.cos(a), cursor[1] + radius * np.sin(a)], axis=1)
        verts = np.empty((2 * RING_SEGMENTS, 2), dtype=np.float32)
        verts[0::2], verts[1::2] = pts[:-1], pts[1:]
        self.ring.from_numpy(verts)

    # ---- main loop ---------------------------------------------------------
    def run(self, bench_frames: int | None = None) -> None:
        t0, frames = time.perf_counter(), 0
        pn = self.panel
        while self.window.running and not pn.closed:
            pn.update()
            pn.apply_to(self.sim.params)
            speed = pn.value("speed")
            if not pn.paused.get() or pn.step_requested:
                self.sim.step(speed)
                pn.step_requested = False
            self._keys_and_mouse(speed)
            self.renderer.draw(
                self.sim.pos,
                self.sim.ptype,
                self.cam,
                pn.value("glow"),
                pn.value("exposure"),
                pn.value("trail"),
            )
            self.canvas.set_image(self.renderer.image)
            self._draw_ring_overlay()
            self.window.show()
            frames += 1
            if bench_frames and frames == WARMUP_FRAMES:
                ti.sync()
                t0 = time.perf_counter()  # exclude kernel JIT compilation from the measurement
            if bench_frames and frames >= WARMUP_FRAMES + bench_frames:
                ti.sync()
                p = self.sim.params
                fps = bench_frames / (time.perf_counter() - t0)
                print(f"{fps:.1f} FPS over {bench_frames} frames (N={p.n_particles}, r={p.r_max})")
                break
        pn.destroy()

    def _draw_ring_overlay(self) -> None:
        self.canvas.lines(self.ring, 0.002, color=(0.7, 0.7, 0.7))


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
