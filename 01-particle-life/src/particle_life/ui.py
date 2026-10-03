"""Tk control panel (HiDPI-aware, unlike Taichi GGUI's fixed-size ImGui font)."""

import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass
from tkinter import font as tkfont

import numpy as np

from .camera import ZOOM_MAX, ZOOM_MIN
from .colors import type_hex
from .params import R_MAX_LIMIT, R_MIN, Params


@dataclass
class Slider:
    key: str
    label: str
    lo: float
    hi: float
    step: float
    default: float | None  # None: taken from Params
    help: str


SIM_SLIDERS = [
    Slider(
        "r_max",
        "radius",
        R_MIN,
        R_MAX_LIMIT,
        0.001,
        None,
        "How far a particle 'sees'. Bigger = larger structures, but slower.",
    ),
    Slider(
        "beta",
        "repulsion core (beta)",
        0.05,
        0.95,
        0.01,
        None,
        "Hard repulsive core: below this distance every pair pushes apart.",
    ),
    Slider(
        "force_scale",
        "force",
        0.0,
        30.0,
        0.1,
        None,
        "How strongly particles accelerate. Higher = more violent motion.",
    ),
    Slider(
        "friction_half_life",
        "friction half-life (s)",
        0.005,
        0.5,
        0.005,
        None,
        "Time for speed to halve. Higher = slides longer, more chaotic.",
    ),
    Slider(
        "speed",
        "speed",
        0.02,
        1.5,
        0.01,
        1.0,
        "Time scale. Low = slow motion so you can follow the interactions.",
    ),
]
LOOK_SLIDERS = [
    Slider(
        "glow",
        "glow size (px)",
        0.5,
        6.0,
        0.1,
        1.5,
        "Size of each particle's halo. Small = crisp dots.",
    ),
    Slider(
        "exposure",
        "brightness",
        0.1,
        5.0,
        0.05,
        1.0,
        "Overlapping halos add up: dense clumps glow brighter (nebula effect).",
    ),
    Slider("trail", "trail", 0.0, 0.98, 0.01, 0.0, "How long motion trails last. 0 = none."),
]
VIEW_SLIDERS = [
    Slider("zoom", "zoom (keys Q/E)", ZOOM_MIN, ZOOM_MAX, 0.1, 1.0, "Zoom into the world."),
    Slider("center_x", "view x (key A/D)", 0.0, 1.0, 0.005, 0.5, "The world wraps around."),
    Slider("center_y", "view y (key W/S)", 0.0, 1.0, 0.005, 0.5, "Middle mouse drag also pans."),
]
BRUSH_SLIDERS = [
    Slider(
        "brush_radius",
        "brush radius",
        0.01,
        0.3,
        0.005,
        0.06,
        "Left mouse button on the render window pulls particles, right button pushes.",
    ),
    Slider("brush_strength", "brush strength", 0.0, 30.0, 0.5, 8.0, "How hard the brush acts."),
    Slider(
        "mutation",
        "mutation amount",
        0.02,
        0.8,
        0.01,
        0.15,
        "How much 'mutate matrix' perturbs every cell.",
    ),
]
ALL_SLIDERS = SIM_SLIDERS + LOOK_SLIDERS + VIEW_SLIDERS + BRUSH_SLIDERS
SLIDER_BY_KEY = {s.key: s for s in ALL_SLIDERS}

HELP_TEXT = """\
WHAT YOU SEE
Each glowing dot is a particle; its colour is its type. Particles only push or
pull their neighbours depending on both types. Structures emerge by themselves.

THE RULES (between two particles)
- Very close (inside the core, 'beta'): they always repel, so they never collapse.
- Medium distance (up to 'radius'): attract or repel according to the matrix.
- Farther than 'radius': they ignore each other.

THE MATRIX
Row = the type that reacts. Column = the type it reacts to.
+1: the row type is attracted to the column type (moves toward it).
-1: the row type is repelled by it (runs away).
 0: ignores it.
It is NOT symmetric: red->green and green->red are independent, so you can
get chasers and chased.

SEEING BETTER
- 'speed': slow motion. Pause + 'step' advances one frame at a time.
- 'trail': motion leaves fading streaks.
- Highlight buttons: dim every type except one to follow who chases whom.
- 'brightness' / 'glow size': dense clumps glow like nebulae.

PLAYING
- Mouse on the render window: left button pulls, right button pushes.
- Zoom: Q / E or slider. Pan: W A S D, slider or middle-mouse drag. R resets the view.
- 'mutate matrix' jitters the matrix a bit: explore variations of a good one.
- 'random matrix' makes a brand-new universe. 'save preset' keeps the current one.
"""


@dataclass
class Callbacks:
    on_matrix: Callable[[np.ndarray], None]
    on_random_matrix: Callable[[], None]
    on_mutate: Callable[[], None]
    on_reset: Callable[[], None]
    on_save: Callable[[], None]
    on_load_next: Callable[[], None]
    on_highlight: Callable[[int | None], None]
    on_reset_view: Callable[[], None]


class ControlPanel:
    def __init__(self, params: Params, font_size: int, cb: Callbacks) -> None:
        self.root = tk.Tk()
        self.root.title("Particle Life - controls")
        for name in ("TkDefaultFont", "TkTextFont", "TkFixedFont"):
            tkfont.nametofont(name).configure(size=font_size)
        self.closed = False
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.cb = cb
        self._silent = False
        self.step_requested = False

        self.vars = {
            s.key: tk.DoubleVar(
                value=s.default if s.default is not None else getattr(params, s.key)
            )
            for s in ALL_SLIDERS
        }
        self.paused = tk.BooleanVar(value=False)

        left = tk.Frame(self.root)
        left.grid(row=0, column=0, sticky="n", padx=8, pady=8)
        right = tk.Frame(self.root)
        right.grid(row=0, column=1, sticky="n", padx=8, pady=8)

        self._section(left, "Simulation", SIM_SLIDERS)
        row = tk.Frame(left)
        row.pack(fill="x", pady=2)
        tk.Checkbutton(row, text="paused", variable=self.paused).pack(side="left")
        tk.Button(row, text="step", command=self._request_step).pack(side="left", padx=3)
        tk.Button(row, text="randomise positions", command=cb.on_reset).pack(side="left", padx=3)
        self._section(left, "Look", LOOK_SLIDERS)
        self.highlight_frame = tk.Frame(left)
        self.highlight_frame.pack(fill="x", pady=2)
        self._section(left, "View", VIEW_SLIDERS)
        tk.Button(left, text="reset view (R)", command=cb.on_reset_view).pack(anchor="w")
        self._section(left, "Mouse brush and mutation", BRUSH_SLIDERS)

        tk.Label(
            right,
            text="Attraction matrix: ROW type reacts to COLUMN type\n"
            "(+1 = row moves toward column, -1 = row runs away, 0 = ignores)",
        ).pack()
        self.matrix_frame = tk.Frame(right)
        self.matrix_frame.pack(pady=6)
        btns = tk.Frame(right)
        btns.pack(fill="x", pady=4)
        for text, fn in (
            ("random matrix", cb.on_random_matrix),
            ("mutate matrix", cb.on_mutate),
            ("save preset", cb.on_save),
            ("load next preset", cb.on_load_next),
            ("help", self._show_help),
        ):
            tk.Button(btns, text=text, command=fn).pack(side="left", padx=3)

        self.matrix_vars: list[list[tk.DoubleVar]] = []
        self.set_matrix(params.matrix_array())

    def _section(self, parent: tk.Frame, title: str, sliders: list[Slider]) -> None:
        tk.Label(parent, text=title, font=("TkDefaultFont", self._font_size(), "bold")).pack(
            anchor="w", pady=(8, 0)
        )
        for s in sliders:
            tk.Scale(
                parent,
                label=s.label,
                variable=self.vars[s.key],
                from_=s.lo,
                to=s.hi,
                resolution=s.step,
                orient="horizontal",
                length=480,
            ).pack(fill="x")
            tk.Label(
                parent, text=s.help, fg="#666666", anchor="w", justify="left", wraplength=480
            ).pack(fill="x")

    @staticmethod
    def _font_size() -> int:
        return int(tkfont.nametofont("TkDefaultFont").cget("size"))

    def _request_step(self) -> None:
        self.step_requested = True

    def _show_help(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Help")
        tk.Label(win, text=HELP_TEXT, justify="left", anchor="w").pack(padx=12, pady=12)

    def _close(self) -> None:
        self.closed = True

    def value(self, key: str) -> float:
        return self.vars[key].get()

    def nudge(self, key: str, delta: float, wrap: bool = False) -> None:
        s = SLIDER_BY_KEY[key]
        v = self.vars[key].get() + delta
        self.vars[key].set(v % 1.0 if wrap else min(s.hi, max(s.lo, v)))

    def set_value(self, key: str, value: float) -> None:
        self.vars[key].set(value)

    def set_matrix(self, matrix: np.ndarray) -> None:
        """Update the matrix grid; (re)builds the widgets only if the size changed."""
        n = matrix.shape[0]
        self._silent = True
        if len(self.matrix_vars) != n:
            self._build_type_widgets(n)
        for i in range(n):
            for j in range(n):
                self.matrix_vars[i][j].set(float(matrix[i, j]))
        self._silent = False

    def _build_type_widgets(self, n: int) -> None:
        for w in self.matrix_frame.winfo_children() + self.highlight_frame.winfo_children():
            w.destroy()
        colors = type_hex(n)
        self.matrix_vars = [[tk.DoubleVar() for _ in range(n)] for _ in range(n)]
        for k in range(n):
            for r, c in ((k + 1, 0), (0, k + 1)):
                tk.Label(self.matrix_frame, text=f"type {k}", bg=colors[k], fg="black").grid(
                    row=r, column=c, padx=2, pady=2, sticky="nsew"
                )
        for i in range(n):
            for j in range(n):
                tk.Scale(
                    self.matrix_frame,
                    variable=self.matrix_vars[i][j],
                    from_=-1.0,
                    to=1.0,
                    resolution=0.01,
                    orient="horizontal",
                    length=130,
                    command=lambda _v: self._matrix_changed(),
                ).grid(row=i + 1, column=j + 1, padx=2, pady=2)
        tk.Label(self.highlight_frame, text="highlight:").pack(side="left")
        tk.Button(
            self.highlight_frame, text="all", command=lambda: self.cb.on_highlight(None)
        ).pack(side="left", padx=2)
        for k in range(n):
            tk.Button(
                self.highlight_frame,
                text=str(k),
                bg=colors[k],
                activebackground=colors[k],
                fg="black",
                command=lambda k=k: self.cb.on_highlight(k),
            ).pack(side="left", padx=2)

    def _matrix_changed(self) -> None:
        if not self._silent:
            self.cb.on_matrix(self.matrix())

    def matrix(self) -> np.ndarray:
        return np.array([[v.get() for v in row] for row in self.matrix_vars], dtype=np.float32)

    def apply_to(self, params: Params) -> None:
        """Copy the simulation sliders into params."""
        params.r_max = self.value("r_max")
        params.beta = self.value("beta")
        params.force_scale = self.value("force_scale")
        params.friction_half_life = self.value("friction_half_life")

    def load_params(self, params: Params) -> None:
        for key in ("r_max", "beta", "force_scale", "friction_half_life"):
            self.vars[key].set(getattr(params, key))
        self.set_matrix(params.matrix_array())

    def update(self) -> None:
        self.root.update()

    def destroy(self) -> None:
        self.root.destroy()
