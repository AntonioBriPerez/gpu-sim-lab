"""Tk control panel (HiDPI-aware, unlike Taichi GGUI's fixed-size ImGui font)."""

import tkinter as tk
from collections.abc import Callable
from tkinter import font as tkfont

import numpy as np

from .params import R_MAX_LIMIT, R_MIN, Params


class ControlPanel:
    def __init__(
        self,
        params: Params,
        font_size: int,
        on_matrix: Callable[[np.ndarray], None],
        on_random_matrix: Callable[[], None],
        on_reset: Callable[[], None],
        on_save: Callable[[], None],
        on_load_next: Callable[[], None],
    ) -> None:
        self.root = tk.Tk()
        self.root.title("Particle Life - controls")
        for name in ("TkDefaultFont", "TkTextFont", "TkFixedFont"):
            tkfont.nametofont(name).configure(size=font_size)
        self.closed = False
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self._on_matrix = on_matrix
        self._silent = False

        self.vars = {
            "r_max": tk.DoubleVar(value=params.r_max),
            "beta": tk.DoubleVar(value=params.beta),
            "force_scale": tk.DoubleVar(value=params.force_scale),
            "friction_half_life": tk.DoubleVar(value=params.friction_half_life),
            "point_radius": tk.DoubleVar(value=0.0015),
        }
        self.paused = tk.BooleanVar(value=False)

        top = tk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=8)
        self._scale(top, "radius", "r_max", R_MIN, R_MAX_LIMIT, 0.001)
        self._scale(top, "repulsion core (beta)", "beta", 0.05, 0.95, 0.01)
        self._scale(top, "force", "force_scale", 0.0, 30.0, 0.1)
        self._scale(top, "friction half-life (s)", "friction_half_life", 0.005, 0.5, 0.005)
        self._scale(top, "point size", "point_radius", 0.0005, 0.005, 0.0001)

        buttons = tk.Frame(self.root)
        buttons.pack(fill="x", padx=8)
        tk.Checkbutton(buttons, text="paused", variable=self.paused).pack(side="left")
        for text, cb in (
            ("random matrix", on_random_matrix),
            ("randomise positions", on_reset),
            ("save preset", on_save),
            ("load next preset", on_load_next),
        ):
            tk.Button(buttons, text=text, command=cb).pack(side="left", padx=3)

        tk.Label(self.root, text="Attraction matrix (row i is attracted to column j)").pack(
            pady=(10, 0)
        )
        self.matrix_frame = tk.Frame(self.root)
        self.matrix_frame.pack(padx=8, pady=8)
        self.matrix_vars: list[list[tk.DoubleVar]] = []
        self.set_matrix(params.matrix_array())

    def _scale(
        self, parent: tk.Frame, label: str, key: str, lo: float, hi: float, step: float
    ) -> None:
        tk.Scale(
            parent,
            label=label,
            variable=self.vars[key],
            from_=lo,
            to=hi,
            resolution=step,
            orient="horizontal",
            length=520,
        ).pack(fill="x")

    def _close(self) -> None:
        self.closed = True

    def set_matrix(self, matrix: np.ndarray) -> None:
        """(Re)build the matrix grid; rebuilds only if the size changed."""
        n = matrix.shape[0]
        self._silent = True
        if len(self.matrix_vars) != n:
            for w in self.matrix_frame.winfo_children():
                w.destroy()
            self.matrix_vars = [[tk.DoubleVar() for _ in range(n)] for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    tk.Scale(
                        self.matrix_frame,
                        label=f"{i}->{j}",
                        variable=self.matrix_vars[i][j],
                        from_=-1.0,
                        to=1.0,
                        resolution=0.01,
                        orient="horizontal",
                        length=130,
                        command=lambda _v: self._matrix_changed(),
                    ).grid(row=i, column=j, padx=2, pady=2)
        for i in range(n):
            for j in range(n):
                self.matrix_vars[i][j].set(float(matrix[i, j]))
        self._silent = False

    def _matrix_changed(self) -> None:
        if not self._silent:
            self._on_matrix(self.matrix())

    def matrix(self) -> np.ndarray:
        return np.array([[v.get() for v in row] for row in self.matrix_vars], dtype=np.float32)

    def apply_to(self, params: Params) -> float:
        """Copy slider values into params; returns the point radius for rendering."""
        params.r_max = self.vars["r_max"].get()
        params.beta = self.vars["beta"].get()
        params.force_scale = self.vars["force_scale"].get()
        params.friction_half_life = self.vars["friction_half_life"].get()
        return self.vars["point_radius"].get()

    def load_params(self, params: Params) -> None:
        self.vars["r_max"].set(params.r_max)
        self.vars["beta"].set(params.beta)
        self.vars["force_scale"].set(params.force_scale)
        self.vars["friction_half_life"].set(params.friction_half_life)
        self.set_matrix(params.matrix_array())

    def update(self) -> None:
        self.root.update()

    def destroy(self) -> None:
        self.root.destroy()
