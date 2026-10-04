"""Prototype of the planned GlowRenderer3D."""

from __future__ import annotations

import colorsys
import math

import numpy as np
import wgpu
from camera3d import OrbitCamera
from gpu import load_shader

MAX_TYPES = 16
ACCUM_FORMAT = wgpu.TextureFormat.rgba16float
DIM_GAIN = 0.12
VS, FS = wgpu.ShaderStage.VERTEX, wgpu.ShaderStage.FRAGMENT
ADD = {"operation": "add", "src_factor": "one", "dst_factor": "one"}
FADE = {"operation": "add", "src_factor": "zero", "dst_factor": "constant"}

CUBE_EDGES = np.array(
    [
        (a, b)
        for a in range(8)
        for b in range(a + 1, 8)
        if (a ^ b).bit_count() == 1  # corners differ in exactly one axis
    ]
)
CUBE_CORNERS = np.array([[(i >> 2) & 1, (i >> 1) & 1, i & 1] for i in range(8)], np.float32) - 0.5


def type_colors(n_types: int) -> np.ndarray:
    return np.array(
        [colorsys.hsv_to_rgb(i / n_types, 0.85, 1.0) for i in range(n_types)], dtype=np.float32
    )


def pack_render_params(
    cam: OrbitCamera,
    width,
    height,
    point_size,
    slice_half,
    fog,
    exposure,
    min_px,
    colors: np.ndarray,
) -> bytes:
    vp = cam.view_proj(width / height).T.astype(np.float32)  # column-major for WGSL
    fwd = cam.basis()[2]
    proj_scale = height / 2.0 / math.tan(cam.fov_y / 2.0)
    head = np.array(
        [
            *cam.target,
            point_size,
            *fwd,
            slice_half,
            width,
            height,
            proj_scale,
            fog,
            exposure,
            min_px,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    cols = np.zeros((MAX_TYPES, 4), np.float32)
    cols[: len(colors), :3] = colors
    return vp.tobytes() + head.tobytes() + cols.tobytes()  # 64 + 64 + 256 = 384


class Renderer:
    def __init__(self, device, out_format, n_types):
        self.device = d = device
        self.n_types = n_types
        self.base_colors = type_colors(n_types)
        self.gains = np.ones(n_types, np.float32)
        m = load_shader(d, "render")
        self.u = d.create_buffer(
            size=384, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        uni = {"binding": 0, "visibility": VS | FS, "buffer": {"type": "uniform"}}
        self.l_particles = d.create_bind_group_layout(
            entries=[uni, {"binding": 1, "visibility": VS, "buffer": {"type": "read-only-storage"}}]
        )
        self.l_tonemap = d.create_bind_group_layout(
            entries=[
                uni,
                {
                    "binding": 2,
                    "visibility": FS,
                    "texture": {"sample_type": "float", "view_dimension": "2d"},
                },
            ]
        )
        self.l_lines = d.create_bind_group_layout(entries=[uni])
        pl = lambda *ls: d.create_pipeline_layout(bind_group_layouts=list(ls))

        def target(fmt, blend=None):
            t = {"format": fmt}
            if blend:
                t["blend"] = {"color": blend, "alpha": blend}
            return t

        self.p_particles = d.create_render_pipeline(
            layout=pl(self.l_particles),
            vertex={"module": m, "entry_point": "vs_particle"},
            primitive={"topology": "triangle-strip"},
            fragment={
                "module": m,
                "entry_point": "fs_particle",
                "targets": [target(ACCUM_FORMAT, ADD)],
            },
        )
        self.p_fade = d.create_render_pipeline(
            layout=pl(),
            vertex={"module": m, "entry_point": "vs_fullscreen"},
            primitive={"topology": "triangle-list"},
            fragment={
                "module": m,
                "entry_point": "fs_fade",
                "targets": [target(ACCUM_FORMAT, FADE)],
            },
        )
        self.p_tonemap = d.create_render_pipeline(
            layout=pl(self.l_tonemap),
            vertex={"module": m, "entry_point": "vs_fullscreen"},
            primitive={"topology": "triangle-list"},
            fragment={"module": m, "entry_point": "fs_tonemap", "targets": [target(out_format)]},
        )
        self.p_lines = d.create_render_pipeline(
            layout=pl(self.l_lines),
            vertex={
                "module": m,
                "entry_point": "vs_line",
                "buffers": [
                    {
                        "array_stride": 24,
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0},
                            {"format": "float32x3", "offset": 12, "shader_location": 1},
                        ],
                    }
                ],
            },
            primitive={"topology": "line-list"},
            fragment={"module": m, "entry_point": "fs_line", "targets": [target(out_format, ADD)]},
        )
        self.line_vbo = d.create_buffer(
            size=24 * 512, usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST
        )
        self.g_lines = d.create_bind_group(
            layout=self.l_lines, entries=[{"binding": 0, "resource": {"buffer": self.u}}]
        )
        self.size = None
        self._pos_buffer = None

    def set_highlight(self, t):
        self.gains[:] = 1.0 if t is None else DIM_GAIN
        if t is not None:
            self.gains[t] = 1.0

    def resize(self, w, h):
        if self.size == (w, h):
            return
        self.size = (w, h)
        self.accum = self.device.create_texture(
            size=(w, h, 1),
            format=ACCUM_FORMAT,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        self.accum_view = self.accum.create_view()
        self.g_tonemap = self.device.create_bind_group(
            layout=self.l_tonemap,
            entries=[
                {"binding": 0, "resource": {"buffer": self.u}},
                {"binding": 2, "resource": self.accum_view},
            ],
        )

    def _bind_particles(self, pos_buffer):
        if pos_buffer is not self._pos_buffer:
            self._pos_buffer = pos_buffer
            self.g_particles = self.device.create_bind_group(
                layout=self.l_particles,
                entries=[
                    {"binding": 0, "resource": {"buffer": self.u}},
                    {"binding": 1, "resource": {"buffer": pos_buffer}},
                ],
            )

    def encode(
        self,
        enc,
        out_view,
        pos_buffer,
        n,
        cam,
        point_size=0.004,
        slice_half=10.0,
        fog=0.0,
        exposure=1.0,
        trail=0.0,
        min_px=0.75,
        show_box=True,
        ring=None,
    ):
        w, h = self.size
        self._bind_particles(pos_buffer)
        colors = self.base_colors * self.gains[:, None]
        self.device.queue.write_buffer(
            self.u,
            0,
            pack_render_params(cam, w, h, point_size, slice_half, fog, exposure, min_px, colors),
        )
        keep = trail > 0.0
        rp = enc.begin_render_pass(
            color_attachments=[
                {
                    "view": self.accum_view,
                    "load_op": "load" if keep else "clear",
                    "store_op": "store",
                    "clear_value": (0, 0, 0, 0),
                }
            ]
        )
        if keep:
            rp.set_pipeline(self.p_fade)
            rp.set_blend_constant((trail, trail, trail, trail))
            rp.draw(3)
        rp.set_pipeline(self.p_particles)
        rp.set_bind_group(0, self.g_particles)
        rp.draw(4, n)
        rp.end()

        lines = []
        if show_box:
            col = (0.18, 0.18, 0.22)
            for a, b in CUBE_EDGES:
                lines += [(*CUBE_CORNERS[a], *col), (*CUBE_CORNERS[b], *col)]
        if ring is not None:
            lines += ring
        rp = enc.begin_render_pass(
            color_attachments=[
                {
                    "view": out_view,
                    "load_op": "clear",
                    "store_op": "store",
                    "clear_value": (0, 0, 0, 1),
                }
            ]
        )
        rp.set_pipeline(self.p_tonemap)
        rp.set_bind_group(0, self.g_tonemap)
        rp.draw(3)
        if lines:
            data = np.array(lines, np.float32)
            self.device.queue.write_buffer(self.line_vbo, 0, data)
            rp.set_pipeline(self.p_lines)
            rp.set_bind_group(0, self.g_lines)
            rp.set_vertex_buffer(0, self.line_vbo)
            rp.draw(len(data))
        rp.end()
