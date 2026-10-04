# /// script
# requires-python = ">=3.11"
# dependencies = ["wgpu==0.32.0", "glfw==2.10.2", "numpy>=2.0"]
# ///
"""Environment probe for the 3D Particle Life plan: run it on the target machine.

    uv run probe.py          # native Wayland when the session is Wayland
    uv run probe.py --x11    # force the X11 build of GLFW (XWayland)

Checks: WebGPU adapters, a compute shader with atomics (vs NumPy), GPU timestamps,
and a GLFW window presenting frames (prints the surface platform, formats and FPS).
"""

import argparse
import os
import sys
import time

import numpy as np

COUNT_WGSL = """
@group(0) @binding(0) var<storage, read> cell_of: array<u32>;
@group(0) @binding(1) var<storage, read_write> count: array<atomic<u32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&cell_of)) {
        atomicAdd(&count[cell_of[id.x]], 1u);
    }
}
"""


def compute_check(wgpu, device, n=4_000_000, cells=125_000):
    cell_of = np.random.default_rng(0).integers(0, cells, n).astype(np.uint32)
    usage = wgpu.BufferUsage
    b_in = device.create_buffer_with_data(data=cell_of, usage=usage.STORAGE)
    b_out = device.create_buffer(
        size=cells * 4,
        usage=usage.STORAGE | usage.COPY_SRC | usage.COPY_DST,  # clear_buffer needs COPY_DST
    )
    module = device.create_shader_module(code=COUNT_WGSL)
    pipe = device.create_compute_pipeline(
        layout="auto", compute={"module": module, "entry_point": "main"}
    )
    group = device.create_bind_group(
        layout=pipe.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": b_in}},
            {"binding": 1, "resource": {"buffer": b_out}},
        ],
    )
    times = []
    for _ in range(4):
        enc = device.create_command_encoder()
        enc.clear_buffer(b_out)
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipe)
        cp.set_bind_group(0, group)
        cp.dispatch_workgroups((n + 255) // 256)
        cp.end()
        t0 = time.perf_counter()
        device.queue.submit([enc.finish()])
        got = np.frombuffer(device.queue.read_buffer(b_out), np.uint32)  # also syncs
        times.append(time.perf_counter() - t0)
    ok = np.array_equal(got, np.bincount(cell_of, minlength=cells))
    print(
        f"[compute] {n:,} atomic adds into {cells:,} cells: "
        f"{'OK' if ok else 'MISMATCH'} (best {min(times) * 1e3:.2f} ms incl. readback)"
    )
    return ok


def window_check(wgpu, glfw, device, frames, vsync):
    from wgpu.utils.glfw_present_info import get_glfw_present_info

    glfw.init()
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    win = glfw.create_window(900, 600, "wgpu probe", None, None)
    info = get_glfw_present_info(win, vsync=vsync)
    context = wgpu.gpu.get_canvas_context(info)
    caps = context._get_capabilities(device.adapter)
    fmt = context.get_preferred_format(device.adapter).removesuffix("-srgb")
    context.configure(device=device, format=fmt)
    print(
        f"[window] platform={info['platform']} formats={caps['formats']} "
        f"present_modes={caps['present_modes']} -> using {fmt}, vsync={vsync}"
    )
    print(
        f"[window] window size {glfw.get_window_size(win)}, "
        f"framebuffer {glfw.get_framebuffer_size(win)}, "
        f"content scale {glfw.get_window_content_scale(win)}"
    )
    shown = cancelled = 0
    t0 = time.perf_counter()
    while shown < frames and not glfw.window_should_close(win):
        glfw.poll_events()
        context.set_physical_size(*glfw.get_framebuffer_size(win))
        try:
            tex = context.get_current_texture()
        except wgpu.DrawCancelled:
            cancelled += 1
            time.sleep(0.01)
            continue
        hue = (shown % 120) / 120.0
        enc = device.create_command_encoder()
        rp = enc.begin_render_pass(
            color_attachments=[
                {
                    "view": tex.create_view(),
                    "load_op": "clear",
                    "store_op": "store",
                    "clear_value": (
                        0.5 + 0.5 * np.sin(6.283 * hue),
                        0.2,
                        0.5 - 0.5 * np.sin(6.283 * hue),
                        1,
                    ),
                }
            ]
        )
        rp.end()
        device.queue.submit([enc.finish()])
        context.present()
        shown += 1
    dt = time.perf_counter() - t0
    print(
        f"[window] presented {shown} frames in {dt:.2f}s = {shown / dt:.0f} FPS "
        f"(cancelled {cancelled})"
    )
    context.unconfigure()
    glfw.destroy_window(win)
    glfw.terminate()
    return shown == frames


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--x11", action="store_true", help="use the X11 build of GLFW (XWayland)")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--vsync", action="store_true", help="present with vsync (default: off)")
    ap.add_argument("--no-window", action="store_true")
    args = ap.parse_args()
    if args.x11:
        os.environ["PYGLFW_LIBRARY_VARIANT"] = "x11"  # must be set before importing glfw

    import wgpu

    print(
        f"python {sys.version.split()[0]} | wgpu {wgpu.__version__} | "
        f"XDG_SESSION_TYPE={os.environ.get('XDG_SESSION_TYPE')}"
    )
    for a in wgpu.gpu.enumerate_adapters_sync():
        i = a.info
        print(
            f"[adapter] {i['device']} | {i['adapter_type']} | {i['backend_type']} | "
            f"{i['description']}"
        )
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    lim = adapter.limits
    print(
        f"[chosen] {adapter.info['device']} | timestamp-query="
        f"{'timestamp-query' in adapter.features} | max-storage-buffer-binding-size="
        f"{lim['max-storage-buffer-binding-size'] / 2**20:.0f} MiB | max-buffer-size="
        f"{lim['max-buffer-size'] / 2**20:.0f} MiB"
    )
    device = adapter.request_device_sync()
    ok = compute_check(wgpu, device)
    if not args.no_window:
        import glfw

        print(
            f"[glfw] pyGLFW {glfw.__version__}, library variant "
            f"{os.environ.get('PYGLFW_LIBRARY_VARIANT') or 'auto'}"
        )
        ok = window_check(wgpu, glfw, device, args.frames, args.vsync) and ok
    print("ALL OK" if ok else "SOMETHING FAILED (see above)")


if __name__ == "__main__":
    main()
