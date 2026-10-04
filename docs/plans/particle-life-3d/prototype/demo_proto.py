# /// script
# requires-python = ">=3.11"
# dependencies = ["wgpu==0.32.0", "glfw==2.10.2", "numpy>=2.0"]
# ///
"""Interactive prototype: 3D Particle Life with wgpu + GLFW (no Tk panel).

    uv run demo_proto.py                          # 200k particles
    uv run demo_proto.py --particles 1000000 --r-max 0.03 --no-vsync
    uv run demo_proto.py --x11                    # force XWayland if native Wayland fails

Mouse: left drag orbit, right/middle drag pan, wheel zoom,
       Shift+left pull, Shift+right push (on the plane through the view centre).
Keys:  W/S forward/back, A/D left/right, Q/E down/up (the world wraps around),
       O auto-rotate, Space pause, B box, R reset view, F slice on/off, +/- slice width.
"""

import argparse
import math
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--particles", type=int, default=200_000)
    ap.add_argument("--types", type=int, default=6)
    ap.add_argument("--r-max", type=float, default=0.045)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=float, default=0.0035, help="glow radius (world units)")
    ap.add_argument("--no-vsync", action="store_true")
    ap.add_argument("--x11", action="store_true", help="use the X11 build of GLFW (XWayland)")
    ap.add_argument("--frames", type=int, default=0, help="quit after N frames (0 = never)")
    args = ap.parse_args()
    if args.x11:
        os.environ["PYGLFW_LIBRARY_VARIANT"] = "x11"  # must be set before importing glfw

    import glfw
    import wgpu
    from camera3d import OrbitCamera
    from gpu import create_device
    from render3d import Renderer
    from sim3d import Sim, neighbours
    from wgpu.utils.glfw_present_info import get_glfw_present_info

    glfw.init()
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    win = glfw.create_window(1400, 1000, "Particle Life 3D (prototype)", None, None)
    info = get_glfw_present_info(win, vsync=not args.no_vsync)
    context = wgpu.gpu.get_canvas_context(info)
    device = create_device()
    fmt = context.get_preferred_format(device.adapter).removesuffix("-srgb")
    context.configure(device=device, format=fmt)
    print(f"adapter: {device.adapter.info['device']} | surface: {info['platform']} | format: {fmt}")
    print(
        f"N={args.particles:,} r_max={args.r_max} -> "
        f"{neighbours(args.particles, args.r_max):.0f} neighbours/particle"
    )

    sim = Sim(device, args.particles, args.types, args.r_max, seed=args.seed)
    renderer = Renderer(device, fmt, args.types)
    cam = OrbitCamera()
    state = {
        "auto": True,
        "paused": False,
        "box": True,
        "slice": False,
        "slice_half": 0.08,
        "scroll": 0.0,
    }

    def on_key(_w, key, _sc, action, _mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_O:
            state["auto"] = not state["auto"]
        elif key == glfw.KEY_SPACE:
            state["paused"] = not state["paused"]
        elif key == glfw.KEY_B:
            state["box"] = not state["box"]
        elif key == glfw.KEY_F:
            state["slice"] = not state["slice"]
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            state["slice_half"] = min(0.5, state["slice_half"] * 1.25)
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            state["slice_half"] = max(0.01, state["slice_half"] / 1.25)
        elif key == glfw.KEY_R:
            cam.__init__()

    def on_scroll(_w, _x, y):
        state["scroll"] += y

    glfw.set_key_callback(win, on_key)
    glfw.set_scroll_callback(win, on_scroll)

    prev = glfw.get_cursor_pos(win)
    t_last = t_fps = time.perf_counter()
    frames = fps_frames = 0
    while not glfw.window_should_close(win):
        glfw.poll_events()
        now = time.perf_counter()
        dt_frame, t_last = now - t_last, now
        ww, wh = glfw.get_window_size(win)
        fw, fh = glfw.get_framebuffer_size(win)
        if min(ww, wh, fw, fh) == 0:
            time.sleep(0.01)
            continue

        # --- input -> camera / brush (cursor in window coords, not framebuffer pixels)
        cur = glfw.get_cursor_pos(win)
        dx, dy = cur[0] - prev[0], cur[1] - prev[1]
        prev = cur
        shift = (
            glfw.get_key(win, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(win, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        lmb = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        rmb = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        mmb = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        if lmb and not shift:
            cam.orbit(-dx / ww * math.pi, dy / wh * math.pi)
        if (rmb and not shift) or mmb:
            k = 2.0 * cam.dist * math.tan(cam.fov_y / 2.0) / wh
            cam.pan(-dx * k, dy * k)
        if state["scroll"]:
            cam.zoom_by(0.9 ** state["scroll"])
            state["scroll"] = 0.0
        right, up, fwd = cam.basis()
        fly = 0.25 * dt_frame
        for key, vec in (
            (glfw.KEY_W, fwd),
            (glfw.KEY_S, -fwd),
            (glfw.KEY_D, right),
            (glfw.KEY_A, -right),
            (glfw.KEY_E, up),
            (glfw.KEY_Q, -up),
        ):
            if glfw.get_key(win, key) == glfw.PRESS:
                cam.target = (cam.target + vec * fly) % 1.0
        if state["auto"] and not lmb:
            cam.orbit(0.15 * dt_frame, 0.0)

        aspect = fw / fh
        ring = None
        if shift:
            rel = cam.focal_point_rel(cur[0] / ww, cur[1] / wh, aspect)
            if rel is not None:
                radius = 0.08
                a = np.linspace(0, 2 * math.pi, 49)
                pts = rel + radius * (np.cos(a)[:, None] * right + np.sin(a)[:, None] * up)
                col = (0.5, 0.5, 0.5)
                ring = [(*pts[i], *col) for k in range(48) for i in (k, k + 1)]
                if lmb or rmb:
                    world = (cam.target + rel) % 1.0
                    sim.brush(world, radius, 8.0 if lmb else -8.0)

        # --- frame
        context.set_physical_size(fw, fh)
        renderer.resize(fw, fh)
        try:
            tex = context.get_current_texture()
        except wgpu.DrawCancelled:
            time.sleep(0.01)
            continue
        enc = device.create_command_encoder()
        if not state["paused"]:
            n_cells = sim._write_uniforms(1.0)
            sim.encode_grid(enc, n_cells)
            sim.encode_forces(enc)
        renderer.encode(
            enc,
            tex.create_view(),
            sim.pos_a,
            sim.n,
            cam,
            point_size=args.size,
            fog=0.8,
            exposure=1.2,
            show_box=state["box"],
            ring=ring,
            slice_half=state["slice_half"] if state["slice"] else 10.0,
        )
        device.queue.submit([enc.finish()])
        context.present()

        frames += 1
        fps_frames += 1
        if now - t_fps >= 1.0:
            glfw.set_window_title(
                win, f"Particle Life 3D (prototype) - {fps_frames / (now - t_fps):.0f} FPS"
            )
            print(f"{fps_frames / (now - t_fps):.1f} FPS", flush=True)
            t_fps, fps_frames = now, 0
        if args.frames and frames >= args.frames:
            break
    context.unconfigure()
    glfw.terminate()


if __name__ == "__main__":
    main()
