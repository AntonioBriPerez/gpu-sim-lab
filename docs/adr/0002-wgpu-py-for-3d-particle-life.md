# 2. wgpu-py (WebGPU) for the 3D Particle Life

Date: 2026-10-04 · Status: proposed (accepted when `docs/plans/particle-life-3d/probe.py` passes on the reference machine)

## Context
We want a 3D, interactive version of `01-particle-life` that is as fast and as good-looking as the hardware allows, without leaving Python. Options considered:
- **Taichi 3D** (what 01 uses): least work, but development has slowed (last release 1.7.4, July 2025) and GGUI limits custom rendering.
- **NVIDIA Warp**: excellent simulation kernels, but rendering needs a second library and a GPU→renderer hand-off.
- **wgpu-py** (Python bindings to wgpu-native, WebGPU): compute and render shaders (WGSL) share the same GPU buffers, full control of the look; actively released (0.32.0, July 2026).
- **Rust + wgpu**: same engine as wgpu-py, much more work, breaks the Python/uv convention.

## Decision
A new standalone uv project `01b-particle-life-3d` built on wgpu-py 0.32.0 (pinned), a GLFW window (`glfw` 2.10.2) and the Tk control panel pattern from 01. Simulation (counting-sort grid, prefix sum, forces) and rendering (additive glow billboards, trails, tonemap) are WGSL shaders over the same storage buffers. 01 stays untouched. A verified prototype lives in `docs/plans/particle-life-3d/prototype/`.

## Consequences
- More hand-written GPU code (grid sort, scan) than with Taichi; mitigated by the verified prototype and NumPy reference tests.
- GPU tests run in CI on a software Vulkan driver (lavapipe from `mesa-vulkan-drivers`, installed on the GitHub runner only, as wgpu-py's own CI does). Performance is only measured on the reference machine.
- wgpu-py is pre-1.0: pin exact versions and upgrade deliberately. Known 0.32.0 issue: `queue.on_submitted_work_done_sync()` is broken; synchronise with `queue.read_buffer()`.
- Native Wayland presentation is untested so far; fallback is the X11 build of GLFW (XWayland) via `PYGLFW_LIBRARY_VARIANT=x11`.
- Second copy of `params.py`, `colors.py`, `reference.py` and `ui.py` (from 01). Extract a shared package only if a third simulator needs them (ADR 0001).
