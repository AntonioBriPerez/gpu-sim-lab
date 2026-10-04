# Prototipo (spike) de Particle Life 3D con wgpu-py

Código de referencia **ya verificado** que acompaña a [`../PLAN.md`](../PLAN.md). No es el simulador final: sirve para comprobar que el enfoque funciona y como punto de partida para portarlo a `01b-particle-life-3d/` (hito M1–M5 del plan). Se borra cuando el MVP esté aceptado (hito M6).

Verificado el 2026-10-04 en un contenedor sin GPU con Vulkan por software (lavapipe, Mesa 25.2.8) y Xvfb:
- 27 tests pasan (suma de prefijos hasta 1 048 576 elementos, ordenación por celdas, fuerzas contra la referencia NumPy de fuerza bruta, pincel con bordes periódicos, proyección de la cámara contra lo que pinta la GPU, estelas, corte, niebla, resaltado).
- Ventana GLFW + superficie wgpu + presentación + redimensionado + panel Tk en el mismo bucle (X11 vía Xvfb).
- **No verificado**: Wayland nativo ni rendimiento en una GPU real.

## Ficheros
| Fichero | Qué es | Se porta a |
|---|---|---|
| `shaders/common.wgsl` | `SimParams` + helpers de celda | `src/particle_life_3d/shaders/` |
| `shaders/grid.wgsl` | conteo por celda + rango atómico | igual |
| `shaders/scan.wgsl` | suma de prefijos exclusiva en dos niveles | igual |
| `shaders/scatter.wgsl` | reordenación A → B por celda | igual |
| `shaders/forces.wgsl` | fuerzas + integración (lee B, escribe A) | igual |
| `shaders/brush.wgsl` | pincel del ratón | igual |
| `shaders/render.wgsl` | halos aditivos, estelas, tonemap, líneas | igual |
| `gpu.py` | dispositivo, carga de WGSL, layouts explícitos | `gpu.py` |
| `sim3d.py` | buffers, pipelines y pasos de la simulación | `sim.py` + `uniforms.py` |
| `camera3d.py` | cámara orbital (NumPy puro) | `camera.py` |
| `render3d.py` | renderizador | `render.py` + `overlay.py` + `uniforms.py` |
| `test_proto.py`, `test_render_proto.py` | tests de GPU | `tests/` |
| `bench_proto.py` | benchmark sin ventana | `bench.py` |
| `demo_proto.py` | demo interactiva mínima (sin panel Tk) | `app.py` + `window.py` + `controls.py` |
| `snapshot.py` | render fuera de pantalla a PNG | (herramienta de depuración) |

## Cómo ejecutarlo
Los scripts llevan sus dependencias declaradas (PEP 723), así que `uv run` las instala solo:

```bash
cd docs/plans/particle-life-3d/prototype
uv run demo_proto.py                               # demo 3D interactiva (200 000 partículas)
uv run demo_proto.py --particles 1000000 --r-max 0.03 --no-vsync
uv run bench_proto.py                              # tabla pasos/s y ms por fase en GPU
uv run --no-project --with wgpu==0.32.0 --with "numpy>=2.0" --with pytest pytest -q
```

Controles de la demo: arrastre izquierdo = orbitar, derecho/central = desplazar, rueda = zoom, Shift+izquierdo/derecho = atraer/repeler, W/S/A/D/Q/E = volar (el mundo es periódico), O = auto-rotación, F = corte, +/- = grosor del corte, B = cubo, R = reset, Espacio = pausa. Si Wayland nativo falla: `--x11`.

En una máquina sin GPU (CI, contenedor en la nube) hace falta Vulkan por software: ver la sección «Entorno del agente» del plan.
