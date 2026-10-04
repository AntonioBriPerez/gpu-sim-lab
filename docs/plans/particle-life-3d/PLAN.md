# Plan técnico — Particle Life 3D con wgpu-py (`01b-particle-life-3d`)

Fecha: 2026-10-04 · Estado: **pendiente de aprobación del usuario** · Decisión de arquitectura: [ADR 0002](../../adr/0002-wgpu-py-for-3d-particle-life.md)

Este documento está pensado para dos lectores: el usuario (para aprobarlo) y el agente que lo implemente. El prototipo verificado está en [`prototype/`](prototype/) y la sonda de entorno en [`probe.py`](probe.py).

---

## 0. Instrucciones para el agente que implementa

1. Lee entero este plan, el ADR 0002, [`prototype/README.md`](prototype/README.md) y el `CLAUDE.md` del repo antes de tocar nada.
2. **Porta el prototipo; no lo rediseñes.** Los shaders WGSL del prototipo están verificados contra NumPy. Si te desvías de algo, explica el motivo en el mensaje del commit.
3. Implementa los hitos (§14) **en orden**. Cada hito son 1–3 commits pequeños (Conventional Commits) y cada commit deja `pytest`, `ruff check`, `ruff format --check` y `pyright` en verde.
4. En los puntos 🔶 **para y pide al usuario** que ejecute algo en su máquina. Tú no tienes GPU real: no inventes cifras de rendimiento.
5. No instales paquetes del sistema en la máquina del usuario. En tu contenedor, usa lavapipe en espacio de usuario (§16).
6. No subas nada pesado (capturas, vídeos, `.npy`). Las salidas van a `outputs/`, que ya está en `.gitignore`.

---

## 1. Resumen

Variante 3D de `01-particle-life`: N partículas de T tipos en un **cubo unidad periódico** (toroide 3D), con la misma regla de fuerzas y la misma matriz de atracción. La simulación y el dibujo son shaders WGSL sobre los mismos buffers de GPU (wgpu-py → wgpu-native → Vulkan). La ventana es GLFW y los controles, el panel Tk de 01 adaptado.

**Qué está verificado y qué no** (contenedor sin GPU, lavapipe = Vulkan por software, Xvfb):

| Aspecto | Estado |
|---|---|
| Rejilla por celdas (conteo atómico + suma de prefijos en dos niveles + reordenación) | ✅ tests contra NumPy, hasta 1 048 576 celdas |
| Fuerzas + integración 3D contra la referencia de fuerza bruta | ✅ `rtol=1e-3` con r = 0.1, 0.2 y 0.33 (27 tests en total, ~3 s) |
| Pincel con bordes periódicos | ✅ |
| Renderizado: proyección, envoltura periódica, resaltado, estelas, corte, niebla, tamaño mínimo | ✅ píxel a píxel contra la cámara NumPy |
| Ventana GLFW + superficie wgpu + presentar + redimensionar + Tk en el mismo bucle | ✅ en X11 (Xvfb) |
| Timestamps de GPU por fase | ✅ (lavapipe los soporta; NVIDIA también debería) |
| **Wayland nativo** | ❌ sin probar → `probe.py` en tu máquina (§14, P0) |
| **Rendimiento real** | ❌ sin medir → `bench` en tu máquina (§14, M2) |

---

## 2. Decisiones cerradas

| Tema | Decisión | Motivo |
|---|---|---|
| Carpeta / paquete | `01b-particle-life-3d/`, paquete `particle_life_3d` | Variante de 01; ordena justo después |
| Python | `.python-version` = `3.13`, `requires-python = ">=3.13"`, `[tool.uv] python-preference = "only-managed"` | El CPython de uv incluye tkinter; los Python del sistema pueden no tenerlo (verificado: el `python3.13` de Ubuntu del contenedor no lo trae) |
| Dependencias | `wgpu==0.32.0` (PyPI, 19-07-2026), `glfw==2.10.2` (PyPI, 21-07-2026), `numpy>=2.0`. Dev: `pytest>=8`, `ruff>=0.8`, `pyright>=1.1` | Versiones comprobadas en PyPI el 2026-10-04. wgpu es 0.x: versión exacta |
| Ventana | GLFW directo + `wgpu.utils.glfw_present_info.get_glfw_present_info` (no se usa la API de `rendercanvas`, aunque es dependencia transitiva de wgpu) | Bucle manual igual que el de 01 (`while: poll; panel.update(); draw`) |
| Panel | Tk, copiado de 01 y adaptado | Escala bien en HiDPI; ya lo conoces |
| Mundo | Cubo `[0,1)^3` periódico; se dibuja la "celda de vista" de lado 1 centrada en el objetivo de la cámara | Se puede volar sin fin por un universo periódico |
| Aspecto | Halos gaussianos aditivos en HDR (`rgba16float`) + tonemap `1-exp(-c·exposure)`, igual que en 01 | Conserva tu estética; la mezcla aditiva no necesita ordenar por profundidad |
| CI | GitHub `ubuntu-latest` + `apt-get install mesa-vulkan-drivers` (lavapipe) solo en el runner | Los tests de GPU corren en CI con CPU, como hace el propio CI de wgpu-py |
| 01 | No se toca | |

Prerrequisito según tu `CLAUDE.md`: el README dice que el MVP de 01 está «pendiente de tu prueba». Conviene marcarlo como probado (o decidir explícitamente que 01b va antes) antes de empezar M0.

---

## 3. Arquitectura

```
                    ┌──────────────────────────── un paso de simulación ───────────────────────────┐
 estado A ──count──▶ cell_count, cell_of, rank ──scan(2 niveles)──▶ cell_start ──scatter──▶ estado B
 (pos_a, vel_a)                                                                         (ordenado por celda)
      ▲                                                                                        │
      └─────────────────────────────── forces.step (lee B, escribe A) ◀────────────────────────┘

 por fotograma:  [pasos × k] → [pincel] → pase A: (fade) + halos aditivos → accum rgba16float
                                         → pase B: tonemap + líneas (cubo, anillo) → swapchain bgra8unorm
```

- **Ping-pong A/B**: el estado vive en A. `scatter` copia A→B en orden de celda y `forces.step` lee B (vecinos contiguos en memoria) y escribe el estado nuevo en A, en ese mismo orden. Nunca lee lo que escribe, así que no hay carreras. Tras cada paso el estado vuelve a estar en A, ya ordenado: el siguiente paso apenas reordena.
- **Una sola lectura por vecino**: `vec4<f32>` con xyz = posición y w = tipo (como float). El bucle de vecinos lee 16 bytes por candidato.
- Todo un fotograma va en **un encoder y un `submit`**. Python solo escribe uniforms pequeños y codifica.

---

## 4. Estructura del proyecto

```
01b-particle-life-3d/
├── .python-version                 # 3.13
├── pyproject.toml                  # §4.1
├── uv.lock
├── README.md                       # español, como el de 01 (M6)
├── presets/.gitkeep                # presets/user/ en .gitignore
├── src/particle_life_3d/
│   ├── __init__.py
│   ├── params.py      # de 01 + límites 3D (§10)                         puro
│   ├── colors.py      # copia literal de 01                               puro
│   ├── reference.py   # copia de 01 (ya vale para N dimensiones)           puro
│   ├── uniforms.py    # pack_* + tamaños; espejo byte a byte de los structs WGSL   puro
│   ├── camera.py      # OrbitCamera (de prototype/camera3d.py + reset/fly/proj_scale)  puro
│   ├── overlay.py     # vértices del cubo y del anillo del pincel         puro
│   ├── controls.py    # entrada → cámara/pincel (InputState → acciones)   puro
│   ├── gpu.py         # dispositivo, carga WGSL, pipelines y bind groups explícitos
│   ├── sim.py         # ParticleLife3D (de prototype/sim3d.py)
│   ├── render.py      # GlowRenderer3D + Look (de prototype/render3d.py)
│   ├── window.py      # GLFW + GPUCanvasContext (fino; sin tests en CI)
│   ├── ui.py          # panel Tk de 01 adaptado (§9)
│   ├── app.py         # CLI + bucle principal
│   ├── bench.py       # benchmark sin ventana (de prototype/bench_proto.py)
│   └── shaders/       # common, grid, scan, scatter, forces, brush, render (.wgsl, copiados del prototipo)
└── tests/
    ├── conftest.py                       # fixture `device` (§11.1)
    ├── test_params.py  test_uniforms.py  test_camera.py  test_overlay.py  test_controls.py   # puros
    └── test_scan.py    test_grid.py      test_sim.py     test_render.py                      # wgpu
```

Separación que pide `CLAUDE.md`: la lógica pura (`params`, `uniforms`, `camera`, `overlay`, `controls`) no importa wgpu ni glfw. La simulación (`sim`) no sabe nada de ventanas. El renderizado (`render`) y la entrada (`window`, `controls`) van en módulos aparte.

### 4.1 `pyproject.toml`
```toml
[project]
name = "particle-life-3d"
version = "0.1.0"
description = "3D Particle Life on the GPU with wgpu-py (WebGPU)"
requires-python = ">=3.13"
dependencies = ["wgpu==0.32.0", "glfw==2.10.2", "numpy>=2.0"]

[project.scripts]
particle-life-3d = "particle_life_3d.app:main"
particle-life-3d-bench = "particle_life_3d.bench:main"

[dependency-groups]
dev = ["pytest>=8", "ruff>=0.8", "pyright>=1.1"]

[tool.uv]
python-preference = "only-managed"  # uv's CPython ships tkinter; distro Pythons may not

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/particle_life_3d"]   # includes shaders/*.wgsl

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "NPY"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["wgpu: needs a WebGPU adapter (lavapipe in CI, the real GPU locally)"]

[tool.pyright]
include = ["src", "tests"]
typeCheckingMode = "basic"
```

---

## 5. Memoria de GPU

### 5.1 Constantes
| Constante | Valor | Dónde | Por qué |
|---|---|---|---|
| `WG` | 256 | WGSL y Python | tamaño de workgroup de todos los kernels 1D |
| `BLOCK` | 1024 | `scan.wgsl`, `sim.py` | 256 hilos × 4 elementos por bloque de suma de prefijos |
| `NC_MAX` | 100 | `sim.py` | 100³ = 1 000 000 celdas ≤ 1024² (límite del scan en dos niveles) |
| `MAX_CELLS` | 1 000 000 | `sim.py` | tamaño fijo de los buffers de celdas: cambiar el radio no realoca |
| `MAX_TYPES` | 16 | `forces.wgsl`, `params.py`, `RenderParams` | la matriz cabe en memoria de workgroup (256 floats) |
| `R_MIN` | 0.01 | `params.py` | ⌊1/0.01⌋ = 100 = `NC_MAX` |
| `MAX_PARTICLES` | 8 000 000 | `params.py` | además se comprueba `16·N ≤ device.limits["max-storage-buffer-binding-size"]` |

Celdas por lado: `nc = min(NC_MAX, max(3, int(1 / r_max)))`. Con menos de 3 celdas, los vecinos periódicos se repetirían.

### 5.2 Buffers (N partículas)
Todos los de almacenamiento llevan `STORAGE | COPY_SRC | COPY_DST`: `clear_buffer` exige `COPY_DST` y `read_buffer` exige `COPY_SRC`.

| Buffer | Tipo WGSL | Bytes | Escribe | Lee |
|---|---|---|---|---|
| `pos_a` | `array<vec4<f32>>` (xyz ∈ [0,1), w = tipo) | 16N | `forces.step`, `reset`/`write_state` | `count`, `scatter`, `brush`, render VS |
| `vel_a` | `array<vec4<f32>>` (w = 0) | 16N | `forces.step`, `brush` | `scatter` |
| `pos_b`, `vel_b` | ídem, ordenados por celda | 16N c/u | `scatter` | `forces.step`; tests |
| `cell_of`, `rank` | `array<u32>` | 4N c/u | `count` | `scatter` |
| `cell_count` | `array<atomic<u32>>` en `count`, `array<u32>` en el resto | 4·MAX_CELLS | `clear_buffer` + `count` | scan nivel 1, `forces.step` |
| `cell_start` | `array<u32>` | 4·MAX_CELLS | scan nivel 1 + `add_offsets` | `scatter`, `forces.step` |
| `block_sums` / `block_offsets` | `array<u32>` | 4·1024 c/u | scan nivel 1 / nivel 2 | scan nivel 2 / `add_offsets` |
| `scan_total` | `array<u32>` | 16 | scan nivel 2 | sincronización (`read_buffer`) y depuración |
| `matrix` | `array<f32>` | 4·256 | `set_matrix` | `forces.step` |
| `u_sim` | `uniform SimParams` | 48 | `encode_steps` | `count`, `scatter`, `forces.step` |
| `u_scan_cells` / `u_scan_blocks` | `uniform ScanParams` | 16 c/u | `encode_steps` | scan nivel 1 + `add_offsets` / scan nivel 2 |
| `u_brush` | `uniform BrushParams` | 32 | `encode_brush` | `brush` |
| `u_render` | `uniform RenderParams` | 384 | `GlowRenderer3D.encode` | todos los pipelines de render |
| `accum` | textura `rgba16float` (`RENDER_ATTACHMENT \| TEXTURE_BINDING`) | 8·W·H | pase A | tonemap |
| `line_vbo` | vértices (6 × f32) | 24·512 | `encode` | pipeline de líneas |

Memoria: 1M partículas ≈ 64 MB de estado + ~20 MB de rejilla. 8M ≈ 600 MB, de sobra para 16 GB.

### 5.3 Uniforms (byte a byte; `uniforms.py` debe reproducirlo exactamente)

`SimParams` — 48 bytes (`common.wgsl`):
| off | campo | tipo |
|---|---|---|
| 0 | `n` | u32 |
| 4 | `nc` | u32 |
| 8 | `n_types` | u32 |
| 12 | `_pad0` | u32 |
| 16 | `r_max` | f32 |
| 20 | `beta` | f32 |
| 24 | `force_scale` | f32 |
| 28 | `dt` | f32 (ya multiplicado por `speed`) |
| 32 | `friction` | f32 = `0.5 ** (dt / friction_half_life)` |
| 36–47 | `_pad1.._pad3` | f32 |

`ScanParams` — 16 bytes: `n: u32` + 3 × u32 de relleno.

`BrushParams` — 32 bytes: `center: vec3<f32>` @0, `radius` @12, `strength` @16 (con signo: + atrae, − repele), `dt` @20, `n: u32` @24, `_pad0: u32` @28.

`RenderParams` — 384 bytes (`render.wgsl`):
| off | campo | notas |
|---|---|---|
| 0 | `view_proj: mat4x4<f32>` | subir `M.T` (WGSL es column-major) |
| 64 | `center: vec3<f32>` | objetivo de la cámara en el mundo. **No llamarlo `target`: es palabra reservada en WGSL** |
| 76 | `point_size: f32` | radio del halo en unidades de mundo |
| 80 | `fwd: vec3<f32>` | dirección de vista unitaria |
| 92 | `slice_half: f32` | semigrosor del corte; 10.0 = sin corte |
| 96 | `viewport: vec2<f32>` | tamaño del framebuffer en píxeles |
| 104 | `proj_scale: f32` | `height / (2·tan(fov/2))` |
| 108 | `fog: f32` | atenuación por profundidad (0 = nada) |
| 112 | `exposure: f32` | brillo |
| 116 | `min_px: f32` | radio mínimo del halo en píxeles (0.75, ver §15.14) |
| 120 | `_pad0, _pad1: f32` | |
| 128 | `colors: array<vec4<f32>, 16>` | rgb = color del tipo × ganancia de resaltado |

---

## 6. Simulación

### 6.1 Secuencia de un paso (`ParticleLife3D.encode_steps`)
`n_cells = nc³`. Por cada paso:

| # | Dónde | Operación | Dispatch | Lee | Escribe |
|---|---|---|---|---|---|
| 1 | encoder | `clear_buffer(cell_count, 0, 4·n_cells)` | — | | `cell_count` |
| 2 | pase «grid» | `count` | ⌈N/256⌉ | `u_sim`, `pos_a` | `cell_count` (atómico), `cell_of`, `rank` |
| 3 | | `scan_blocks` (nivel 1) | ⌈n_cells/1024⌉ | `u_scan_cells`, `cell_count` | `cell_start` (parcial), `block_sums` |
| 4 | | `scan_blocks` (nivel 2) | 1 | `u_scan_blocks`, `block_sums` | `block_offsets`, `scan_total` |
| 5 | | `add_offsets` | ⌈n_cells/256⌉ | `u_scan_cells`, `block_offsets` | `cell_start` |
| 6 | | `scatter` | ⌈N/256⌉ | `u_sim`, `pos_a`, `vel_a`, `cell_of`, `rank`, `cell_start` | `pos_b`, `vel_b` |
| 7 | pase «forces» | `step` | ⌈N/256⌉ | `u_sim`, `pos_b`, `vel_b`, `cell_start`, `cell_count`, `matrix` | `pos_a`, `vel_a` |

WebGPU pone barreras entre dispatches de un mismo pase, así que todo cabe en dos pases. El dispatch máximo es ⌈8M/256⌉ = 31 250, por debajo del límite de 65 535.

Los uniforms se escriben con `queue.write_buffer` **una vez por `encode_steps`**. `write_buffer` se aplica en el siguiente `submit`, así que todos los pasos codificados en un mismo `submit` comparten `dt`. No llames dos veces a `encode_steps` con distinto `dt_scale` antes del mismo `submit`.

### 6.2 Kernels (WGSL de referencia en `prototype/shaders/`)
- **`count`** (`grid.wgsl`): `c = cell_index(cell_coord(pos.xyz))`; `cell_of[i] = c`; `rank[i] = atomicAdd(&cell_count[c], 1)`. Guardar el rango ahora evita una segunda pasada atómica en `scatter`.
- **`scan_blocks`** (`scan.wgsl`): cada hilo suma 4 elementos en local (exclusiva). Hillis–Steele sobre los 256 totales en memoria de workgroup (8 rondas con `workgroupBarrier()` en flujo uniforme). Escribe la suma exclusiva y el total del bloque. La misma entrada sirve para el nivel 2 sobre `block_sums` (≤ 1024 bloques → 1 workgroup).
- **`add_offsets`**: `dst[i] += block_sums[i / 1024]`, donde la binding 3 apunta a `block_offsets`. Usa las bindings {0, 2, 3}; `scan_blocks`, {0, 1, 2, 3}.
- **`scatter`** (`scatter.wgsl`): `dst = cell_start[cell_of[i]] + rank[i]`; copia `pos` y `vel` de A a B.
- **`step`** (`forces.wgsl`):
  1. Carga la matriz `n_types²` en `var<workgroup> m: array<f32, 256>` y luego `workgroupBarrier()`. Esto va **antes** del `if (i >= n) return`.
  2. Recorre las 27 celdas vecinas con envoltura `(c + d + nc) % nc` y, en cada una, el rango `[cell_start, cell_start + cell_count)`.
  3. Imagen mínima `d -= round(d)`; si `0 < |d|² < r_max²`: `acc += d · force(|d|/r_max, m[row + tipo_j], beta) / |d|`.
  4. `acc *= r_max · force_scale`; `v = v·friction + acc·dt`; `q = fract(p + v·dt)`, forzando `q = 0` si `fract` devuelve 1.0.
  5. Escribe `pos_a[i] = (q, tipo)` y `vel_a[i] = (v, 0)`.

  La función `force` es idéntica a `_force` de 01 y a `reference.force_profile`.
- **`brush`** (`brush.wgsl`): igual que en 01 pero en 3D: `d = center − p` con imagen mínima; si `0 < |d| < radius`, `v += d/|d| · strength · (1 − |d|/radius) · dt`. Lee `pos_a` y modifica `vel_a`, en el mismo encoder y después de los pasos.

### 6.3 Determinismo
El orden en que `atomicAdd` reparte los rangos no es determinista en GPU, y con él tampoco el orden de las partículas dentro de una celda ni las sumas en coma flotante. Por eso los tests usan tolerancias e invariantes, no reproducibilidad bit a bit. En 01 se lograba con Taichi en CPU de un hilo; aquí no hay equivalente.

### 6.4 API de `sim.py`
```python
class ParticleLife3D:
    def __init__(self, device: wgpu.GPUDevice, params: Params) -> None: ...
        # allocates every buffer of §5.2, builds pipelines with explicit layouts and the
        # bind groups once; raises ValueError if 16*N > max-storage-buffer-binding-size
    params: Params
    n: int
    pos_buffer: wgpu.GPUBuffer          # = pos_a (renderer input)
    def reset(self, seed: int) -> None   # uniform positions, random types, v = 0
    def set_matrix(self, matrix: np.ndarray) -> None   # also sets params.matrix
    def grid_cells_per_side(self) -> int
    def encode_steps(self, encoder, steps: int = 1, dt_scale: float = 1.0,
                     query_set: wgpu.GPUQuerySet | None = None) -> None
        # query_set (4 timestamps: grid begin/end, forces begin/end) only with steps == 1
    def encode_brush(self, encoder, center: np.ndarray, radius: float, strength: float,
                     dt_scale: float = 1.0) -> None
    def step(self, steps: int = 1, dt_scale: float = 1.0) -> None   # encoder + submit
    # tests / debug only:
    def write_state(self, pos: np.ndarray, vel: np.ndarray, types: np.ndarray) -> None
    def read_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]     # pos (N,3), vel (N,3), types (N,)
    def read_sorted(self) -> tuple[np.ndarray, np.ndarray]                # buffer B: pos (N,3), types (N,)
```
`gpu.py`:
```python
def request_device(power_preference="high-performance", features=()) -> wgpu.GPUDevice
    # requests the adapter's max storage-binding and buffer sizes; clear error if there is no adapter
def wgsl(*names: str) -> str          # importlib.resources: "common" + "grid" ... concatenated
def compute_pipeline(device, code: str, entry: str, kinds: dict[int, str]) -> tuple[pipeline, layout]
    # kinds: binding -> "uniform" | "read" | "rw"   (explicit layouts, never layout="auto")
def bind_buffers(device, layout, buffers: dict[int, wgpu.GPUBuffer]) -> wgpu.GPUBindGroup
def sync(device, buffer) -> None       # queue.read_buffer(buffer, 0, 4): waits for the queue
```

---

## 7. Renderizado

### 7.1 Coordenadas
- Mundo: `[0,1)^3` periódico. La cámara mira a su objetivo `center` (∈ [0,1)^3).
- Celda de vista: `rel = p − center; rel −= round(rel)`, así que `rel ∈ [−0.5, 0.5]^3`. En estas coordenadas el objetivo es el origen, y el cubo dibujado siempre es `[−0.5, 0.5]^3`. Mover `center` (desplazar o volar) recorre el universo periódico sin bordes.

### 7.2 Pases por fotograma
| Pase | Destino | load_op | Pipelines |
|---|---|---|---|
| A | `accum` (`rgba16float`) | `"load"` si `trail > 0`, si no `"clear"` | `fade` (`draw(3)`, solo si `trail > 0`), después `particles` (`draw(4, N)`) |
| B | textura del swapchain (o una offscreen `rgba8unorm` en los tests) | `"clear"` negro | `tonemap` (`draw(3)`), después `lines` (`draw(V)`) |

### 7.3 Pipelines (`render.wgsl`; layouts explícitos)
| Pipeline | VS / FS | Topología | Destino | Blend (color y alfa) | Bindings |
|---|---|---|---|---|---|
| `particles` | `vs_particle` / `fs_particle` | `triangle-strip`, 4 vértices por instancia | `rgba16float` | `src one, dst one, add` | 0 uniform (VS\|FS), 1 `read-only-storage` (VS) = `pos_a` |
| `fade` | `vs_fullscreen` / `fs_fade` | `triangle-list` | `rgba16float` | `src zero, dst constant, add` + `set_blend_constant((t,t,t,t))` | ninguno (layout vacío) |
| `tonemap` | `vs_fullscreen` / `fs_tonemap` | `triangle-list` | formato del swapchain | sin blend | 0 uniform, 2 `texture_2d<f32>` (`sample_type "float"`) |
| `lines` | `vs_line` / `fs_line` | `line-list` | formato del swapchain | `src one, dst one, add` | 0 uniform; vertex buffer stride 24: `float32x3` pos @0, `float32x3` color @12 |

### 7.4 Halo de cada partícula (`vs_particle`)
```
corner = (vi & 1, vi >> 1)·2 − 1                 # (−1,−1) (1,−1) (−1,1) (1,1)
rel = wrap(p.xyz − center); depth = dot(rel, fwd)
c = view_proj · (rel, 1)
if |depth| > slice_half  or  c.z < 0:  → posición fuera del volumen de recorte (descartada)
px = point_size · proj_scale / c.w                # radio en píxeles
s = max(px, min_px);  energy = (px / s)²          # no parpadea por debajo de 1 px y conserva la energía
fade = exp(−fog · clamp(depth + 0.5, 0, 1.5))     # lo lejano, más tenue
clip = (c.xy + corner · 2s / viewport · c.w, c.z, c.w)
color = colors[type] · energy · fade
fs: r² = |uv|²; discard si r² > 1; salida = color · exp(−4 r²)
```

### 7.5 Estelas, tonemap y formato de salida
- Estelas: el pase A no se limpia y antes de los halos se dibuja un triángulo a pantalla completa con blend `dst · constant`, con constante = `trail`. Sin shader de lectura.
- Tonemap: `textureLoad(accum, frag.xy)·exposure → 1 − exp(−c)`.
- El formato preferido de la superficie es `bgra8unorm-srgb`. Usa **`bgra8unorm`** (quita `-srgb`) para que el resultado se vea como en 01. Está verificado que la superficie lo admite.

### 7.6 Superposiciones (`overlay.py`, puro)
```python
def cube_wireframe(color=(0.18, 0.18, 0.22)) -> np.ndarray      # (24, 6) float32: xyz rgb, corners ±0.5
def ring(center_rel, right, up, radius, color=(0.5, 0.5, 0.5), segments=48) -> np.ndarray  # (96, 6)
```
El anillo del pincel se dibuja en el plano focal, centrado en `focal_point_rel`, orientado con `right` y `up` de la cámara. `MAX_LINE_VERTICES = 512`.

### 7.7 API de `render.py`
```python
@dataclass
class Look:
    point_size: float = 0.0035
    exposure: float = 1.2
    trail: float = 0.0
    fog: float = 0.8
    slice_half: float | None = None    # None = no slice (packed as 10.0)
    show_box: bool = True
    min_px: float = 0.75

class GlowRenderer3D:
    def __init__(self, device, out_format: str, n_types: int) -> None
    def set_highlight(self, type_id: int | None) -> None     # DIM_GAIN = 0.12 like 01
    def resize(self, width: int, height: int) -> None        # no-op if unchanged; recreates accum + tonemap bind group
    def encode(self, encoder, out_view, pos_buffer, n: int, cam: OrbitCamera, look: Look,
               ring: np.ndarray | None = None) -> None
```
El bind group de partículas se cachea por `pos_buffer`. Recréalo solo si cambia (por ejemplo, al cargar otro preset).

---

## 8. Cámara, interacción y ventana

### 8.1 `OrbitCamera` (puro; de `prototype/camera3d.py`)
- Estado: `yaw=35°`, `pitch=25°`, `dist=2.0`, `target=(0.5, 0.5, 0.5)`, `fov_y=50°`. Límites: `pitch ∈ ±89°`, `dist ∈ [0.05, 6]` (con dist < 0.5 la cámara queda dentro del cubo dibujado).
- `basis()`: `to_eye = (cos p·sin y, sin p, cos p·cos y)`, `fwd = −to_eye`, `right = normalize(fwd × ŷ)`, `up = right × fwd`.
- `eye() = −fwd·dist` (coordenadas de la celda de vista). `near = 0.005`, `far = dist + 2`.
- `view_proj(aspect)` = perspectiva RH con z ∈ [0, 1] (`[[f/a,0,0,0],[0,f,0,0],[0,0,far/(near−far),near·far/(near−far)],[0,0,−1,0]]`) · `look_at` (filas `right`, `up`, `−fwd`; traslación `−R·eye`).
- `ray(sx, sy, aspect)` con `sy` hacia abajo (como GLFW); `focal_point_rel()` es el corte del rayo con el plano que pasa por el objetivo, o `None` si queda fuera de `|x| ≤ 0.5`; `focal_point()` lo devuelve en coordenadas de mundo envueltas; `project(world, w, h)` sirve para tests y superposiciones.
- Añadir en el port: `reset()`, `fly(d_right, d_up, d_fwd)` (mueve `target` y envuelve) y `proj_scale(height)`.

### 8.2 Controles (ventana GLFW)
| Entrada | Acción |
|---|---|
| Arrastrar con botón izquierdo | orbitar: `yaw −= dx/ancho·π`, `pitch += dy/alto·π` |
| Arrastrar con botón derecho o central | desplazar el objetivo en el plano de pantalla: `k = 2·dist·tan(fov/2)/alto` |
| Rueda | `dist *= 0.9^ticks` |
| Shift + izquierdo / derecho | pincel atrae / repele en `focal_point` (anillo visible mientras Shift está pulsado) |
| W/S, A/D, Q/E | volar adelante/atrás, izquierda/derecha, abajo/arriba (0.25 unidades/s); el mundo envuelve. En 01, Q/E eran zoom |
| O | auto-rotación sí/no (velocidad en el panel) |
| F, + / − | corte sí/no y grosor del corte |
| B | cubo sí/no |
| R | restablecer la vista |
| Espacio | pausa |

Las posiciones del cursor van en **coordenadas de ventana**: normaliza con `glfw.get_window_size()`, no con el tamaño del framebuffer (HiDPI).

### 8.3 `controls.py` (puro, testeable)
```python
@dataclass(frozen=True)
class InputState:
    cursor: tuple[float, float]           # window coordinates
    window_size: tuple[int, int]
    framebuffer_size: tuple[int, int]
    left: bool
    right: bool
    middle: bool
    shift: bool
    scroll: float                         # wheel ticks since the previous frame
    held: frozenset[str]                  # subset of {"w", "a", "s", "d", "q", "e"}
    pressed: tuple[str, ...]              # keys pressed this frame: "o", "f", "b", "r", "space", "plus", "minus"

@dataclass(frozen=True)
class BrushAction:
    center_rel: np.ndarray                # view-cell coords (for the ring)
    center: np.ndarray                    # world coords in [0,1)^3 (for the kernel)
    sign: float                           # +1 pull, -1 push, 0 = only show the ring

class CameraController:
    def update(self, cam: OrbitCamera, inp: InputState, dt: float,
               auto_rotate_rad_s: float) -> BrushAction | None
```
Las teclas de alternar (`pressed`) las gestiona `app.py`, cambiando las variables del panel para que este refleje el estado, igual que `nudge` en 01.

### 8.4 `window.py`
```python
class Window:
    def __init__(self, device, title: str, size: tuple[int, int], vsync: bool) -> None
    platform: str          # "wayland" | "x11" (from present_info)
    format: str            # "bgra8unorm"
    def poll(self) -> InputState        # glfw.poll_events + snapshot; scroll and keys via callbacks
    def should_close(self) -> bool
    def acquire(self) -> wgpu.GPUTexture | None   # None if minimised or wgpu.DrawCancelled
    def present(self) -> None
    def set_title(self, title: str) -> None
    def close(self) -> None             # context.unconfigure(); glfw.terminate()
```
Secuencia verificada (`prototype/demo_proto.py`, `probe.py`):
`glfw.init()` → `glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)` → `create_window` → `info = get_glfw_present_info(win, vsync=vsync)` → `context = wgpu.gpu.get_canvas_context(info)` → `fmt = context.get_preferred_format(adapter).removesuffix("-srgb")` → `context.configure(device=device, format=fmt)`.

Cada fotograma: `set_physical_size(*framebuffer_size)` → `get_current_texture()` (capturar `wgpu.DrawCancelled`) → … → `context.present()`.

- **Wayland/X11**: pyGLFW carga su build de Wayland si `XDG_SESSION_TYPE=wayland`, y `get_glfw_present_info` entonces da `platform="wayland"`. La opción `--x11` pone `os.environ["PYGLFW_LIBRARY_VARIANT"] = "x11"` **antes de importar glfw**: `app.main` importa `window` de forma perezosa, después de parsear los argumentos.
- **vsync**: va dentro de `present_info` (`fifo` si está activo; si no, `immediate` > `mailbox` > `fifo`).

### 8.5 Bucle principal (`app.py`)
```python
def run(self, bench_frames: int | None, max_frames: int | None) -> None:
    while not self.window.should_close() and not self.panel.closed:
        inp = self.window.poll()
        self.panel.update()
        self.panel.apply_to(self.sim.params)
        self._handle_toggles(inp.pressed)  # O F B R space + -
        brush = self.controller.update(self.cam, inp, frame_dt, auto_rotate)
        tex = self.window.acquire()
        if tex is None:
            continue
        self.renderer.resize(*inp.framebuffer_size)
        enc = self.device.create_command_encoder()
        if not paused or step_requested:
            self.sim.encode_steps(enc, steps=steps_per_frame, dt_scale=speed)
        if brush and brush.sign:
            self.sim.encode_brush(enc, brush.center, radius, brush.sign * strength, speed)
        self.renderer.encode(
            enc,
            tex.create_view(),
            self.sim.pos_buffer,
            self.sim.n,
            self.cam,
            look,
            ring=overlay.ring(...) if brush else None,
        )
        self.device.queue.submit([enc.finish()])
        self.window.present()
        # FPS in the window title every second; --bench-frames: WARMUP_FRAMES = 30, then
        # gpu.sync() and print FPS like 01; --frames N: quit after N frames (smoke tests)
```
Cargar un preset con otro `n_types` recrea el simulador, el renderizador y la matriz del panel, igual que `_load_next` en 01.

CLI: `--particles` (por defecto, el de `Params`), `--types 6`, `--seed 0`, `--r-max`, `--preset`, `--size 1600 1000`, `--ui-font 14`, `--no-vsync`, `--bench-frames N`, `--frames N`, `--x11`.

---

## 9. Panel Tk (`ui.py`, desde 01)
Se mantienen la estructura, la matriz con cabeceras de color, los botones (random, mutate, save, load next, help), el resaltado por tipo y el escalado de fuente. Cambios:

| Sección | Control | Rango | Paso | Por defecto |
|---|---|---|---|---|
| Simulation | radius | 0.01 – 0.333 | 0.001 | `Params` |
| | repulsion core (beta), force, friction half-life, speed | como en 01 | | |
| | **steps per frame** (nuevo) | 1 – 8 | 1 | 1 |
| Look | **particle size** (×0.001 mundo) | 0.5 – 15 | 0.1 | 3.5 |
| | brightness | 0.1 – 5 | 0.05 | 1.2 |
| | trail | 0 – 0.98 | 0.01 | 0 |
| | **depth fog** | 0 – 3 | 0.05 | 0.8 |
| View (sustituye zoom y view x/y) | **auto-rotate (°/s)** + casilla «auto-rotate (O)» | −30 – 30 | 0.5 | 8, activada |
| | **slice width** + casilla «slice (F)» | 0.01 – 0.5 | 0.005 | 0.08, desactivado |
| | casilla **box (B)**; botón «reset view (R)» | | | activada |
| Brush & mutation | brush radius | 0.01 – 0.3 | 0.005 | 0.08 |
| | brush strength, mutation amount | como en 01 | | |

`HELP_TEXT` se reescribe con los controles 3D de §8.2. Ya no se importa `ZOOM_MIN`/`ZOOM_MAX`.

---

## 10. Parámetros y presets (`params.py`)
Cambios respecto a 01: `R_MIN = 0.01`, `MAX_TYPES = 16`, `MAX_PARTICLES = 8_000_000`. Valores por defecto provisionales: `n_particles = 200_000`, `r_max = 0.045` (≈ 76 vecinos por partícula). Se fijan con las cifras de M2. `validate()` añade `1 ≤ n_types ≤ MAX_TYPES` y `n_particles ≤ MAX_PARTICLES`.

El formato JSON de los presets es el mismo que en 01: un preset 2D se puede cargar si `r_max ≥ 0.01` y `n_types ≤ 16`; si no, se muestra el error y se ignora.

---

## 11. Tests

### 11.1 Estrategia
- **Puros** (sin wgpu): siempre, en todas partes.
- **wgpu** (`pytestmark = pytest.mark.wgpu`): en CI corren sobre lavapipe y en tu máquina sobre la RTX. Usan N pequeño; el conjunto completo tarda unos 3 s en lavapipe.
- `conftest.py`:
```python
@pytest.fixture(scope="session")
def device():
    try:
        return request_device()
    except Exception as exc:
        if os.environ.get("REQUIRE_WGPU") == "1":  # set in CI: never skip silently there
            raise
        pytest.skip(f"no WebGPU adapter: {exc}")
```

### 11.2 Lista (los tests de GPU ya existen en el prototipo y pasan, salvo los marcados **(nuevo)**)
| Fichero | Test | Comprueba |
|---|---|---|
| `test_params.py` | de 01 + límites nuevos | `n_types > 16`, `r_max < 0.01` y `N > 8M` se rechazan; ida y vuelta de presets |
| `test_uniforms.py` | tamaños y offsets | 48/16/32/384 bytes; cada campo en su offset (`np.frombuffer`); bytes 0–64 de `RenderParams` = `M.T.ravel()` |
| `test_camera.py` | base, proyección, rayos | base ortonormal y dextrógira; el objetivo se proyecta al centro; cursor → `focal_point` → `project` devuelve el cursor (1e-6); fuera del cubo da `None`; límites de pitch y dist; `pan`/`fly` envuelven |
| `test_overlay.py` | cubo y anillo | 24 vértices en ±0.5; 12 aristas paralelas a los ejes; anillo a distancia `radius` del centro y en el plano `right`/`up` |
| `test_controls.py` | entrada → acción | arrastre izquierdo cambia yaw y pitch; con Shift no orbita y devuelve `sign=+1`; Shift+derecho da −1; la rueda cambia dist; WASD mueve el objetivo; fuera del cubo da `None` |
| `test_scan.py` | suma exclusiva | n ∈ {1, 5, 1023, 1024, 1025, 4097, 70 000, 1 000 000} contra `np.cumsum`; `scan_total` = suma |
| `test_grid.py` | ordenación | r ∈ {0.05, 0.1, 0.33}: `cell_count` = `np.bincount`; `cell_start` = suma exclusiva; B ordenado por celda; mismo multiconjunto de partículas que A |
| `test_sim.py` | fuerzas | `dt=1`, `friction_half_life=1e-9` (fricción 0) → velocidad nueva = aceleración de B; contra `reference` con r ∈ {0.1, 0.2, 0.33}, `rtol=atol=1e-3` |
| | estabilidad | 50 pasos: finitos, dentro de `[0,1)`, el histograma de tipos se conserva |
| | pincel | atrae, repele, nada fuera del radio, cruza el borde periódico |
| | otros **(nuevo)** | cámara lenta mueve menos; `set_matrix` actualiza `params`; `nc` limitado a 100 con r = 0.01; N demasiado grande → `ValueError` |
| `test_render.py` (offscreen `rgba8unorm`, 160×120) | proyección | 3 orientaciones: píxel más brillante a ≤ 1.5 px de `cam.project` |
| | envoltura, resaltado, estela, corte, mínimo | copia más cercana al objetivo; resaltado < 50 %; la estela persiste con 0.9 y desaparece con 0; el corte oculta lo que está fuera de la losa; una partícula de ~0.3 px sigue visible con `min_px=0.75` |
| | niebla | dos partículas iguales a distinta profundidad con `fog=2`: la lejana es más tenue |

---

## 12. CI (`.github/workflows/01b-particle-life-3d.yml`)
```yaml
name: 01b-particle-life-3d
on:
  push:
    branches: [main]
    paths: ["01b-particle-life-3d/**", ".github/workflows/01b-particle-life-3d.yml"]
  pull_request:
    paths: ["01b-particle-life-3d/**", ".github/workflows/01b-particle-life-3d.yml"]
permissions:
  contents: read
defaults:
  run:
    working-directory: 01b-particle-life-3d
jobs:
  test:
    runs-on: ubuntu-latest
    env:
      REQUIRE_WGPU: "1"
    steps:
      - uses: actions/checkout@v4
      - name: Software Vulkan (lavapipe) for the WebGPU tests
        run: sudo apt-get update -qq && sudo apt-get install -y -qq mesa-vulkan-drivers
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --locked
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run pyright
      - run: uv run pytest -q  # wgpu tests on lavapipe (CPU); performance is measured locally
```
Además: añadir `01b-particle-life-3d/presets/user/` al `.gitignore` de la raíz y una fila en la tabla del `README.md` de la raíz.

---

## 13. Benchmark y rendimiento

### 13.1 `particle-life-3d-bench` (de `prototype/bench_proto.py`)
Opciones: `--counts` (100k 250k 500k 1M 2M), `--r-max` (0.03 0.045 0.06), `--steps 100`, `--warmup 10`, `--types 6`.

Imprime en Markdown: partículas | r_max | vecinos/partícula (`N·4/3·π·r³`) | pasos/s | ms de rejilla | ms de fuerzas. Las dos últimas columnas usan timestamps de GPU (`timestamp-query`, con un `query_set` de 4 entradas). Sincroniza con `gpu.sync()`.

La app con `--bench-frames N --no-vsync` mide los FPS completos (simulación + render) después de 30 fotogramas de calentamiento.

### 13.2 Qué esperar (estimación, **sin medir**)
En 01 (Taichi 2D, sin reordenar los datos), 500k partículas con 157 vecinos daban 50 pasos/s, unos 1.1·10¹⁰ candidatos/s. En 3D se recorren unos 6.45 candidatos por vecino útil (27 celdas frente a la esfera). Con datos reordenados y lectura de 16 bytes por candidato espero 3–10× más candidatos/s que en 01. Eso apunta a **~1M partículas con ~50 vecinos a 60 FPS como objetivo plausible**, con 300–500k como mínimo razonable. M2 lo confirma o lo corrige.

### 13.3 Cómo fijar los valores por defecto (tras M2)
Elegir (N, r_max) con rejilla + fuerzas ≤ 6 ms por paso (deja margen al render a 60 FPS con `steps per frame = 1`) y entre 50 y 100 vecinos por partícula. Apuntar la tabla en el README.

### 13.4 Palancas si va lento (solo con datos de M2 delante)
1. Probar workgroup de 128 o 64 en `forces.wgsl` (ocupación).
2. Rejilla de media celda: celda = r/2 y 5×5×5 celdas vecinas. Baja los candidatos de 27·r³·N a 15.6·r³·N (−42 %).
3. Render: es proporcional a N × área del halo. Bajar `particle size` o `min_px`.

---

## 14. Hitos

| Hito | Contenido | Hecho cuando | Commit(s) |
|---|---|---|---|
| 🔶 **P0** (usuario, antes de empezar) | En `docs/plans/particle-life-3d/`: `uv run probe.py` (y `--x11` si falla). En `prototype/`: `uv run demo_proto.py` y `uv run bench_proto.py` | `probe.py` dice `ALL OK`; la demo se ve; el usuario pega las tres salidas | — |
| **M0** esqueleto | §4 y §4.1, `params/colors/reference/uniforms` + tests puros, CI (§12), `.gitignore`, fila en el README raíz («en desarrollo»), README mínimo | `uv sync && uv run pytest && ruff && pyright` en verde; CI verde | `chore(particle-life-3d): project skeleton`, `ci(particle-life-3d): workflow with lavapipe for WebGPU tests` |
| **M1** simulación | shaders de simulación, `gpu.py`, `sim.py`, `conftest.py`, `test_scan/grid/sim` | tests de GPU verdes en lavapipe (§16) y en CI | `feat(particle-life-3d): GPU simulation with counting-sort grid and two-level scan` |
| **M2** benchmark | `bench.py` con timestamps | 🔶 el usuario ejecuta `uv run particle-life-3d-bench` y pega la tabla; se fijan los valores por defecto (§13.3) | `feat(particle-life-3d): headless benchmark with per-phase GPU timings` |
| **M3** cámara y controles | `camera.py`, `overlay.py`, `controls.py` + tests | tests puros verdes | `feat(particle-life-3d): orbit camera, overlays and input mapping` |
| **M4** render | `render.wgsl`, `render.py`, `test_render.py` | tests offscreen verdes en lavapipe | `feat(particle-life-3d): additive glow renderer with trails, fog and slicing` |
| **M5** app | `window.py`, `ui.py`, `app.py`, CLI | prueba de humo en Xvfb (`--frames 60`); 🔶 **el usuario prueba el MVP** en su máquina (Wayland, FPS, controles) | `feat(particle-life-3d): GLFW window and main loop`, `feat(particle-life-3d): Tk control panel for 3D` |
| **M6** cierre | README en español (instalación, controles, tablas de rendimiento **medidas por el usuario**, estructura), estado en el README raíz, ADR 0002 → *accepted*, borrar `docs/plans/particle-life-3d/prototype/` | usuario conforme | `docs(particle-life-3d): README with controls and measured performance` |
| **M7** extras (solo si el usuario los pide) | captura F12 a `outputs/` (PNG con `zlib`, como `prototype/snapshot.py`); modo esferas sólidas (impostores + buffer de profundidad + iluminación Lambert); rejilla de media celda (§13.4); grabación de vídeo | | |

No se empieza un hito sin el anterior en verde.

---

## 15. Trampas verificadas (cada una costó un fallo en el prototipo o está comprobada en el código de wgpu-py)
1. **`target` es palabra reservada en WGSL** y el shader no compila. Evita también como identificadores `common`, `filter`, `module`, `pass`, `set`, `get`, `type`, `self`, `meta`, `mod`, `match`, `move`, `from`, `with`, `use`, `shared` y `std` (lista de reservadas de WGSL).
2. `array<vec3<f32>>` tiene stride de 16 bytes: usa `vec4<f32>` (xyz + tipo en w) y en NumPy `(N, 4) float32`.
3. En los structs uniform, `vec3` se alinea a 16 y el tamaño total es múltiplo de 16. Rellena a mano y testea los offsets.
4. `workgroupBarrier()` solo en flujo de control uniforme: carga en memoria compartida y pon la barrera **antes** de `if (i >= n) { return; }`.
5. `fract(x)` puede devolver 1.0 con `x` negativo diminuto: `select(q, vec3(0.0), q >= vec3(1.0))`. Además, limita la celda con `clamp` antes de convertir a `u32`.
6. Un buffer no puede estar enlazado como escribible y como solo lectura en el mismo dispatch. Por eso existen `block_sums`/`block_offsets` por separado y el ping-pong A/B.
7. Usa **layouts explícitos** (`create_bind_group_layout`), no `layout="auto"`: el automático omite las bindings que la entrada no usa y rompe bind groups compartidos.
8. `encoder.clear_buffer()` exige `COPY_DST` y `queue.read_buffer()` exige `COPY_SRC`.
9. **`queue.on_submitted_work_done_sync()` está roto en wgpu 0.32.0** (la firma del callback de cffi no coincide). Sincroniza con `queue.read_buffer(buffer_pequeño)`.
10. `queue.write_buffer()` se aplica en el siguiente `submit`, antes de sus command buffers: un valor por buffer uniform y por `submit`. Por eso hay dos uniforms distintos para el scan.
11. NumPy es row-major y WGSL `mat4x4` column-major: sube `M.T`.
12. El clip space de WebGPU tiene z ∈ [0, 1] (perspectiva «ZO»). Descarta con `c.z < 0` antes de dividir por `w`.
13. `rgba32float` no admite blending sin la feature `float32-blendable`: acumula en `rgba16float`.
14. Un halo de radio < 0.707 px puede no cubrir ningún centro de píxel y la partícula desaparece. Por eso `min_px = 0.75` con compensación de energía `(px/s)²`.
15. El formato preferido de la superficie es `bgra8unorm-srgb`. Usa `bgra8unorm` para mantener el aspecto de 01.
16. Fuera del cubo dibujado, el punto del pincel se envolvería a otra zona de la pantalla: `focal_point_rel()` devuelve `None`.
17. Cursor en coordenadas de ventana ≠ píxeles del framebuffer en HiDPI.
18. `PYGLFW_LIBRARY_VARIANT` debe ponerse antes de `import glfw`.
19. Algunos Python del sistema no traen tkinter: de ahí `python-preference = "only-managed"`.
20. Las atómicas hacen que el orden en GPU no sea determinista: tests con tolerancias e invariantes (§6.3).

---

## 16. Entorno del agente (contenedor en la nube, sin GPU)
Vulkan por software **en espacio de usuario**: no instala nada en el sistema; solo descarga y extrae un `.deb`.
```bash
LVP="$HOME/.cache/lavapipe"; mkdir -p "$LVP" && cd "$LVP"
apt-get download mesa-vulkan-drivers                 # downloads the .deb only
dpkg -x mesa-vulkan-drivers_*.deb root               # extracts here; installs nothing
ldd root/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so | grep "not found"   # must print nothing
sed "s#\"library_path\": \"libvulkan_lvp.so\"#\"library_path\": \"$LVP/root/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so\"#" \
    root/usr/share/vulkan/icd.d/lvp_icd.json > lvp_icd.json
export VK_ICD_FILENAMES="$LVP/lvp_icd.json"          # in every shell that runs wgpu
```
- Requiere `libvulkan.so.1` y las librerías de LLVM en el contenedor (estaban). Comprobación: en `docs/plans/particle-life-3d/`, `uv run probe.py --no-window` debe listar `llvmpipe`.
- Prueba de humo de la ventana (X11 sobre Xvfb, Tk incluido): `XDG_SESSION_TYPE=x11 xvfb-run -a uv run particle-life-3d --particles 20000 --r-max 0.08 --frames 60 --no-vsync`.
- Las cifras de rendimiento de lavapipe no significan nada: solo valen las de la máquina del usuario.

---

## 17. Riesgos y mitigaciones
| Riesgo | Mitigación |
|---|---|
| La superficie Wayland nativa falla con GLFW | P0 lo detecta; `--x11` (XWayland). Si solo funciona X11, se usa por defecto |
| Tk (XWayland) y GLFW (Wayland) en el mismo proceso dan problemas | P0 + demo; `--x11` deja ambos en X11 |
| Falta `libvulkan1` en la máquina del usuario | La sonda lo dice («vulkan drivers/libraries could not be loaded»); instalar paquetes del sistema **solo con permiso del usuario** |
| Cambia la API de wgpu-py (0.x) | Versión exacta fijada; actualizar a propósito, con tests |
| Rendimiento por debajo de lo esperado | §13.4, siempre con datos de M2 delante |
| Tests de GPU lentos en CI | N pequeño; ~3 s en lavapipe |

## 18. Fuera de alcance (por ahora)
Esferas sólidas con iluminación, sombras, profundidad de campo, grabación de vídeo, más de 16 tipos, varias especies con parámetros distintos, simulación en doble precisión, versión web. Algunos están como M7.
