# 01 · Particle Life

Vida artificial emergente: N tipos de partículas con una matriz de atracción/repulsión entre tipos. Simulación en GPU con [Taichi](https://www.taichi-lang.org/) 1.7.4 y búsqueda de vecinos con rejilla espacial (ordenación por conteo), no fuerza bruta.

## Requisitos
- Linux con GPU NVIDIA (probado: Ubuntu 26.04, RTX 4070 Ti Super, driver 595, sesión Wayland/XWayland) y [`uv`](https://docs.astral.sh/uv/).
- Python 3.12 (lo gestiona `uv`; Taichi 1.7.4 no tiene ruedas para 3.14).

## Instalar y lanzar
```bash
cd 01-particle-life
uv sync
uv run particle-life                                   # 30k partículas, 6 tipos
uv run particle-life --particles 200000 --r-max 0.01   # más partículas, radio menor
uv run particle-life --preset presets/user/mi_preset.json
```
Opciones: `--particles`, `--types`, `--seed`, `--r-max`, `--preset`, `--size` (píxeles de la ventana, 1200 por defecto; sube el valor si los paneles se ven pequeños), `--no-vsync`, `--bench-frames N`.

## Controles (panel en la ventana)
| Control | Efecto |
|---|---|
| `radius` | radio de interacción (fracción del mundo, 0.004–0.33) |
| `repulsion core (beta)` | radio normalizado bajo el cual todas se repelen |
| `force` / `friction half-life` | intensidad de la fuerza / tiempo en que la velocidad se reduce a la mitad |
| `point size`, `paused`, `show matrix` | visualización |
| Matriz (sliders `i->j`) | cuánto es atraída (+1) o repelida (−1) la fila `i` por la columna `j` |
| `random matrix` / `randomise positions` | nueva matriz / nuevo estado inicial |
| `save preset` | guarda en `presets/user/` (ignorado por git) |
| `load next preset` | recorre `presets/*.json` y `presets/user/*.json` |

El mundo es el cuadrado unidad con bordes periódicos (toroidal).

## Rendimiento medido
RTX 4070 Ti Super, Taichi 1.7.4 (CUDA), 6 tipos. **El coste depende de los vecinos por partícula (`N·π·r²`), no solo de N**: con un radio grande, 100k partículas ya pesan más que 500k con radio pequeño.

Solo simulación (`uv run particle-life-bench`):

| partículas | r_max | vecinos/partícula | pasos/s |
|---|---|---|---|
| 50 000 | 0.01 | 16 | 1827 |
| 100 000 | 0.01 | 31 | 895 |
| 200 000 | 0.01 | 63 | 278 |
| 500 000 | 0.01 | 157 | 50 |
| 50 000 | 0.02 | 63 | 752 |
| 100 000 | 0.02 | 126 | 298 |
| 200 000 | 0.02 | 251 | 86 |
| 500 000 | 0.02 | 628 | 17 |

Aplicación completa con ventana, sin vsync (`--no-vsync --bench-frames 300`), `r_max=0.01` salvo indicación:

| partículas | FPS |
|---|---|
| 30 000 (r=0.03) | 216 |
| 100 000 | 194 |
| 200 000 | 108 |
| 300 000 | 68 |
| 500 000 | 35 |

Conclusión: **~300 000 partículas a 60+ FPS** con radio 0.01 en este equipo. El objetivo de "cientos de miles" se cumple hasta ahí.

## Desarrollo
```bash
uv run pytest            # tests en CPU (rejilla vs fuerza bruta, reproducibilidad, presets)
uv run ruff check . && uv run ruff format --check .
uv run pyright
```

## Estructura
- `src/particle_life/sim.py` — simulación (Taichi), sin ventana ni entrada de usuario.
- `src/particle_life/params.py` — parámetros y presets (Python puro).
- `src/particle_life/reference.py` — referencia NumPy O(N²) usada en los tests.
- `src/particle_life/app.py` — ventana GGUI y controles.
- `src/particle_life/bench.py` — benchmark sin ventana.
