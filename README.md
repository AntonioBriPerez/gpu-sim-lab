# gpu-sim-lab

Laboratorio personal de simuladores interactivos acelerados por GPU. Proyecto para aprender y disfrutar, con buenas prácticas de ingeniería.

**Hardware de referencia:** Intel i7-14700 · 64 GB RAM · NVIDIA RTX 4070 Ti Super (16 GB) · Ubuntu (Linux nativo).

## Simuladores

| # | Carpeta | Qué es | Stack | Estado |
|---|---|---|---|---|
| 1 | [`01-particle-life`](01-particle-life) | Vida artificial emergente con matriz de atracción/repulsión | Taichi | pendiente |
| 2 | [`02-fluids`](02-fluids) | Fluido 2D interactivo (stable fluids) | Taichi | pendiente |
| 3 | [`03-evolution`](03-evolution) | Ecosistema evolutivo con cerebros MLP | JAX | pendiente |
| 4 | [`04-agent-town`](04-agent-town) | Pueblo de agentes con LLM local | Ollama / llama.cpp | pendiente |
| 5 | [`05-robot-locomotion`](05-robot-locomotion) | Locomoción de robot por RL | MJX / Playground | pendiente |

## Requisitos generales

- Linux con driver NVIDIA reciente, [`uv`](https://docs.astral.sh/uv/) y `git`.
- Cada simulador es un proyecto independiente con su propio `pyproject.toml` y entorno `uv` (las dependencias CUDA de Taichi, JAX, PyTorch y MuJoCo chocan entre sí).

## Cómo lanzar un simulador

Cada carpeta tiene su propio README con instalación, controles y rendimiento medido.

## Desarrollo

Ver [`CLAUDE.md`](CLAUDE.md) para convenciones y comandos, y [`docs/adr/`](docs/adr) para las decisiones técnicas.

## Licencia

MIT, ver [`LICENSE`](LICENSE).
