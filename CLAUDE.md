# gpu-sim-lab

Monorepo of independent GPU simulators. See README.md for the index.

## Conventions
- Each simulator is a standalone `uv` project (own `pyproject.toml` + `uv.lock`). Never share an environment between simulators: CUDA/dependency versions conflict.
- No shared code until the same thing is duplicated in 2-3 simulators.
- `src/` layout. Simulation logic is pure and testable; rendering and user input live in separate modules.
- Quality: `ruff` (lint+format), type checking (pyright), `pytest`, pre-commit (incl. gitleaks).
- CI runs on CPU with small sizes only. GPU performance tests run locally.
- Nothing heavy in git: models, checkpoints, videos, experiment logs are gitignored.
- Conventional Commits, small commits. Code and comments in English; READMEs may be Spanish.
- Verify library versions against official docs before pinning; do not rely on memory.
- Do not install system-level packages (drivers, CUDA, apt) without asking the user.
- Never write tokens or secrets into files, commits or logs.

## Environment
Ubuntu 26.04 native, Wayland (XWayland available), RTX 4070 Ti Super 16 GB, driver 595 / CUDA 13.2, system Python 3.14 (use `uv` to pin the version each project needs).

## Commands
Per project: `cd <NN-name> && uv sync && uv run pytest && uv run ruff check . && uv run ruff format --check .`
Repo-wide: `pre-commit run --all-files`

## Workflow
Plan -> user approval -> small steps with tests -> measure real performance -> commit/push. Do not start the next simulator until the previous MVP is tested by the user.
