# 1. One independent uv project per simulator

Date: 2026-10-03 · Status: accepted

## Context
Taichi, JAX, PyTorch and MuJoCo pin different CUDA wheels and dependency versions.

## Decision
Each simulator has its own `pyproject.toml` and `uv.lock`. No shared package until duplication appears in 2-3 simulators.

## Consequences
Some duplicated boilerplate; no dependency conflicts; CI can test each folder independently via path filters.
