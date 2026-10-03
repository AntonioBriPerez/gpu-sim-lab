import pytest
import taichi as ti


@pytest.fixture(scope="session", autouse=True)
def _taichi_cpu() -> None:
    # One thread keeps the grid scatter order (and float sums) deterministic.
    ti.init(arch=ti.cpu, cpu_max_num_threads=1, random_seed=0)
