"""Small wgpu helpers: device creation, shader loading, explicit bind groups."""

from __future__ import annotations

from pathlib import Path

import wgpu

SHADER_DIR = Path(__file__).parent / "shaders"

# binding kind -> layout entry body
_KINDS = {
    "uniform": {"buffer": {"type": wgpu.BufferBindingType.uniform}},
    "read": {"buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
    "rw": {"buffer": {"type": wgpu.BufferBindingType.storage}},
}


def create_device(
    power_preference: str = "high-performance", features: tuple[str, ...] = ()
) -> wgpu.GPUDevice:
    adapter = wgpu.gpu.request_adapter_sync(power_preference=power_preference)
    lim = adapter.limits
    return adapter.request_device_sync(
        required_features=[f for f in features if f in adapter.features],
        required_limits={
            "max-storage-buffer-binding-size": lim["max-storage-buffer-binding-size"],
            "max-buffer-size": lim["max-buffer-size"],
        },
    )


def load_shader(device: wgpu.GPUDevice, *names: str) -> wgpu.GPUShaderModule:
    """Concatenate WGSL files (e.g. common + kernel) into one module."""
    code = "\n".join((SHADER_DIR / f"{n}.wgsl").read_text() for n in names)
    return device.create_shader_module(label="+".join(names), code=code)


def compute_pipeline(
    device: wgpu.GPUDevice,
    module: wgpu.GPUShaderModule,
    entry: str,
    kinds: dict[int, str],
) -> tuple[wgpu.GPUComputePipeline, wgpu.GPUBindGroupLayout]:
    """Pipeline with an explicit layout: kinds maps binding -> 'uniform' | 'read' | 'rw'."""
    bgl = device.create_bind_group_layout(
        entries=[
            {"binding": b, "visibility": wgpu.ShaderStage.COMPUTE, **_KINDS[k]}
            for b, k in sorted(kinds.items())
        ]
    )
    layout = device.create_pipeline_layout(bind_group_layouts=[bgl])
    pipe = device.create_compute_pipeline(
        label=entry, layout=layout, compute={"module": module, "entry_point": entry}
    )
    return pipe, bgl


def bind(device: wgpu.GPUDevice, bgl: wgpu.GPUBindGroupLayout, buffers: dict[int, wgpu.GPUBuffer]):
    return device.create_bind_group(
        layout=bgl,
        entries=[{"binding": b, "resource": {"buffer": buf}} for b, buf in sorted(buffers.items())],
    )
