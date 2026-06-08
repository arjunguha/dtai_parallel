from __future__ import annotations

import copy
import os
import tempfile
from collections import OrderedDict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_

from dtai_parallel import CPUStreamingDataParallel, StageSpec


# ---------------------------------------------------------------------------
# Test models and deterministic data
# ---------------------------------------------------------------------------


def make_tiny_model(dtype: torch.dtype = torch.float64) -> nn.Sequential:
    """A small sequential network with both parameterized and parameterless stages.

    The stage names are stable so a state_dict from the materialized streaming
    model can be compared directly with an ordinary PyTorch reference model.
    """

    torch.manual_seed(2026)
    return nn.Sequential(
        OrderedDict(
            [
                ("in_proj", nn.Linear(5, 7)),
                ("act0", nn.Tanh()),
                ("hidden", nn.Linear(7, 7)),
                ("norm", nn.LayerNorm(7)),
                ("act1", nn.GELU()),
                ("out_proj", nn.Linear(7, 3)),
            ]
        )
    ).to(dtype=dtype)


def make_stage_specs(model: nn.Sequential, offload_pattern: Mapping[str, bool]) -> List[StageSpec]:
    return [
        StageSpec(name=name, module=copy.deepcopy(module), offload=offload_pattern[name])
        for name, module in model.named_children()
    ]


def pattern_all_offloaded(model: nn.Sequential) -> Dict[str, bool]:
    return {name: True for name, _ in model.named_children()}


def pattern_mixed(model: nn.Sequential) -> Dict[str, bool]:
    """Keep the embedding-ish and normalization stages resident; stream the rest."""

    return {name: name not in {"in_proj", "norm"} for name, _ in model.named_children()}


def make_local_batches(world_size: int, dtype: torch.dtype) -> Tuple[List[Tensor], List[Tensor]]:
    generator = torch.Generator(device="cpu").manual_seed(12345)
    xs = [torch.randn(4, 5, generator=generator, dtype=dtype) for _ in range(world_size)]
    ys = [torch.randn(4, 3, generator=generator, dtype=dtype) for _ in range(world_size)]
    return xs, ys


def optimizer_kwargs_for(name: str) -> Dict[str, object]:
    if name == "adamw":
        # foreach=False keeps tests focused on the abstraction rather than on
        # backend-specific foreach/fused kernels.
        return {"lr": 3.0e-3, "betas": (0.8, 0.9), "eps": 1.0e-8, "weight_decay": 0.01, "foreach": False}
    if name == "sgd":
        return {"lr": 1.0e-2, "momentum": 0.2, "weight_decay": 0.01, "foreach": False}
    raise AssertionError(name)


def optimizer_cls_for(name: str):
    if name == "adamw":
        return torch.optim.AdamW
    if name == "sgd":
        return torch.optim.SGD
    raise AssertionError(name)


# ---------------------------------------------------------------------------
# Ordinary mathematical DDP reference
# ---------------------------------------------------------------------------


def train_reference_model(
    initial_model: nn.Sequential,
    xs_cpu: Sequence[Tensor],
    ys_cpu: Sequence[Tensor],
    *,
    optimizer_name: str,
    optimizer_kwargs: Mapping[str, object],
    max_grad_norm: Optional[float],
    steps: int,
    device: torch.device,
) -> Tuple[nn.Sequential, List[Optional[Tensor]]]:
    """Train a normal model with the same mean-gradient semantics as DDP.

    DDP computes local gradients on each rank and all-reduces their average.  In
    a single-process mathematical reference, this is exactly the same as using
    the average of the local losses.  The streaming model's training loop instead
    sums local losses and lets the abstraction divide gradients by world_size.
    """

    model = copy.deepcopy(initial_model).to(device)
    optimizer = optimizer_cls_for(optimizer_name)(model.parameters(), **dict(optimizer_kwargs))
    criterion = nn.MSELoss()
    norms: List[Optional[Tensor]] = []

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        local_losses = []
        for x_cpu, y_cpu in zip(xs_cpu, ys_cpu):
            x = x_cpu.to(device)
            y = y_cpu.to(device)
            local_losses.append(criterion(model(x), y))
        loss = sum(local_losses) / float(len(local_losses))
        loss.backward()
        if max_grad_norm is not None:
            norms.append(clip_grad_norm_(model.parameters(), max_grad_norm))
        else:
            norms.append(None)
        optimizer.step()

    return model.cpu(), norms


def train_streaming_model(
    initial_model: nn.Sequential,
    offload_pattern: Mapping[str, bool],
    xs_cpu: Sequence[Tensor],
    ys_cpu: Sequence[Tensor],
    *,
    optimizer_name: str,
    optimizer_kwargs: Mapping[str, object],
    max_grad_norm: Optional[float],
    steps: int,
    devices: Sequence[torch.device],
) -> Tuple[nn.Sequential, List[Optional[Tensor]]]:
    stages = make_stage_specs(initial_model, offload_pattern)
    model = CPUStreamingDataParallel(
        stages,
        devices=devices,
        optimizer_cls=optimizer_cls_for(optimizer_name),
        optimizer_kwargs=optimizer_kwargs,
        max_grad_norm=max_grad_norm,
    )
    criterion = nn.MSELoss()
    norms: List[Optional[Tensor]] = []

    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        inputs = [x.to(device) for x, device in zip(xs_cpu, devices)]
        targets = [y.to(device) for y, device in zip(ys_cpu, devices)]
        outputs = model(inputs)

        # DDP-style contract for this abstraction: sum local losses here, then
        # let CPUStreamingDataParallel average gradients across replicas.
        loss_device = outputs[0].device
        loss = sum(criterion(out, target).to(loss_device) for out, target in zip(outputs, targets))
        loss.backward()
        norms.append(model.step())

    return model.close().cpu(), norms


def assert_state_dicts_close(actual: Mapping[str, Tensor], expected: Mapping[str, Tensor], *, dtype: torch.dtype) -> None:
    assert actual.keys() == expected.keys()
    if dtype == torch.float64:
        rtol, atol = 5e-10, 5e-11
    else:
        rtol, atol = 3e-5, 3e-6
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=rtol, atol=atol, msg=lambda msg: f"{key}: {msg}")


@pytest.mark.parametrize("device_kind", ["cpu", "cuda"])
@pytest.mark.parametrize("offload_name", ["all", "mixed"])
@pytest.mark.parametrize("optimizer_name", ["adamw"])
@pytest.mark.parametrize("max_grad_norm", [None, 0.35])
def test_streaming_matches_single_process_ddp_math(
    device_kind: str,
    offload_name: str,
    optimizer_name: str,
    max_grad_norm: Optional[float],
) -> None:
    if device_kind == "cuda" and torch.cuda.device_count() < 2:
        pytest.skip("needs at least two CUDA devices")

    dtype = torch.float64 if device_kind == "cpu" else torch.float32
    world_size = 2
    devices = [torch.device("cpu") for _ in range(world_size)] if device_kind == "cpu" else [torch.device(f"cuda:{i}") for i in range(world_size)]

    initial_model = make_tiny_model(dtype=dtype)
    offload_pattern = pattern_all_offloaded(initial_model) if offload_name == "all" else pattern_mixed(initial_model)
    xs_cpu, ys_cpu = make_local_batches(world_size, dtype=dtype)
    optimizer_kwargs = optimizer_kwargs_for(optimizer_name)

    reference, reference_norms = train_reference_model(
        initial_model,
        xs_cpu,
        ys_cpu,
        optimizer_name=optimizer_name,
        optimizer_kwargs=optimizer_kwargs,
        max_grad_norm=max_grad_norm,
        steps=2,
        device=torch.device("cpu") if device_kind == "cpu" else torch.device("cuda:0"),
    )
    streaming, streaming_norms = train_streaming_model(
        initial_model,
        offload_pattern,
        xs_cpu,
        ys_cpu,
        optimizer_name=optimizer_name,
        optimizer_kwargs=optimizer_kwargs,
        max_grad_norm=max_grad_norm,
        steps=2,
        devices=devices,
    )

    assert_state_dicts_close(streaming.state_dict(), reference.state_dict(), dtype=dtype)
    if max_grad_norm is not None:
        for actual, expected in zip(streaming_norms, reference_norms):
            assert actual is not None and expected is not None
            torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-6 if dtype == torch.float64 else 1e-4, atol=1e-7)


# ---------------------------------------------------------------------------
# Real DDP reference.  This is intentionally a smaller test because it launches
# worker processes.  It proves the mathematical reference above agrees with
# PyTorch's DistributedDataParallel behavior.
# ---------------------------------------------------------------------------


def _ddp_worker(
    rank: int,
    world_size: int,
    init_file: str,
    result_file: str,
    initial_state: Mapping[str, Tensor],
    xs_cpu: Sequence[Tensor],
    ys_cpu: Sequence[Tensor],
    optimizer_name: str,
    optimizer_kwargs: Mapping[str, object],
    max_grad_norm: Optional[float],
    steps: int,
    dtype: torch.dtype,
    use_cuda: bool,
) -> None:
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size, init_method=f"file://{init_file}")
    try:
        device = torch.device(f"cuda:{rank}") if use_cuda else torch.device("cpu")
        model = make_tiny_model(dtype=dtype).to(device)
        model.load_state_dict(initial_state, strict=True)
        ddp = DDP(model, device_ids=[rank] if use_cuda else None)
        optimizer = optimizer_cls_for(optimizer_name)(ddp.parameters(), **dict(optimizer_kwargs))
        criterion = nn.MSELoss()

        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            x = xs_cpu[rank].to(device)
            y = ys_cpu[rank].to(device)
            loss = criterion(ddp(x), y)
            loss.backward()
            if max_grad_norm is not None:
                clip_grad_norm_(ddp.parameters(), max_grad_norm)
            optimizer.step()

        dist.barrier()
        if rank == 0:
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, result_file)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("device_kind", ["cpu", "cuda"])
def test_streaming_matches_real_ddp(device_kind: str) -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    if device_kind == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() < 2):
        pytest.skip("needs at least two CUDA devices")

    use_cuda = device_kind == "cuda"
    dtype = torch.float32 if use_cuda else torch.float64
    world_size = 2
    initial_model = make_tiny_model(dtype=dtype)
    offload_pattern = pattern_mixed(initial_model)
    xs_cpu, ys_cpu = make_local_batches(world_size, dtype=dtype)
    optimizer_name = "adamw"
    optimizer_kwargs = optimizer_kwargs_for(optimizer_name)
    max_grad_norm = 0.5
    steps = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "ddp_init")
        result_file = os.path.join(tmpdir, "rank0_state.pt")
        # The file init method wants the path not to exist yet.
        if os.path.exists(init_file):
            os.unlink(init_file)

        mp.start_processes(
            _ddp_worker,
            args=(
                world_size,
                init_file,
                result_file,
                initial_model.state_dict(),
                xs_cpu,
                ys_cpu,
                optimizer_name,
                optimizer_kwargs,
                max_grad_norm,
                steps,
                dtype,
                use_cuda,
            ),
            nprocs=world_size,
            start_method="spawn",
            join=True,
        )
        ddp_state = torch.load(result_file, map_location="cpu")

    devices = [torch.device("cpu") for _ in range(world_size)] if not use_cuda else [torch.device(f"cuda:{i}") for i in range(world_size)]
    streaming_model, _ = train_streaming_model(
        initial_model,
        offload_pattern,
        xs_cpu,
        ys_cpu,
        optimizer_name=optimizer_name,
        optimizer_kwargs=optimizer_kwargs,
        max_grad_norm=max_grad_norm,
        steps=steps,
        devices=devices,
    )

    assert_state_dicts_close(streaming_model.state_dict(), ddp_state, dtype=dtype)


@pytest.mark.parametrize("device_kind", ["cpu", "cuda"])
def test_close_returns_materialized_model_on_device_zero(device_kind: str) -> None:
    if device_kind == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    dtype = torch.float32
    devices = [torch.device("cpu")] if device_kind == "cpu" else [torch.device("cuda:0")]
    initial_model = make_tiny_model(dtype=dtype)
    model = CPUStreamingDataParallel(
        make_stage_specs(initial_model, pattern_all_offloaded(initial_model)),
        devices=devices,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs=optimizer_kwargs_for("adamw"),
    )
    materialized = model.close()
    assert isinstance(materialized, nn.Sequential)
    for parameter in materialized.parameters():
        assert parameter.device == devices[0]
    with pytest.raises(RuntimeError):
        _ = model([torch.randn(2, 5, device=devices[0])])


def test_training_loop_must_place_inputs_on_expected_device() -> None:
    initial_model = make_tiny_model(dtype=torch.float32)
    model = CPUStreamingDataParallel(
        make_stage_specs(initial_model, pattern_all_offloaded(initial_model)),
        devices=[torch.device("cpu"), torch.device("cpu")],
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs=optimizer_kwargs_for("adamw"),
    )
    with pytest.raises(ValueError, match="expected 2 input tensors"):
        _ = model([torch.randn(2, 5)])


def test_optimizer_dispatch_is_not_adamw_specific() -> None:
    """The optimizer path should work for another per-parameter PyTorch optimizer."""

    dtype = torch.float64
    world_size = 2
    devices = [torch.device("cpu") for _ in range(world_size)]
    initial_model = make_tiny_model(dtype=dtype)
    offload_pattern = pattern_all_offloaded(initial_model)
    xs_cpu, ys_cpu = make_local_batches(world_size, dtype=dtype)
    optimizer_kwargs = optimizer_kwargs_for("sgd")

    reference, _ = train_reference_model(
        initial_model,
        xs_cpu,
        ys_cpu,
        optimizer_name="sgd",
        optimizer_kwargs=optimizer_kwargs,
        max_grad_norm=None,
        steps=2,
        device=torch.device("cpu"),
    )
    streaming, _ = train_streaming_model(
        initial_model,
        offload_pattern,
        xs_cpu,
        ys_cpu,
        optimizer_name="sgd",
        optimizer_kwargs=optimizer_kwargs,
        max_grad_norm=None,
        steps=2,
        devices=devices,
    )
    assert_state_dicts_close(streaming.state_dict(), reference.state_dict(), dtype=dtype)
