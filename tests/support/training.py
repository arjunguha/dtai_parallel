from __future__ import annotations

import copy
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_

from dtai_parallel import apply_cpu_streaming_


def make_batches(world_size: int, dtype: torch.dtype) -> Tuple[Sequence[Tensor], Sequence[Tensor]]:
    """Create deterministic per-rank regression batches for equivalence checks."""
    generator = torch.Generator(device="cpu").manual_seed(20260107)
    xs = [torch.randn(4, 5, generator=generator, dtype=dtype) for _ in range(world_size)]
    ys = [torch.randn(4, 3, generator=generator, dtype=dtype) for _ in range(world_size)]
    return xs, ys


def make_kwarg_batches(dtype: torch.dtype):
    """Create deterministic inputs for nested-output and keyword replay coverage."""
    generator = torch.Generator(device="cpu").manual_seed(20260108)
    x = torch.randn(4, 5, generator=generator, dtype=dtype)
    mask = torch.sigmoid(torch.randn(4, 8, generator=generator, dtype=dtype))
    context = torch.randn(4, 4, generator=generator, dtype=dtype)
    y = torch.randn(4, 3, generator=generator, dtype=dtype)
    return x, mask, context, y


def optimizer_kwargs_for(name: str) -> Dict[str, object]:
    """Return stable optimizer settings shared by reference and streaming paths."""
    if name == "adamw":
        return {"lr": 3e-3, "betas": (0.8, 0.9), "eps": 1e-8, "weight_decay": 0.01, "foreach": False}
    if name == "sgd":
        return {"lr": 1e-2, "momentum": 0.2, "weight_decay": 0.01, "foreach": False}
    raise AssertionError(name)


def optimizer_cls_for(name: str):
    """Map compact optimizer names used in parametrized tests to PyTorch classes."""
    if name == "adamw":
        return torch.optim.AdamW
    if name == "sgd":
        return torch.optim.SGD
    raise AssertionError(name)


def assert_state_dicts_close(actual: Mapping[str, Tensor], expected: Mapping[str, Tensor], *, dtype: torch.dtype) -> None:
    """Compare model states with tolerances appropriate for the selected dtype."""
    assert actual.keys() == expected.keys()
    if dtype == torch.float64:
        rtol, atol = 2e-9, 2e-10
    else:
        rtol, atol = 5e-5, 5e-6
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=rtol, atol=atol, msg=lambda msg: f"{key}: {msg}")


def assert_state_dicts_equal(actual: Mapping[str, Tensor], expected: Mapping[str, Tensor]) -> None:
    """Compare model states that should be bitwise identical."""
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0.0, atol=0.0, msg=lambda msg: f"{key}: {msg}")


def train_single_process_reference(
    initial_model: nn.Module,
    xs_cpu: Sequence[Tensor],
    ys_cpu: Sequence[Tensor],
    *,
    optimizer_name: str,
    optimizer_kwargs: Mapping[str, object],
    max_grad_norm: Optional[float],
    steps: int,
    device: torch.device,
) -> nn.Module:
    """Train an ordinary model as the baseline for streaming equivalence."""
    model = copy.deepcopy(initial_model).to(device)
    optimizer = optimizer_cls_for(optimizer_name)(model.parameters(), **dict(optimizer_kwargs))
    criterion = nn.MSELoss()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = sum(criterion(model(x.to(device)), y.to(device)) for x, y in zip(xs_cpu, ys_cpu)) / float(len(xs_cpu))
        loss.backward()
        if max_grad_norm is not None:
            clip_grad_norm_(model.parameters(), max_grad_norm, foreach=False)
        optimizer.step()
    return model.cpu()


def train_single_process_streaming(
    initial_model: nn.Module,
    module_path: str,
    xs_cpu: Sequence[Tensor],
    ys_cpu: Sequence[Tensor],
    *,
    offload_policy,
    optimizer_name: str,
    optimizer_kwargs: Mapping[str, object],
    max_grad_norm: Optional[float],
    steps: int,
    device: torch.device,
    pin_cpu_masters=True,
    resident_suffix_count: int = 0,
) -> nn.Module:
    """Train a streamed model through the public engine API for equivalence checks."""
    model = copy.deepcopy(initial_model)
    engine = apply_cpu_streaming_(
        model,
        module_path,
        offload_policy=offload_policy,
        resident_suffix_count=resident_suffix_count,
        optimizer_cls=optimizer_cls_for(optimizer_name),
        optimizer_kwargs=optimizer_kwargs,
        max_grad_norm=max_grad_norm,
        device=device,
        wrap_ddp=False,
        auto_init_process_group=False,
        pin_cpu_masters=pin_cpu_masters,
    )
    criterion = nn.MSELoss()
    for _ in range(steps):
        engine.zero_grad(set_to_none=True)
        loss = sum(criterion(engine.model(x.to(device)), y.to(device)) for x, y in zip(xs_cpu, ys_cpu)) / float(len(xs_cpu))
        loss.backward()
        engine.step()
    closed = engine.close(return_on_all_ranks=True, device=torch.device("cpu"))
    assert closed is not None
    return closed.cpu()
