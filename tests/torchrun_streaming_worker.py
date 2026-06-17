from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_

from dtai_parallel import CPUStreamingModuleList, apply_cpu_streaming_
from dtai_parallel.streaming import _SharedCpuStore
from tests.models import KwargSandwichModel, SandwichModel, SequentialSandwich
from tests.support.training import (
    make_batches,
    make_kwarg_batches,
    optimizer_cls_for,
    optimizer_kwargs_for,
    train_single_process_reference,
    train_single_process_streaming,
)


def _local_rank() -> int:
    """Read the torchrun local rank for device selection."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def _rank() -> int:
    """Read the torchrun global rank for rank-zero output."""
    return int(os.environ.get("RANK", "0"))


def _world_size() -> int:
    """Read the torchrun world size for distributed setup."""
    return int(os.environ.get("WORLD_SIZE", "1"))


def _device() -> torch.device:
    """Select the CUDA device assigned to this torchrun worker."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this torchrun worker")
    device = torch.device("cuda", _local_rank())
    torch.cuda.set_device(device)
    return device


def _init_distributed_if_needed() -> None:
    """Initialize NCCL for torchrun worker cases that need collectives."""
    if not dist.is_initialized():
        device = torch.device("cuda", _local_rank())
        torch.cuda.set_device(device)
        try:
            dist.init_process_group(backend="nccl", device_id=device)
        except TypeError:
            dist.init_process_group(backend="nccl")


def _destroy_distributed_if_needed() -> None:
    """Tear down the process group when a worker case initialized it."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _streaming_ddp_state(
    initial_state: Mapping[str, Tensor],
    xs_cpu: Sequence[Tensor],
    ys_cpu: Sequence[Tensor],
    *,
    max_grad_norm: Optional[float],
    steps: int,
) -> Mapping[str, Tensor] | None:
    """Train the streamed DDP variant and return its rank-zero state."""
    _init_distributed_if_needed()
    try:
        device = _device()
        rank = _rank()
        model = SandwichModel(dtype=torch.float32)
        model.load_state_dict(initial_state, strict=True)
        engine = apply_cpu_streaming_(
            model,
            "layers",
            offload_policy=[True, False, True],
            optimizer_cls=optimizer_cls_for("adamw"),
            optimizer_kwargs=optimizer_kwargs_for("adamw"),
            max_grad_norm=max_grad_norm,
            device=device,
        )
        assert isinstance(engine.model, DDP)
        criterion = nn.MSELoss()
        for _ in range(steps):
            engine.zero_grad(set_to_none=True)
            loss = criterion(engine.model(xs_cpu[rank].to(device)), ys_cpu[rank].to(device))
            loss.backward()
            engine.step()
        closed = engine.close(device=torch.device("cpu"))
        if rank == 0:
            assert closed is not None
            return {key: value.detach().cpu() for key, value in closed.state_dict().items()}
        return None
    finally:
        _destroy_distributed_if_needed()


def _standard_state(
    initial_state: Mapping[str, Tensor],
    xs_cpu: Sequence[Tensor],
    ys_cpu: Sequence[Tensor],
    *,
    max_grad_norm: Optional[float],
    steps: int,
) -> Mapping[str, Tensor]:
    """Train a one-rank full-batch CUDA reference state."""
    device = _device()
    model = SandwichModel(dtype=torch.float32).to(device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs_for("adamw"))
    criterion = nn.MSELoss()
    x = torch.cat(list(xs_cpu), dim=0)
    y = torch.cat(list(ys_cpu), dim=0)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x.to(device)), y.to(device))
        loss.backward()
        if max_grad_norm is not None:
            clip_grad_norm_(model.parameters(), max_grad_norm, foreach=False)
        optimizer.step()
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _deterministic_inputs() -> tuple[Mapping[str, Tensor], Sequence[Tensor], Sequence[Tensor]]:
    """Create identical initial model state and batches for separate torchrun jobs."""
    dtype = torch.float32
    torch.manual_seed(20260610)
    initial_model = SandwichModel(dtype=dtype)
    xs_cpu, ys_cpu = make_batches(world_size=2, dtype=dtype)
    return initial_model.state_dict(), xs_cpu, ys_cpu


def run_streaming_ddp_state(result_file: Path) -> None:
    """Write the streamed DDP state for pytest-side comparison."""
    initial_state, xs_cpu, ys_cpu = _deterministic_inputs()
    state = _streaming_ddp_state(initial_state, xs_cpu, ys_cpu, max_grad_norm=0.5, steps=2)
    if _rank() == 0:
        assert state is not None
        torch.save(state, result_file)


def run_standard_state(result_file: Path) -> None:
    """Write the one-rank full-batch CUDA reference state."""
    if _rank() != 0:
        return
    initial_state, xs_cpu, ys_cpu = _deterministic_inputs()
    state = _standard_state(initial_state, xs_cpu, ys_cpu, max_grad_norm=0.5, steps=2)
    torch.save(state, result_file)


def run_single_gpu_reference_state(
    result_file: Path,
    *,
    container_kind: str,
    optimizer_name: str,
    max_grad_norm: Optional[float],
) -> None:
    """Write ordinary one-GPU training state for pytest-side comparison."""
    if _rank() != 0:
        return
    dtype = torch.float32
    device = _device()
    torch.manual_seed(20260610)
    initial_model, module_path = _single_gpu_model_and_path(container_kind, dtype)
    del module_path
    xs_cpu, ys_cpu = make_batches(world_size=2, dtype=dtype)
    state = train_single_process_reference(
        initial_model,
        xs_cpu,
        ys_cpu,
        optimizer_name=optimizer_name,
        optimizer_kwargs=optimizer_kwargs_for(optimizer_name),
        max_grad_norm=max_grad_norm,
        steps=2,
        device=device,
    ).state_dict()
    torch.save({key: value.detach().cpu() for key, value in state.items()}, result_file)


def run_single_gpu_streaming_state(
    result_file: Path,
    *,
    container_kind: str,
    optimizer_name: str,
    max_grad_norm: Optional[float],
    resident_suffix_count: int = 0,
) -> None:
    """Write streamed one-GPU training state for pytest-side comparison."""
    if _rank() != 0:
        return
    _init_distributed_if_needed()
    try:
        dtype = torch.float32
        device = _device()
        torch.manual_seed(20260610)
        initial_model, module_path = _single_gpu_model_and_path(container_kind, dtype)
        xs_cpu, ys_cpu = make_batches(world_size=2, dtype=dtype)
        state = train_single_process_streaming(
            initial_model,
            module_path,
            xs_cpu,
            ys_cpu,
            offload_policy=True if resident_suffix_count else [True, False, True],
            optimizer_name=optimizer_name,
            optimizer_kwargs=optimizer_kwargs_for(optimizer_name),
            max_grad_norm=max_grad_norm,
            steps=2,
            device=device,
            resident_suffix_count=resident_suffix_count,
        ).state_dict()
        torch.save({key: value.detach().cpu() for key, value in state.items()}, result_file)
    finally:
        _destroy_distributed_if_needed()


def _single_gpu_model_and_path(container_kind: str, dtype: torch.dtype) -> tuple[nn.Module, str]:
    """Build the small model variant used by one-GPU equivalence cases."""
    if container_kind == "modulelist":
        return SandwichModel(dtype=dtype), "layers"
    if container_kind == "sequential":
        return SequentialSandwich(dtype=dtype), "blocks"
    raise ValueError(f"unknown container kind: {container_kind}")


def run_kwarg_reference_state(result_file: Path) -> None:
    """Write ordinary kwarg/nested-output training state for pytest-side comparison."""
    if _rank() != 0:
        return
    dtype = torch.float32
    device = _device()
    torch.manual_seed(20260610)
    initial_model = KwargSandwichModel(dtype=dtype)
    x, mask, context, y = make_kwarg_batches(dtype)
    optimizer_kwargs = optimizer_kwargs_for("adamw")
    criterion = nn.MSELoss()
    reference = KwargSandwichModel(dtype=dtype).to(device)
    reference.load_state_dict(initial_model.state_dict(), strict=True)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), **optimizer_kwargs)
    for _ in range(2):
        reference_optimizer.zero_grad(set_to_none=True)
        loss = criterion(reference(x.to(device), mask.to(device), context.to(device), scale=0.7), y.to(device))
        loss.backward()
        clip_grad_norm_(reference.parameters(), 0.4, foreach=False)
        reference_optimizer.step()
    torch.save({key: value.detach().cpu() for key, value in reference.state_dict().items()}, result_file)


def run_kwarg_streaming_state(result_file: Path) -> None:
    """Write streamed kwarg/nested-output training state for pytest-side comparison."""
    if _rank() != 0:
        return
    _init_distributed_if_needed()
    try:
        dtype = torch.float32
        device = _device()
        torch.manual_seed(20260610)
        initial_model = KwargSandwichModel(dtype=dtype)
        x, mask, context, y = make_kwarg_batches(dtype)
        optimizer_kwargs = optimizer_kwargs_for("adamw")
        criterion = nn.MSELoss()
        streaming_model = KwargSandwichModel(dtype=dtype)
        streaming_model.load_state_dict(initial_model.state_dict(), strict=True)
        engine = apply_cpu_streaming_(
            streaming_model,
            "decoder.layers",
            offload_policy=True,
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs=optimizer_kwargs,
            max_grad_norm=0.4,
            device=device,
        )
        assert isinstance(streaming_model.decoder.layers, CPUStreamingModuleList)
        for _ in range(2):
            engine.zero_grad(set_to_none=True)
            loss = criterion(engine.model(x.to(device), mask.to(device), context.to(device), scale=0.7), y.to(device))
            loss.backward()
            engine.step()
        closed = engine.close(return_on_all_ranks=True, device=torch.device("cpu"))
        assert closed is not None
        torch.save({key: value.detach().cpu() for key, value in closed.state_dict().items()}, result_file)
    finally:
        _destroy_distributed_if_needed()


def run_transfer_timing(result_file: Path) -> None:
    """Verify CUDA transfer timing records streamed parameter and gradient copies."""
    if _rank() != 0:
        return
    _init_distributed_if_needed()
    try:
        device = _device()
        model = SandwichModel(dtype=torch.float32)
        engine = apply_cpu_streaming_(
            model,
            "layers",
            offload_policy=True,
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs=optimizer_kwargs_for("adamw"),
            device=device,
            collect_timing=True,
        )

        criterion = nn.MSELoss()
        x = torch.randn(4, 5, device=device)
        y = torch.randn(4, 3, device=device)
        engine.zero_grad(set_to_none=True)
        loss = criterion(engine.model(x), y)
        loss.backward()
        engine.step()

        timings = engine.transfer_timing_summary(synchronize=True)
        closed = engine.close(return_on_all_ranks=True, device=torch.device("cpu"))
        assert closed is not None

        for kind in ("state_h2d", "grad_d2h", "optimizer_param_h2d", "optimizer_param_d2h"):
            assert kind in timings
            assert timings[kind]["calls"] > 0
            assert timings[kind]["bytes"] > 0
            assert timings[kind]["enqueue_ms"] >= 0.0
        assert any(handle._prefetch_streams for handle in engine.handles)
        torch.save({"ok": True}, result_file)
    finally:
        _destroy_distributed_if_needed()


def run_shared_cpu_storage(result_file: Path) -> None:
    """Verify CUDA torchrun ranks share offloaded CPU master storage through /dev/shm."""
    device = _device()
    try:
        dist.init_process_group(backend="nccl", device_id=device)
    except TypeError:
        dist.init_process_group(backend="nccl")
    try:
        model = SandwichModel(dtype=torch.float32)
        engine = apply_cpu_streaming_(
            model,
            "layers",
            offload_policy=True,
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs=optimizer_kwargs_for("adamw"),
            device=device,
        )
        first_parameter = next(iter(engine.handles[0].parameters_by_name.values()))
        shared_dir = engine.config.shared_cpu_dir
        assert shared_dir is not None
        if _rank() == 0:
            first_parameter.data.fill_(123.0)
        dist.barrier()
        observed = float(first_parameter.detach().flatten()[0].item())
        criterion = nn.MSELoss()
        engine.zero_grad(set_to_none=True)
        x = torch.randn(4, 5, device=device)
        y = torch.randn(4, 3, device=device)
        loss = criterion(engine.model(x), y)
        loss.backward()
        engine.step()
        dist.barrier()
        gradient_paths = [
            os.path.join(shared_dir, _SharedCpuStore._safe_name(handle._shared_key("gradient", name)))
            for handle in engine.handles
            for name in handle.param_names
        ]
        local_gradient_paths = [
            os.path.join(shared_dir, _SharedCpuStore._safe_name(handle._local_gradient_key(name)))
            for handle in engine.handles
            for name in handle.param_names
        ]
        torch.save(
            {
                "rank": _rank(),
                "observed": observed,
                "is_shm": os.path.commonpath([os.path.realpath("/dev/shm"), os.path.realpath(shared_dir)])
                == os.path.realpath("/dev/shm"),
                "gradient_files_remaining": [path for path in gradient_paths if os.path.exists(path)],
                "local_gradient_files_remaining": [path for path in local_gradient_paths if os.path.exists(path)],
            },
            f"{result_file}.{_rank()}",
        )
        engine.close(return_on_all_ranks=True, device=torch.device("cpu"))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    """Dispatch torchrun worker cases used by pytest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("--result-file", required=True, type=Path)
    parser.add_argument("--container-kind", default="modulelist")
    parser.add_argument("--optimizer-name", default="adamw")
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--resident-suffix-count", type=int, default=0)
    args = parser.parse_args()

    if args.case == "streaming-ddp-state":
        run_streaming_ddp_state(args.result_file)
    elif args.case == "standard-state":
        run_standard_state(args.result_file)
    elif args.case == "single-gpu-reference-state":
        run_single_gpu_reference_state(
            args.result_file,
            container_kind=args.container_kind,
            optimizer_name=args.optimizer_name,
            max_grad_norm=args.max_grad_norm,
        )
    elif args.case == "single-gpu-streaming-state":
        run_single_gpu_streaming_state(
            args.result_file,
            container_kind=args.container_kind,
            optimizer_name=args.optimizer_name,
            max_grad_norm=args.max_grad_norm,
            resident_suffix_count=args.resident_suffix_count,
        )
    elif args.case == "kwarg-reference-state":
        run_kwarg_reference_state(args.result_file)
    elif args.case == "kwarg-streaming-state":
        run_kwarg_streaming_state(args.result_file)
    elif args.case == "transfer-timing":
        run_transfer_timing(args.result_file)
    elif args.case == "shared-cpu-storage":
        run_shared_cpu_storage(args.result_file)
    else:
        raise SystemExit(f"unknown case: {args.case}")


if __name__ == "__main__":
    main()
