import copy
import os
import sys
import tempfile
from pathlib import Path
from types import MethodType
from typing import Dict, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The CPU DDP test launches multiple processes.  Keeping CPU math to one thread
# per process prevents small CI machines from oversubscribing hundreds of threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")

import pytest
import torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # PyTorch allows this to be set only before parallel work has started.
    pass
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_

from dtai_parallel import (
    CPUStreamingModuleList,
    CPUStreamingSequential,
    apply_cpu_streaming_,
    apply_cpu_streaming_to_modulelist_,
)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.lin1 = nn.Linear(width, width, dtype=dtype)
        self.act = nn.GELU()
        self.lin2 = nn.Linear(width, width, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return x + 0.25 * self.lin2(self.act(self.lin1(x)))


class SandwichModel(nn.Module):
    """Embedding and unembedding around a ModuleList of decoder-ish blocks."""

    def __init__(self, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        self.embed = nn.Linear(5, 8, dtype=dtype)
        self.layers = nn.ModuleList([ResidualBlock(8, dtype=dtype) for _ in range(3)])
        self.norm = nn.LayerNorm(8, dtype=dtype)
        self.unembed = nn.Linear(8, 3, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(x)
        # This is the intended user-facing case: after transformation, self.layers
        # is still iterable, so existing ModuleList-based code keeps working.
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.unembed(x)


class SequentialSandwich(nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        self.in_proj = nn.Linear(5, 8, dtype=dtype)
        self.blocks = nn.Sequential(
            ResidualBlock(8, dtype=dtype),
            ResidualBlock(8, dtype=dtype),
            ResidualBlock(8, dtype=dtype),
        )
        self.out_proj = nn.Linear(8, 3, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_proj(self.blocks(self.in_proj(x)))


class KwargDecoderBlock(nn.Module):
    """A block whose forward has multiple args, kwargs, and nested outputs."""

    def __init__(self, width: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.in_proj = nn.Linear(width, width, dtype=dtype)
        self.mix = nn.Linear(width, width, dtype=dtype)
        self.norm = nn.LayerNorm(width, dtype=dtype)

    def forward(
        self,
        hidden: Tensor,
        mask: Tensor,
        additive: Optional[Tensor] = None,
        *,
        scale: float = 1.0,
        metadata: Optional[Mapping[str, object]] = None,
        return_aux: bool = True,
    ):
        del metadata
        update = self.mix(torch.tanh(self.in_proj(hidden))) * mask
        if additive is not None:
            update = update + additive
        out = self.norm(hidden + float(scale) * update)
        aux = {
            "mean": out.mean(dim=-1),
            "positive_mask": mask > 0.0,  # a non-differentiable tensor leaf
        }
        return (out, aux) if return_aux else out


class NestedDecoder(nn.Module):
    def __init__(self, width: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.layers = nn.ModuleList([KwargDecoderBlock(width, dtype=dtype) for _ in range(2)])


class KwargSandwichModel(nn.Module):
    """Persistent resident modules around a nested decoder ModuleList."""

    def __init__(self, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        self.embed = nn.Linear(5, 8, dtype=dtype)
        self.context = nn.Linear(4, 8, dtype=dtype)
        self.decoder = NestedDecoder(8, dtype=dtype)
        self.unembed = nn.Linear(8, 3, dtype=dtype)

    def forward(self, x: Tensor, mask: Tensor, context: Tensor, *, scale: float = 0.5) -> Tensor:
        hidden = self.embed(x)
        additive = 0.05 * self.context(context)
        aux_penalty = hidden.new_zeros(())
        for index, layer in enumerate(self.decoder.layers):
            hidden, aux = layer(
                hidden,
                mask,
                additive,
                scale=scale,
                metadata={"layer_index": index, "note": "non-tensor kwargs are replay constants"},
                return_aux=True,
            )
            # This makes a nested tensor output participate in the loss, so the
            # custom autograd boundary must handle more than the first output.
            aux_penalty = aux_penalty + aux["mean"].mean()
        return self.unembed(hidden) + 1e-3 * aux_penalty


def make_batches(world_size: int, dtype: torch.dtype) -> Tuple[Sequence[Tensor], Sequence[Tensor]]:
    generator = torch.Generator(device="cpu").manual_seed(20260107)
    xs = [torch.randn(4, 5, generator=generator, dtype=dtype) for _ in range(world_size)]
    ys = [torch.randn(4, 3, generator=generator, dtype=dtype) for _ in range(world_size)]
    return xs, ys


def make_kwarg_batches(dtype: torch.dtype):
    generator = torch.Generator(device="cpu").manual_seed(20260108)
    x = torch.randn(4, 5, generator=generator, dtype=dtype)
    mask = torch.sigmoid(torch.randn(4, 8, generator=generator, dtype=dtype))
    context = torch.randn(4, 4, generator=generator, dtype=dtype)
    y = torch.randn(4, 3, generator=generator, dtype=dtype)
    return x, mask, context, y


def optimizer_kwargs_for(name: str) -> Dict[str, object]:
    if name == "adamw":
        return {"lr": 3e-3, "betas": (0.8, 0.9), "eps": 1e-8, "weight_decay": 0.01, "foreach": False}
    if name == "sgd":
        return {"lr": 1e-2, "momentum": 0.2, "weight_decay": 0.01, "foreach": False}
    raise AssertionError(name)


def optimizer_cls_for(name: str):
    if name == "adamw":
        return torch.optim.AdamW
    if name == "sgd":
        return torch.optim.SGD
    raise AssertionError(name)


def assert_state_dicts_close(actual: Mapping[str, Tensor], expected: Mapping[str, Tensor], *, dtype: torch.dtype) -> None:
    assert actual.keys() == expected.keys()
    if dtype == torch.float64:
        rtol, atol = 2e-9, 2e-10
    else:
        rtol, atol = 5e-5, 5e-6
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=rtol, atol=atol, msg=lambda msg: f"{key}: {msg}")


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
) -> nn.Module:
    model = copy.deepcopy(initial_model)
    engine = apply_cpu_streaming_(
        model,
        module_path,
        offload_policy=offload_policy,
        optimizer_cls=optimizer_cls_for(optimizer_name),
        optimizer_kwargs=optimizer_kwargs,
        max_grad_norm=max_grad_norm,
        device=device,
        wrap_ddp=False,
        auto_init_process_group=False,
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


@pytest.mark.parametrize("container_kind", ["modulelist", "sequential"])
@pytest.mark.parametrize("optimizer_name", ["adamw", "sgd"])
@pytest.mark.parametrize("max_grad_norm", [None, 0.35])
def test_in_place_transform_matches_single_process_reference(container_kind: str, optimizer_name: str, max_grad_norm: Optional[float]) -> None:
    dtype = torch.float64
    device = torch.device("cpu")
    if container_kind == "modulelist":
        initial_model: nn.Module = SandwichModel(dtype=dtype)
        module_path = "layers"
    else:
        initial_model = SequentialSandwich(dtype=dtype)
        module_path = "blocks"

    xs_cpu, ys_cpu = make_batches(world_size=2, dtype=dtype)
    optimizer_kwargs = optimizer_kwargs_for(optimizer_name)

    reference = train_single_process_reference(
        initial_model,
        xs_cpu,
        ys_cpu,
        optimizer_name=optimizer_name,
        optimizer_kwargs=optimizer_kwargs,
        max_grad_norm=max_grad_norm,
        steps=2,
        device=device,
    )
    streaming = train_single_process_streaming(
        initial_model,
        module_path,
        xs_cpu,
        ys_cpu,
        offload_policy=[True, False, True],
        optimizer_name=optimizer_name,
        optimizer_kwargs=optimizer_kwargs,
        max_grad_norm=max_grad_norm,
        steps=2,
        device=device,
    )
    assert_state_dicts_close(streaming.state_dict(), reference.state_dict(), dtype=dtype)


def test_transformation_keeps_modulelist_iteration_api() -> None:
    model = SandwichModel(dtype=torch.float64)
    engine = apply_cpu_streaming_(
        model,
        "layers",
        offload_policy=[True, False, True],
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs=optimizer_kwargs_for("adamw"),
        device=torch.device("cpu"),
        auto_init_process_group=False,
        wrap_ddp=False,
    )
    assert isinstance(model.layers, CPUStreamingModuleList)
    assert len(model.layers) == 3
    x = torch.randn(2, 5, dtype=torch.float64)
    out = engine.model(x)
    assert out.shape == (2, 3)
    closed = engine.close(return_on_all_ranks=True, device=torch.device("cpu"))
    assert closed is not None
    assert isinstance(closed.layers, nn.ModuleList)
    assert not isinstance(closed.layers, CPUStreamingModuleList)


def test_transformation_supports_sequential_call_api() -> None:
    model = SequentialSandwich(dtype=torch.float64)
    engine = apply_cpu_streaming_to_modulelist_(
        model,
        "blocks",
        offload_policy=True,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs=optimizer_kwargs_for("adamw"),
        device=torch.device("cpu"),
        auto_init_process_group=False,
        wrap_ddp=False,
    )
    assert isinstance(model.blocks, CPUStreamingSequential)
    x = torch.randn(2, 5, dtype=torch.float64)
    out = engine.model(x)
    assert out.shape == (2, 3)


def test_nested_modulelist_supports_arbitrary_args_kwargs_and_nested_outputs() -> None:
    dtype = torch.float64
    device = torch.device("cpu")
    initial_model = KwargSandwichModel(dtype=dtype)
    x, mask, context, y = make_kwarg_batches(dtype)
    optimizer_kwargs = optimizer_kwargs_for("adamw")
    criterion = nn.MSELoss()

    reference = copy.deepcopy(initial_model).to(device)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), **optimizer_kwargs)
    for _ in range(2):
        reference_optimizer.zero_grad(set_to_none=True)
        reference_loss = criterion(reference(x.to(device), mask.to(device), context.to(device), scale=0.7), y.to(device))
        reference_loss.backward()
        clip_grad_norm_(reference.parameters(), 0.4, foreach=False)
        reference_optimizer.step()

    streaming_model = copy.deepcopy(initial_model)
    engine = apply_cpu_streaming_(
        streaming_model,
        "decoder.layers",
        offload_policy=True,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs=optimizer_kwargs,
        max_grad_norm=0.4,
        device=device,
        auto_init_process_group=False,
        wrap_ddp=False,
    )
    assert isinstance(streaming_model.decoder.layers, CPUStreamingModuleList)
    for _ in range(2):
        engine.zero_grad(set_to_none=True)
        streaming_loss = criterion(engine.model(x.to(device), mask.to(device), context.to(device), scale=0.7), y.to(device))
        streaming_loss.backward()
        engine.step()
    closed = engine.close(return_on_all_ranks=True, device=torch.device("cpu"))
    assert closed is not None

    assert_state_dicts_close(closed.state_dict(), reference.cpu().state_dict(), dtype=dtype)


def test_forward_and_backward_prefetch_are_scheduled() -> None:
    model = SandwichModel(dtype=torch.float64)
    engine = apply_cpu_streaming_(
        model,
        "layers",
        offload_policy=[True, False, True],
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs=optimizer_kwargs_for("adamw"),
        device=torch.device("cpu"),
        auto_init_process_group=False,
        wrap_ddp=False,
    )

    counts: Dict[Tuple[str, bool], int] = {}
    for handle in engine.handles:
        original = handle.prefetch

        def counted_prefetch(self, device, *, requires_grad: bool, _original=original):
            counts[(self.qualified_name, bool(requires_grad))] = counts.get((self.qualified_name, bool(requires_grad)), 0) + 1
            return _original(device, requires_grad=requires_grad)

        handle.prefetch = MethodType(counted_prefetch, handle)

    engine.zero_grad(set_to_none=True)
    loss = engine.model(torch.randn(4, 5, dtype=torch.float64)).pow(2).mean()
    loss.backward()

    assert counts.get(("layers.2", False), 0) >= 1, "forward should prefetch a later offloaded stage"
    assert counts.get(("layers.0", True), 0) >= 1, "backward should prefetch the previous offloaded stage"


@pytest.mark.cuda
def test_cuda_transfer_timing_records_streamed_copies() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model = SandwichModel(dtype=torch.float32)
    engine = apply_cpu_streaming_(
        model,
        "layers",
        offload_policy=True,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs=optimizer_kwargs_for("adamw"),
        device=device,
        auto_init_process_group=False,
        wrap_ddp=False,
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

    for kind in ("state_h2d", "optimizer_param_h2d", "optimizer_param_d2h"):
        assert kind in timings
        assert timings[kind]["calls"] > 0
        assert timings[kind]["bytes"] > 0
        assert timings[kind]["enqueue_ms"] >= 0.0
    assert "grad_d2h" not in timings

    assert any(handle._prefetch_streams for handle in engine.handles)


def _reference_ddp_worker(
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
        if use_cuda:
            torch.cuda.set_device(device)
        model = SandwichModel(dtype=dtype).to(device)
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
                clip_grad_norm_(ddp.parameters(), max_grad_norm, foreach=False)
            optimizer.step()
        dist.barrier()
        if rank == 0:
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, result_file)
    finally:
        dist.destroy_process_group()


def _standard_training_worker(
    rank: int,
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
    del rank
    device = torch.device("cuda:0") if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    model = SandwichModel(dtype=dtype).to(device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = optimizer_cls_for(optimizer_name)(model.parameters(), **dict(optimizer_kwargs))
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
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, result_file)


def _streaming_ddp_worker(
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
        if use_cuda:
            torch.cuda.set_device(device)
        model = SandwichModel(dtype=dtype)
        model.load_state_dict(initial_state, strict=True)
        engine = apply_cpu_streaming_(
            model,
            "layers",
            offload_policy=[True, False, True],
            optimizer_cls=optimizer_cls_for(optimizer_name),
            optimizer_kwargs=optimizer_kwargs,
            max_grad_norm=max_grad_norm,
            device=device,
            auto_init_process_group=False,
            wrap_ddp=True,
        )
        assert isinstance(engine.model, DDP)
        criterion = nn.MSELoss()
        for _ in range(steps):
            engine.zero_grad(set_to_none=True)
            x = xs_cpu[rank].to(device)
            y = ys_cpu[rank].to(device)
            loss = criterion(engine.model(x), y)
            loss.backward()
            engine.step()
        closed = engine.close(device=torch.device("cpu"))
        if rank == 0:
            assert closed is not None
            torch.save({k: v.detach().cpu() for k, v in closed.state_dict().items()}, result_file)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("device_kind", ["cpu", "cuda"])
def test_streamed_modulelist_inside_larger_model_matches_real_ddp(device_kind: str) -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    if device_kind == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() < 2):
        pytest.skip("needs at least two CUDA devices")

    use_cuda = device_kind == "cuda"
    dtype = torch.float32 if use_cuda else torch.float64
    world_size = 2
    initial_model = SandwichModel(dtype=dtype)
    xs_cpu, ys_cpu = make_batches(world_size=world_size, dtype=dtype)
    optimizer_name = "adamw"
    optimizer_kwargs = optimizer_kwargs_for(optimizer_name)
    max_grad_norm = 0.5
    steps = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        init_ref = os.path.join(tmpdir, "ref_init")
        init_stream = os.path.join(tmpdir, "stream_init")
        result_ref = os.path.join(tmpdir, "ref.pt")
        result_stream = os.path.join(tmpdir, "stream.pt")
        for path in (init_ref, init_stream):
            if os.path.exists(path):
                os.unlink(path)

        mp.start_processes(
            _reference_ddp_worker,
            args=(
                world_size,
                init_ref,
                result_ref,
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
        mp.start_processes(
            _streaming_ddp_worker,
            args=(
                world_size,
                init_stream,
                result_stream,
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
        ref_state = torch.load(result_ref, map_location="cpu")
        stream_state = torch.load(result_stream, map_location="cpu")

    assert_state_dicts_close(stream_state, ref_state, dtype=dtype)


@pytest.mark.parametrize("device_kind", ["cpu", "cuda"])
def test_streamed_modulelist_inside_larger_model_matches_standard_training_loop(device_kind: str) -> None:
    if device_kind == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() < 2):
        pytest.skip("needs at least two CUDA devices")

    use_cuda = device_kind == "cuda"
    dtype = torch.float32 if use_cuda else torch.float64
    world_size = 2
    initial_model = SandwichModel(dtype=dtype)
    xs_cpu, ys_cpu = make_batches(world_size=world_size, dtype=dtype)
    optimizer_name = "adamw"
    optimizer_kwargs = optimizer_kwargs_for(optimizer_name)
    max_grad_norm = 0.5
    steps = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        init_stream = os.path.join(tmpdir, "stream_init")
        result_ref = os.path.join(tmpdir, "ref.pt")
        result_stream = os.path.join(tmpdir, "stream.pt")
        if os.path.exists(init_stream):
            os.unlink(init_stream)

        mp.start_processes(
            _standard_training_worker,
            args=(
                result_ref,
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
            nprocs=1,
            start_method="spawn",
            join=True,
        )
        mp.start_processes(
            _streaming_ddp_worker,
            args=(
                world_size,
                init_stream,
                result_stream,
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
        ref_state = torch.load(result_ref, map_location="cpu")
        stream_state = torch.load(result_stream, map_location="cpu")

    assert_state_dicts_close(stream_state, ref_state, dtype=dtype)
