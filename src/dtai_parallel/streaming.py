"""In-place CPU streaming for an existing model's ``ModuleList`` or ``Sequential``.

The public API is a transformation, not a base class:

    model = TransformerLikeModel(...)
    engine = apply_cpu_streaming_(model, "decoder.layers", ...)

    for tokens, targets in loader:
        engine.zero_grad()
        loss = criterion(engine.model(tokens.to(engine.local_device)), targets.to(engine.local_device))
        loss.backward()
        engine.step()

    ordinary_model = engine.close()

The model object remains the model written by the user.  The transformation only
replaces one ordered submodule, usually a list of decoder blocks, with an ordered
container of stage wrappers.  Existing forward code such as

    for layer in self.decoder.layers:
        hidden = layer(hidden, attention_mask=mask, rotary=rotary, use_cache=False)

continues to work because the replacement is still iterable.  If the target is an
``nn.Sequential``, calling it directly also continues to work.

The code requires the usual ``torchrun`` layout, including one-GPU runs: one
Python process owns one local GPU and has an initialized process group.  Each
process forwards only its local batch.  Resident parameters remain ordinary
registered parameters.  The engine always wraps the transformed model in
``DistributedDataParallel`` so resident parameters use normal PyTorch DDP.  The
parameters in offloaded stages are hidden from the module tree, kept as
file-backed CPU masters, streamed asynchronously to the local device, and
synchronized by the engine.

The comments and docstrings are intentionally explanatory.  The implementation
still includes the requested practical mechanisms: arbitrary positional and
keyword arguments containing tensors, nested tensor outputs, autograd replay for
streamed parameters, asynchronous CUDA prefetch streams, DDP for resident modules,
optimizer dispatch to PyTorch optimizer classes, integrated gradient clipping,
and materialization back to an ordinary model in ``close()``.
"""

from __future__ import annotations

import copy
import hashlib
import os
import shutil
import socket
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Type, Union

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:  # PyTorch >= 2.0.  The tests in this project run on PyTorch 2.10.
    from torch.func import functional_call
except Exception:  # pragma: no cover
    from torch.nn.utils.stateless import functional_call  # type: ignore


DeviceLike = Union[str, torch.device]
OffloadPolicy = Union[bool, Sequence[bool], Callable[[int, str, nn.Module], bool]]
_SENTINEL = "__cpu_streaming_ddp_tensor_leaf__"


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class StreamingConfig:
    """A compact record of the engine configuration.

    The dataclass is not required by the algorithm.  It is useful in tests,
    checkpoints, logs, and examples because it records which submodule was
    transformed, which device this torchrun process owns, which optimizer class is
    being dispatched, and whether DDP was enabled for resident parameters.
    """

    module_path: str
    device: torch.device
    optimizer_cls: Type[torch.optim.Optimizer]
    optimizer_kwargs: Mapping[str, Any]
    max_grad_norm: Optional[float]
    grad_norm_type: float
    ddp_enabled: bool
    world_size: int
    rank: int
    shared_cpu_dir: Optional[str]


class StreamingTransferMetrics:
    """Low-overhead transfer timing counters collected by the streaming engine."""

    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self._lock = threading.Lock()
        self._counters: Dict[str, Dict[str, float]] = {}
        self._pending_cuda_events: List[Tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    def _counter_for(self, kind: str) -> Dict[str, float]:
        counter = self._counters.get(kind)
        if counter is None:
            counter = {"calls": 0.0, "bytes": 0.0, "enqueue_ms": 0.0, "cuda_ms": 0.0}
            self._counters[kind] = counter
        return counter

    def record_copy(self, kind: str, byte_count: int, device: torch.device, copy_fn: Callable[[], Any]) -> Any:
        if not self.enabled:
            return copy_fn()

        start_event: Optional[torch.cuda.Event] = None
        end_event: Optional[torch.cuda.Event] = None
        if device.type == "cuda":
            stream = torch.cuda.current_stream(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(stream)

        start = time.perf_counter()
        result = copy_fn()
        enqueue_ms = (time.perf_counter() - start) * 1000.0

        with self._lock:
            counter = self._counter_for(kind)
            counter["calls"] += 1.0
            counter["bytes"] += float(byte_count)
            counter["enqueue_ms"] += enqueue_ms
            if start_event is not None and end_event is not None:
                end_event.record(torch.cuda.current_stream(device))
                self._pending_cuda_events.append((kind, start_event, end_event))

        return result

    def _finalize_cuda_events(self, *, synchronize: bool) -> None:
        if not self.enabled:
            return
        with self._lock:
            pending = self._pending_cuda_events
            self._pending_cuda_events = []

        still_pending: List[Tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        completed: List[Tuple[str, float]] = []
        for kind, start_event, end_event in pending:
            if synchronize:
                end_event.synchronize()
            elif not end_event.query():
                still_pending.append((kind, start_event, end_event))
                continue
            completed.append((kind, float(start_event.elapsed_time(end_event))))

        with self._lock:
            for kind, cuda_ms in completed:
                self._counter_for(kind)["cuda_ms"] += cuda_ms
            self._pending_cuda_events.extend(still_pending)

    def summary(self, *, reset: bool = False, synchronize: bool = True) -> Dict[str, Dict[str, float]]:
        self._finalize_cuda_events(synchronize=synchronize)
        with self._lock:
            snapshot = {kind: dict(values) for kind, values in sorted(self._counters.items())}
            if reset:
                self._counters = {}
                self._pending_cuda_events = []
        return snapshot


class _Tree:
    """Move opaque PyTorch optimizer state between devices.

    AdamW, Adam, SGD with momentum, and many other PyTorch optimizers store state
    in nested Python containers.  The streaming engine deliberately treats that
    state as an opaque tree.  Tensor leaves are moved and cloned; non-tensor leaves
    are deep-copied.  This is the small piece that lets the engine dispatch to
    PyTorch optimizer classes instead of reimplementing optimizer equations.
    """

    @staticmethod
    def to_device(
        value: Any,
        device: torch.device,
        *,
        metrics: Optional[StreamingTransferMetrics] = None,
        kind: str = "optimizer_state_h2d",
        pinned_transfer_buffer: "_PinnedTransferBuffer",
    ) -> Any:
        if torch.is_tensor(value):
            return _copy_tensor_to_device(
                value.detach(),
                device,
                metrics=metrics,
                kind=kind,
                pinned_transfer_buffer=pinned_transfer_buffer,
            )
        if isinstance(value, dict):
            return {
                k: _Tree.to_device(
                    v,
                    device,
                    metrics=metrics,
                    kind=kind,
                    pinned_transfer_buffer=pinned_transfer_buffer,
                )
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [
                _Tree.to_device(
                    v,
                    device,
                    metrics=metrics,
                    kind=kind,
                    pinned_transfer_buffer=pinned_transfer_buffer,
                )
                for v in value
            ]
        if isinstance(value, tuple):
            return tuple(
                _Tree.to_device(
                    v,
                    device,
                    metrics=metrics,
                    kind=kind,
                    pinned_transfer_buffer=pinned_transfer_buffer,
                )
                for v in value
            )
        return copy.deepcopy(value)

    @staticmethod
    def detach_cpu(
        value: Any,
        *,
        metrics: Optional[StreamingTransferMetrics] = None,
        kind: str = "optimizer_state_d2h",
        shared_cpu_store: "_SharedCpuStore",
        shared_key: str = "optimizer_state",
    ) -> Any:
        if torch.is_tensor(value):
            out = shared_cpu_store.tensor_like(shared_key, value.detach())
            byte_count = _tensor_nbytes(value)

            def copy_fn() -> Tensor:
                out.copy_(value.detach(), non_blocking=False)
                return out

            if metrics is None:
                return copy_fn()
            return metrics.record_copy(kind, byte_count, value.device, copy_fn)
        if isinstance(value, dict):
            return {
                k: _Tree.detach_cpu(
                    v,
                    metrics=metrics,
                    kind=kind,
                    shared_cpu_store=shared_cpu_store,
                    shared_key=f"{shared_key}.{k}",
                )
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [
                _Tree.detach_cpu(
                    v,
                    metrics=metrics,
                    kind=kind,
                    shared_cpu_store=shared_cpu_store,
                    shared_key=f"{shared_key}.{index}",
                )
                for index, v in enumerate(value)
            ]
        if isinstance(value, tuple):
            return tuple(
                _Tree.detach_cpu(
                    v,
                    metrics=metrics,
                    kind=kind,
                    shared_cpu_store=shared_cpu_store,
                    shared_key=f"{shared_key}.{index}",
                )
                for index, v in enumerate(value)
            )
        return copy.deepcopy(value)

    @staticmethod
    def record_stream(value: Any, stream: torch.cuda.Stream) -> None:
        if torch.is_tensor(value):
            if value.device.type == "cuda":
                value.record_stream(stream)
            return
        if isinstance(value, dict):
            for item in value.values():
                _Tree.record_stream(item, stream)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                _Tree.record_stream(item, stream)

def _tensor_nbytes(tensor: Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


class _PinnedTransferBuffer:
    """Reusable pinned CPU staging buffers for pageable-CPU to CUDA copies."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buffers: Dict[torch.dtype, Tensor] = {}
        self._events: Dict[torch.dtype, torch.cuda.Event] = {}

    def copy_to_device(self, tensor: Tensor, device: torch.device) -> Tensor:
        if tensor.device.type != "cpu" or device.type != "cuda":
            return tensor.to(device=device, non_blocking=True, copy=True)

        flat = tensor.detach().reshape(-1)
        dtype = flat.dtype
        with self._lock:
            event = self._events.pop(dtype, None)
            if event is not None:
                event.synchronize()

            buffer = self._buffers.get(dtype)
            if buffer is None or buffer.numel() < flat.numel():
                buffer = torch.empty(flat.numel(), dtype=dtype, device=torch.device("cpu"), pin_memory=True)
                self._buffers[dtype] = buffer

            staged = buffer[: flat.numel()]
            staged.copy_(flat)
            result = staged.view(tensor.shape).to(device=device, non_blocking=True, copy=True)

            done = torch.cuda.Event()
            done.record(torch.cuda.current_stream(device))
            self._events[dtype] = done
            return result


def _copy_tensor_to_device(
    tensor: Tensor,
    device: torch.device,
    *,
    metrics: Optional[StreamingTransferMetrics],
    kind: str,
    pinned_transfer_buffer: _PinnedTransferBuffer,
) -> Tensor:
    byte_count = _tensor_nbytes(tensor)

    def copy_fn() -> Tensor:
        if tensor.device.type == "cpu" and device.type == "cuda":
            return pinned_transfer_buffer.copy_to_device(tensor, device)
        return tensor.to(device=device, non_blocking=True, copy=True)

    if metrics is None:
        return copy_fn()
    return metrics.record_copy(kind, byte_count, device, copy_fn)


def _grad_total_norm(gradients: Iterable[Tensor], *, norm_type: float, device: torch.device) -> Optional[Tensor]:
    grads = [grad for grad in gradients if grad is not None]
    if not grads:
        return None
    if norm_type == float("inf"):
        return torch.stack([grad.detach().abs().max().to(device) for grad in grads]).max()
    return torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(grad.detach(), ord=norm_type).to(device) for grad in grads]),
        ord=norm_type,
    )


def _combine_grad_norms(norms: Iterable[Optional[Tensor]], *, norm_type: float, device: torch.device) -> Tensor:
    present = [norm.to(device) for norm in norms if norm is not None]
    if not present:
        return torch.zeros((), device=device)
    if norm_type == float("inf"):
        return torch.stack(present).max()
    return torch.linalg.vector_norm(torch.stack(present), ord=norm_type)


def _clip_coef(total_norm: Tensor, *, max_norm: float, error_if_nonfinite: bool) -> Tensor:
    if error_if_nonfinite and not bool(torch.isfinite(total_norm).item()):
        raise RuntimeError(
            "The total norm of order for gradients from `parameters` is non-finite, "
            "so it cannot be clipped. To disable this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    return torch.clamp(torch.as_tensor(float(max_norm), device=total_norm.device) / (total_norm + 1e-6), max=1.0)


class _RNGSnapshot:
    """Save enough RNG state to replay a stochastic layer during backward.

    Offloaded stages run their forward under ``no_grad`` and reconstruct the
    graph in backward by replaying the layer with streamed parameters that require
    gradients.  Dropout and similar stochastic operations must see the same random
    choices during replay, so the custom autograd boundary records CPU and CUDA
    RNG state before the forward call and temporarily restores it for replay.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.cpu_state = torch.random.get_rng_state()
        self.cuda_state: Optional[Tensor]
        if device.type == "cuda":
            self.cuda_state = torch.cuda.get_rng_state(device)
        else:
            self.cuda_state = None

    def fork(self) -> Any:
        snapshot = self

        class _Fork:
            def __enter__(self) -> None:
                self.prev_cpu = torch.random.get_rng_state()
                torch.random.set_rng_state(snapshot.cpu_state)
                if snapshot.cuda_state is not None:
                    self.prev_cuda = torch.cuda.get_rng_state(snapshot.device)
                    torch.cuda.set_rng_state(snapshot.cuda_state, snapshot.device)
                else:
                    self.prev_cuda = None

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
                torch.random.set_rng_state(self.prev_cpu)
                if self.prev_cuda is not None:
                    torch.cuda.set_rng_state(self.prev_cuda, snapshot.device)

        return _Fork()


# ---------------------------------------------------------------------------
# PyTree helpers for arbitrary args, kwargs, and nested outputs.
# ---------------------------------------------------------------------------


def _is_differentiable_tensor(value: Tensor) -> bool:
    return bool(value.is_floating_point() or value.is_complex())


class _TensorTreeSpec:
    """A small tensor-only pytree specification.

    PyTorch's internal pytree utilities are powerful, but the streaming boundary
    needs a slightly different split: tensor leaves travel through autograd, while
    non-tensor leaves are replay-time constants.  This helper records enough
    structure to rebuild positional arguments, keyword arguments, and nested
    tensor outputs.  Lists, tuples, namedtuples, dictionaries, and OrderedDicts are
    traversed.  Other Python objects are treated as constants.
    """

    def __init__(self, spec: Any, tensor_count: int) -> None:
        self.spec = spec
        self.tensor_count = int(tensor_count)

    @staticmethod
    def flatten(value: Any, *, tensor_predicate: Callable[[Tensor], bool]) -> Tuple[List[Tensor], "_TensorTreeSpec"]:
        tensors: List[Tensor] = []

        def visit(obj: Any) -> Any:
            if torch.is_tensor(obj) and tensor_predicate(obj):
                index = len(tensors)
                tensors.append(obj)
                return (_SENTINEL, index)
            if isinstance(obj, OrderedDict):
                return ("ordered_dict", [(k, visit(v)) for k, v in obj.items()])
            if isinstance(obj, dict):
                return ("dict", type(obj), [(k, visit(v)) for k, v in obj.items()])
            if isinstance(obj, list):
                return ("list", [visit(v) for v in obj])
            if isinstance(obj, tuple) and hasattr(type(obj), "_fields"):
                return ("namedtuple", type(obj), [visit(v) for v in obj])
            if isinstance(obj, tuple):
                return ("tuple", [visit(v) for v in obj])
            return ("const", obj)

        return tensors, _TensorTreeSpec(visit(value), len(tensors))

    def unflatten(self, tensors: Sequence[Tensor]) -> Any:
        if len(tensors) != self.tensor_count:
            raise RuntimeError(f"expected {self.tensor_count} tensor leaves, received {len(tensors)}")

        def build(spec: Any) -> Any:
            kind = spec[0]
            if kind == _SENTINEL:
                return tensors[spec[1]]
            if kind == "ordered_dict":
                return OrderedDict((k, build(v)) for k, v in spec[1])
            if kind == "dict":
                dict_type = spec[1]
                items = [(k, build(v)) for k, v in spec[2]]
                try:
                    return dict_type(items)
                except Exception:
                    return dict(items)
            if kind == "list":
                return [build(v) for v in spec[1]]
            if kind == "namedtuple":
                typ = spec[1]
                return typ(*[build(v) for v in spec[2]])
            if kind == "tuple":
                return tuple(build(v) for v in spec[1])
            if kind == "const":
                return spec[1]
            raise RuntimeError(f"unknown tensor tree spec node {kind!r}")

        return build(self.spec)


class _OutputHolder:
    """Mutable bridge from ``autograd.Function.forward`` back to the wrapper.

    ``torch.autograd.Function`` can return a tuple of tensors, but it cannot
    directly return an arbitrary Python pytree.  The forward method flattens the
    differentiable tensor leaves and stores the output tree specification here.
    The stage wrapper receives the tensor tuple from ``apply`` and reconstructs
    the user's original output structure.
    """

    def __init__(self) -> None:
        self.spec: Optional[_TensorTreeSpec] = None

    def flatten_output(self, output: Any) -> Tuple[Tensor, ...]:
        tensors, spec = _TensorTreeSpec.flatten(output, tensor_predicate=_is_differentiable_tensor)
        if not tensors:
            raise RuntimeError(
                "an offloaded stage must return at least one floating-point or complex tensor; "
                "non-differentiable tensor outputs may be nested beside differentiable outputs"
            )
        self.spec = spec
        return tuple(tensors)

    def reconstruct(self, result: Any) -> Any:
        if self.spec is None:
            raise RuntimeError("offloaded stage forward did not record an output structure")
        tensors = result if isinstance(result, tuple) else (result,)
        return self.spec.unflatten(tensors)


@dataclass
class _CallSpec:
    """Flattened representation of ``module(*args, **kwargs)``."""

    spec: _TensorTreeSpec
    tensor_requires_grad: Tuple[bool, ...]

    @classmethod
    def from_call(cls, args: Tuple[Any, ...], kwargs: Mapping[str, Any]) -> Tuple[List[Tensor], "_CallSpec"]:
        tensors, spec = _TensorTreeSpec.flatten((args, OrderedDict(kwargs.items())), tensor_predicate=torch.is_tensor)
        return tensors, cls(spec=spec, tensor_requires_grad=tuple(bool(t.requires_grad) for t in tensors))

    def rebuild(self, tensors: Sequence[Tensor]) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        args, kwargs_ordered = self.spec.unflatten(tensors)
        return tuple(args), dict(kwargs_ordered)


# ---------------------------------------------------------------------------
# Distributed helpers.
# ---------------------------------------------------------------------------


def _world_size(process_group: Optional[Any] = None) -> int:
    _require_process_group(process_group)
    return int(dist.get_world_size(process_group))


def _rank(process_group: Optional[Any] = None) -> int:
    _require_process_group(process_group)
    return int(dist.get_rank(process_group))


def _backend(process_group: Optional[Any] = None) -> Optional[str]:
    _require_process_group(process_group)
    return str(dist.get_backend(process_group)).lower()


def _require_process_group(process_group: Optional[Any] = None) -> None:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("CPU streaming requires torchrun or an initialized torch.distributed process group")


def _infer_local_device() -> torch.device:
    """Infer the single local device used by this torchrun process."""

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        visible = torch.cuda.device_count()
        index = local_rank if local_rank < visible else 0
        torch.cuda.set_device(index)
        return torch.device("cuda", index)
    return torch.device("cpu")


def _normalize_device(device: DeviceLike) -> torch.device:
    normalized = torch.device(device)
    if normalized.type == "cuda":
        index = normalized.index if normalized.index is not None else torch.cuda.current_device()
        torch.cuda.set_device(index)
        return torch.device("cuda", index)
    return normalized


def _init_process_group_from_torchrun(device: torch.device) -> None:
    """Initialize ``torch.distributed`` from torchrun environment variables.

    ``torchrun`` starts processes and sets ``RANK``, ``WORLD_SIZE``,
    ``MASTER_ADDR``, and ``MASTER_PORT``.  It does not call
    ``dist.init_process_group`` inside the program.  This helper initializes only
    when the relevant environment variables are present.
    """

    if not dist.is_available():
        raise RuntimeError("CPU streaming requires torch.distributed")
    if dist.is_initialized():
        return
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("CPU streaming requires torchrun environment variables RANK and WORLD_SIZE")
    backend = "nccl" if device.type == "cuda" else "gloo"
    if backend == "nccl":
        try:
            dist.init_process_group(backend=backend, device_id=device)
            return
        except TypeError:
            pass
    dist.init_process_group(backend=backend)


def _collective_tensor_device(tensor: Tensor, local_device: torch.device, process_group: Optional[Any]) -> torch.device:
    """Choose a legal device for collectives under the active backend."""

    backend = _backend(process_group)
    if backend == "nccl":
        if local_device.type != "cuda":
            raise RuntimeError("NCCL process groups require a CUDA local_device")
        return local_device
    return torch.device("cpu")


def _all_reduce_mean_(tensor: Tensor, local_device: torch.device, process_group: Optional[Any]) -> Tensor:
    """All-reduce ``tensor`` and return an averaged tensor on the original device."""

    _require_process_group(process_group)
    original_device = tensor.device
    comm_device = _collective_tensor_device(tensor, local_device, process_group)
    work = tensor.detach().to(comm_device, non_blocking=True).clone()
    dist.all_reduce(work, op=dist.ReduceOp.SUM, group=process_group)
    work.div_(float(dist.get_world_size(process_group)))
    if work.device != original_device:
        work = work.to(original_device, non_blocking=original_device.type != "cpu")
    return work


def _broadcast_tensor_(tensor: Tensor, src: int, local_device: torch.device, process_group: Optional[Any]) -> None:
    """Broadcast into ``tensor`` in place, using a backend-compatible staging tensor."""

    _require_process_group(process_group)
    comm_device = _collective_tensor_device(tensor, local_device, process_group)
    work = tensor.detach().to(comm_device, non_blocking=True).clone()
    dist.broadcast(work, src=src, group=process_group)
    if work.device != tensor.device:
        work = work.to(tensor.device, non_blocking=tensor.device.type != "cpu")
    tensor.data.copy_(work)


class _SharedCpuStore:
    """File-backed CPU tensors shared by same-host distributed ranks."""

    _SHM_ROOT = os.path.realpath("/dev/shm")

    def __init__(self, *, directory: str, process_group: Optional[Any], created_by_engine: bool) -> None:
        self.directory = directory
        self.process_group = process_group
        self.created_by_engine = bool(created_by_engine)
        self._storages: Dict[str, torch.UntypedStorage] = {}
        os.makedirs(self.directory, exist_ok=True)

    @classmethod
    def create(
        cls,
        *,
        requested_dir: Optional[str],
        process_group: Optional[Any],
    ) -> "_SharedCpuStore":
        _require_process_group(process_group)
        cls._validate_same_host(process_group)
        rank = _rank(process_group)
        created_by_engine = requested_dir is None

        if requested_dir is not None:
            directory = cls._validate_shm_path(requested_dir)
            if rank == 0:
                os.makedirs(directory, exist_ok=True)
        else:
            directory = ""
            if rank == 0:
                directory = tempfile.mkdtemp(prefix="dtai_parallel_", dir=cls._SHM_ROOT)

        payload = [directory]
        dist.broadcast_object_list(payload, src=0, group=process_group)
        directory = cls._validate_shm_path(str(payload[0]))
        dist.barrier(group=process_group)
        return cls(directory=directory, process_group=process_group, created_by_engine=created_by_engine)

    @classmethod
    def _validate_same_host(cls, process_group: Optional[Any]) -> None:
        hostname = socket.gethostname()
        hostnames: List[Optional[str]] = [None for _ in range(_world_size(process_group))]
        dist.all_gather_object(hostnames, hostname, group=process_group)
        unique = {str(value) for value in hostnames}
        if len(unique) != 1:
            raise RuntimeError(
                "distributed CPU offload requires all ranks to run on the same host; "
                f"observed hostnames: {sorted(unique)}"
            )

    @classmethod
    def _validate_shm_path(cls, path: str) -> str:
        real = os.path.realpath(os.path.abspath(path))
        try:
            common = os.path.commonpath([cls._SHM_ROOT, real])
        except ValueError as exc:
            raise ValueError(f"shared_cpu_dir must be under {cls._SHM_ROOT!r}; got {path!r}") from exc
        if common != cls._SHM_ROOT:
            raise ValueError(f"shared_cpu_dir must be under {cls._SHM_ROOT!r}; got {path!r}")
        return real

    @staticmethod
    def _safe_name(key: str) -> str:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return f"{digest}.bin"

    def barrier(self) -> None:
        dist.barrier(group=self.process_group)

    def tensor_like(self, key: str, like: Tensor) -> Tensor:
        reference = like.detach()
        nbytes = max(1, _tensor_nbytes(reference))
        path = os.path.join(self.directory, self._safe_name(key))
        storage = torch.UntypedStorage.from_file(path, shared=True, nbytes=nbytes)
        self._storages[key] = storage
        return torch.empty((), dtype=reference.dtype, device=torch.device("cpu")).set_(
            storage,
            0,
            tuple(reference.shape),
            tuple(reference.stride()),
        )

    def copy_tensor(self, key: str, source: Tensor, *, initialize: bool) -> Tensor:
        shared = self.tensor_like(key, source)
        if initialize:
            shared.copy_(source.detach().to(device=torch.device("cpu")), non_blocking=False)
        return shared

    def release_tensor(self, key: str) -> None:
        self._storages.pop(key, None)
        path = os.path.join(self.directory, self._safe_name(key))
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    def cleanup(self) -> None:
        if not self.created_by_engine:
            return
        if _rank(self.process_group) == 0:
            shutil.rmtree(self.directory, ignore_errors=True)


# ---------------------------------------------------------------------------
# Module-path and policy helpers.
# ---------------------------------------------------------------------------


def _resolve_module_path(root: nn.Module, path: str) -> Tuple[nn.Module, str, nn.Module]:
    """Return ``(parent, child_name, child)`` for a dotted module path."""

    if not path:
        raise ValueError("module_path must be a non-empty dotted path")
    parts = path.split(".")
    parent: nn.Module = root
    for part in parts[:-1]:
        if isinstance(parent, (nn.ModuleList, nn.Sequential)) and part.isdigit():
            parent = parent[int(part)]
        else:
            try:
                child = getattr(parent, part)
            except AttributeError as exc:
                raise ValueError(f"could not resolve module path {path!r}; missing {part!r}") from exc
            if not isinstance(child, nn.Module):
                raise ValueError(f"path component {part!r} is not an nn.Module")
            parent = child
    child_name = parts[-1]
    if isinstance(parent, (nn.ModuleList, nn.Sequential)) and child_name.isdigit():
        child = parent[int(child_name)]
    else:
        try:
            child = getattr(parent, child_name)
        except AttributeError as exc:
            raise ValueError(f"could not resolve module path {path!r}; missing {child_name!r}") from exc
    if not isinstance(child, nn.Module):
        raise ValueError(f"module_path {path!r} does not name an nn.Module")
    return parent, child_name, child


def _set_module_path(parent: nn.Module, child_name: str, replacement: nn.Module) -> None:
    if isinstance(parent, (nn.ModuleList, nn.Sequential)) and child_name.isdigit():
        parent[int(child_name)] = replacement
    else:
        setattr(parent, child_name, replacement)


def _policy_value(policy: OffloadPolicy, index: int, name: str, module: nn.Module) -> bool:
    if isinstance(policy, bool):
        return bool(policy)
    if callable(policy):
        return bool(policy(index, name, module))
    if index >= len(policy):
        raise ValueError(f"offload_policy has length {len(policy)}, but stage index {index} was requested")
    return bool(policy[index])


def _tensor_device_for_call(flat_tensors: Sequence[Tensor], local_device: torch.device, display_name: str) -> torch.device:
    if not flat_tensors:
        return local_device
    first_device = flat_tensors[0].device
    for tensor in flat_tensors:
        if tensor.device != first_device:
            raise ValueError(
                f"stage {display_name!r} received tensors on multiple devices; "
                "a torchrun process should forward one local batch on one local device"
            )
    if first_device != local_device:
        raise ValueError(
            f"stage {display_name!r} received a local batch tensor on {first_device}, "
            f"but this torchrun process is assigned to {local_device}. "
            "Move all input tensors to the process-local device in the training loop."
        )
    return first_device


# ---------------------------------------------------------------------------
# Offloaded layer state, prefetching, autograd replay, and optimizer dispatch.
# ---------------------------------------------------------------------------


@dataclass
class _PrefetchedState:
    """A staged parameter/buffer dictionary, optionally produced on a CUDA stream."""

    device: torch.device
    requires_grad: bool
    state: "OrderedDict[str, Tensor]"
    stream: Optional[torch.cuda.Stream] = None
    event: Optional[torch.cuda.Event] = None

    def wait(self) -> "OrderedDict[str, Tensor]":
        if self.device.type == "cuda" and self.event is not None:
            current = torch.cuda.current_stream(self.device)
            current.wait_event(self.event)
            for tensor in self.state.values():
                if tensor.device.type == "cuda":
                    tensor.record_stream(current)
        return self.state


@dataclass
class _PrefetchedOptimizerStep:
    named_parameters: List[Tuple[str, nn.Parameter]]
    staged_parameters: List[nn.Parameter]
    optimizer: Optional[torch.optim.Optimizer]
    event: Optional[torch.cuda.Event]


class _OffloadedModuleHandle:
    """Hidden CPU master state for one streamed layer.

    The handle is deliberately not an ``nn.Module``.  A stage wrapper stores it in
    a plain Python attribute, so the CPU master parameters are invisible to
    ``model.parameters()`` and to DDP.  The engine explicitly synchronizes their
    gradients and optimizer updates.
    """

    def __init__(
        self,
        *,
        qualified_name: str,
        module: nn.Module,
        stage_index: int,
        owner_rank: int,
        local_device: torch.device,
        process_group: Optional[Any],
        metrics: StreamingTransferMetrics,
        grad_norm_type: float,
        track_grad_norms: bool,
        pinned_transfer_buffer: _PinnedTransferBuffer,
        shared_cpu_store: _SharedCpuStore,
    ) -> None:
        self.qualified_name = qualified_name
        self.stage_index = int(stage_index)
        self.owner_rank = int(owner_rank)
        self.local_device = local_device
        self.process_group = process_group
        self.metrics = metrics
        self.grad_norm_type = float(grad_norm_type)
        self.track_grad_norms = bool(track_grad_norms)
        self.pinned_transfer_buffer = pinned_transfer_buffer
        self.shared_cpu_store = shared_cpu_store
        # The transformation replaces the original container in-place, so the
        # hidden handle can take ownership of the original child module.  In
        # distributed shared-CPU mode, do not call module.cpu() here: ranks that
        # do not initialize the shared tensors would briefly materialize a full
        # private CPU copy only to replace it with file-backed storage below.
        self.module = module
        self._share_module_cpu_state_()
        self.param_names = [name for name, _ in self.module.named_parameters(recurse=True)]
        self.buffer_names = [name for name, _ in self.module.named_buffers(recurse=True)]
        self.optimizer_state: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._prefetched: Optional[_PrefetchedState] = None
        self._prefetch_streams: Dict[Tuple[int, bool], torch.cuda.Stream] = {}
        self._optimizer_prefetch_streams: Dict[int, torch.cuda.Stream] = {}
        self._prefetched_optimizer_step: Optional[_PrefetchedOptimizerStep] = None
        self._grad_norms_by_name: Dict[str, Tensor] = {}

    def _shared_key(self, *parts: str) -> str:
        return ".".join((self.qualified_name, *parts))

    def _local_gradient_key(self, name: str) -> str:
        return self._shared_key("local_gradient", str(_rank(self.process_group)), name)

    def _copy_to_shared_gradient(self, name: str, source: Tensor, *, kind: str) -> Tensor:
        shared = self.shared_cpu_store.tensor_like(self._shared_key("gradient", name), source.detach())
        byte_count = _tensor_nbytes(source)

        def copy_fn() -> Tensor:
            shared.copy_(source.detach(), non_blocking=False)
            return shared

        if self.metrics is None:
            return copy_fn()
        return self.metrics.record_copy(kind, byte_count, source.device, copy_fn)

    def _copy_to_local_shared_gradient(self, name: str, source: Tensor, *, kind: str) -> Tensor:
        shared = self.shared_cpu_store.tensor_like(self._local_gradient_key(name), source.detach())
        byte_count = _tensor_nbytes(source)

        def copy_fn() -> Tensor:
            shared.copy_(source.detach(), non_blocking=False)
            return shared

        if self.metrics is None:
            return copy_fn()
        return self.metrics.record_copy(kind, byte_count, source.device, copy_fn)

    def _release_shared_gradients(self, names: Iterable[str]) -> None:
        for name in names:
            self.shared_cpu_store.release_tensor(self._shared_key("gradient", name))

    def _release_local_shared_gradients(self, names: Iterable[str]) -> None:
        for name in names:
            self.shared_cpu_store.release_tensor(self._local_gradient_key(name))

    def _share_module_cpu_state_(self) -> None:
        initialize = _rank(self.process_group) == 0
        for name, parameter in self.module.named_parameters(recurse=True):
            shared = self.shared_cpu_store.copy_tensor(
                self._shared_key("parameter", name),
                parameter.detach(),
                initialize=initialize,
            )
            parameter.data = shared
        for name, buffer in self.module.named_buffers(recurse=True):
            shared = self.shared_cpu_store.copy_tensor(
                self._shared_key("buffer", name),
                buffer.detach(),
                initialize=initialize,
            )
            buffer.data = shared

    @property
    def parameters_by_name(self) -> "OrderedDict[str, nn.Parameter]":
        return OrderedDict(self.module.named_parameters(recurse=True))

    @property
    def buffers_by_name(self) -> "OrderedDict[str, Tensor]":
        return OrderedDict(self.module.named_buffers(recurse=True))

    def train(self, mode: bool = True) -> None:
        self.module.train(mode)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.wait_optimizer_commit()
        self._grad_norms_by_name = {}
        if set_to_none:
            self._release_shared_gradients(self.parameters_by_name.keys())
            self._release_local_shared_gradients(self.parameters_by_name.keys())
        for parameter in self.parameters_by_name.values():
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter, memory_format=torch.preserve_format)
            else:
                parameter.grad.zero_()

    def _record_grad_norm(self, name: str, grad: Tensor, *, norm_type: float) -> None:
        if not self.track_grad_norms:
            return
        detached = grad.detach()
        if norm_type == float("inf"):
            self._grad_norms_by_name[name] = detached.abs().max()
        else:
            self._grad_norms_by_name[name] = torch.linalg.vector_norm(detached, ord=norm_type)

    def _state_for_call(self, device: torch.device, requires_grad: bool) -> "OrderedDict[str, Tensor]":
        state: "OrderedDict[str, Tensor]" = OrderedDict()
        for name, parameter in self.parameters_by_name.items():
            streamed = _copy_tensor_to_device(
                parameter.detach(),
                device,
                metrics=self.metrics,
                kind="state_h2d",
                pinned_transfer_buffer=self.pinned_transfer_buffer,
            )
            if requires_grad and parameter.requires_grad:
                streamed.requires_grad_(True)
            state[name] = streamed
        for name, buffer in self.buffers_by_name.items():
            state[name] = _copy_tensor_to_device(
                buffer.detach(),
                device,
                metrics=self.metrics,
                kind="state_h2d",
                pinned_transfer_buffer=self.pinned_transfer_buffer,
            )
        return state

    def _prefetch_stream(self, device: torch.device, requires_grad: bool) -> torch.cuda.Stream:
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        key = (int(index), bool(requires_grad))
        stream = self._prefetch_streams.get(key)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._prefetch_streams[key] = stream
        return stream

    def _optimizer_prefetch_stream(self, device: torch.device) -> torch.cuda.Stream:
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        stream = self._optimizer_prefetch_streams.get(int(index))
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._optimizer_prefetch_streams[int(index)] = stream
        return stream

    def clear_prefetch(self) -> None:
        with self._lock:
            self._prefetched = None

    def prefetch(self, device: torch.device, *, requires_grad: bool) -> None:
        """Stage this layer's parameters and buffers for a later call.

        On CUDA, copies are issued on a side stream and the returned state is
        protected by an event.  On CPU, the operation is necessarily synchronous,
        but keeping the same interface makes the correctness tests device-agnostic.
        """

        if device.type != self.local_device.type or (device.type == "cuda" and device.index != self.local_device.index):
            raise ValueError(f"cannot prefetch {self.qualified_name!r} to {device}; local device is {self.local_device}")

        with self._lock:
            existing = self._prefetched
            if existing is not None and existing.device == device and existing.requires_grad == requires_grad:
                return

        if device.type == "cuda":
            stream = self._prefetch_stream(device, requires_grad)
            with torch.cuda.stream(stream):
                state = self._state_for_call(device, requires_grad)
                event = torch.cuda.Event()
                event.record(stream)
            prefetched = _PrefetchedState(device=device, requires_grad=requires_grad, state=state, stream=stream, event=event)
        else:
            prefetched = _PrefetchedState(device=device, requires_grad=requires_grad, state=self._state_for_call(device, requires_grad))

        with self._lock:
            self._prefetched = prefetched

    def _consume_state(self, device: torch.device, *, requires_grad: bool) -> "OrderedDict[str, Tensor]":
        with self._lock:
            prefetched = self._prefetched
            if prefetched is not None and prefetched.device == device and prefetched.requires_grad == requires_grad:
                self._prefetched = None
            else:
                prefetched = None

        if prefetched is not None:
            return prefetched.wait()
        return self._state_for_call(device, requires_grad)

    def prefetch_optimizer_step(
        self,
        *,
        optimizer_cls: Type[torch.optim.Optimizer],
        optimizer_kwargs: Mapping[str, Any],
        clip_coef: Optional[Tensor],
    ) -> None:
        """Stage this layer's optimizer inputs to the local device on a side stream."""

        self.wait_optimizer_commit()
        self.clear_prefetch()
        if self.local_device.type != "cuda":
            return

        named_parameters = list(self.parameters_by_name.items())
        if not named_parameters:
            return

        rank = _rank(self.process_group)
        if rank != self.owner_rank:
            return

        stream = self._optimizer_prefetch_stream(self.local_device)
        staged_parameters: List[nn.Parameter] = []
        with torch.cuda.stream(stream):
            for name, cpu_parameter in named_parameters:
                staged = nn.Parameter(
                    _copy_tensor_to_device(
                        cpu_parameter.detach(),
                        self.local_device,
                        metrics=self.metrics,
                        kind="optimizer_param_h2d",
                        pinned_transfer_buffer=self.pinned_transfer_buffer,
                    ),
                    requires_grad=cpu_parameter.requires_grad,
                )
                if cpu_parameter.grad is not None:
                    staged.grad = _copy_tensor_to_device(
                        cpu_parameter.grad.detach(),
                        self.local_device,
                        metrics=self.metrics,
                        kind="optimizer_grad_h2d",
                        pinned_transfer_buffer=self.pinned_transfer_buffer,
                    )
                    if clip_coef is not None:
                        staged.grad.mul_(clip_coef.to(staged.grad.device))
                staged_parameters.append(staged)

            optimizer = optimizer_cls(staged_parameters, **dict(optimizer_kwargs))
            for (name, _), staged in zip(named_parameters, staged_parameters):
                saved_state = self.optimizer_state.get(name)
                if saved_state is not None:
                    optimizer.state[staged] = _Tree.to_device(
                        saved_state,
                        self.local_device,
                        metrics=self.metrics,
                        pinned_transfer_buffer=self.pinned_transfer_buffer,
                    )
            event = torch.cuda.Event()
            event.record(stream)

        with self._lock:
            self._prefetched_optimizer_step = _PrefetchedOptimizerStep(
                named_parameters=named_parameters,
                staged_parameters=staged_parameters,
                optimizer=optimizer,
                event=event,
            )

    def _consume_optimizer_step(
        self,
        *,
        optimizer_cls: Type[torch.optim.Optimizer],
        optimizer_kwargs: Mapping[str, Any],
        clip_coef: Optional[Tensor],
    ) -> _PrefetchedOptimizerStep:
        with self._lock:
            prefetched = self._prefetched_optimizer_step
            self._prefetched_optimizer_step = None

        if prefetched is None:
            self.prefetch_optimizer_step(
                optimizer_cls=optimizer_cls,
                optimizer_kwargs=optimizer_kwargs,
                clip_coef=clip_coef,
            )
            with self._lock:
                prefetched = self._prefetched_optimizer_step
                self._prefetched_optimizer_step = None
            if prefetched is None:
                raise RuntimeError("failed to prefetch optimizer step")

        if prefetched.event is not None:
            current = torch.cuda.current_stream(self.local_device)
            current.wait_event(prefetched.event)
            for staged in prefetched.staged_parameters:
                staged.data.record_stream(current)
                if staged.grad is not None:
                    staged.grad.record_stream(current)
            if prefetched.optimizer is not None:
                for state in prefetched.optimizer.state.values():
                    _Tree.record_stream(state, current)
        return prefetched

    def _schedule_optimizer_commit(
        self,
        *,
        named_parameters: List[Tuple[str, nn.Parameter]],
        staged_parameters: List[nn.Parameter],
        optimizer: Optional[torch.optim.Optimizer],
    ) -> None:
        for (name, cpu_parameter), staged in zip(named_parameters, staged_parameters):
            byte_count = _tensor_nbytes(staged)

            def copy_parameter() -> Tensor:
                cpu_parameter.data.copy_(staged.detach(), non_blocking=False)
                return cpu_parameter.data

            self.metrics.record_copy("optimizer_param_d2h", byte_count, staged.device, copy_parameter)
            if optimizer is not None and staged in optimizer.state:
                self.optimizer_state[name] = _Tree.detach_cpu(
                        optimizer.state[staged],
                        metrics=self.metrics,
                        shared_cpu_store=self.shared_cpu_store,
                        shared_key=self._shared_key("optimizer", name),
                )

    def wait_optimizer_commit(self) -> None:
        return

    def forward_on_device(self, args: Tuple[Any, ...], kwargs: Mapping[str, Any], device: torch.device) -> Any:
        state = self._consume_state(device=device, requires_grad=False)
        return functional_call(self.module, state, args=args, kwargs=dict(kwargs), strict=False)

    def replay_for_backward(
        self,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
        device: torch.device,
    ) -> Tuple[Any, "OrderedDict[str, Tensor]"]:
        state = self._consume_state(device=device, requires_grad=True)
        output = functional_call(self.module, state, args=args, kwargs=dict(kwargs), strict=False)
        params = OrderedDict((name, state[name]) for name in self.param_names)
        return output, params

    def accumulate_local_gradients(self, grads_by_name: Mapping[str, Optional[Tensor]]) -> None:
        """Accumulate local, not-yet-averaged gradients for offloaded params."""

        with self._lock:
            parameters = self.parameters_by_name
            for name, grad in grads_by_name.items():
                if grad is None:
                    continue
                target = parameters[name]
                if grad.device.type == "cuda" and self.local_device.type == "cuda":
                    grad_detached = grad.detach()
                    if target.grad is None:
                        accumulated = grad_detached
                    else:
                        accumulated = _copy_tensor_to_device(
                            target.grad.detach(),
                            grad_detached.device,
                            metrics=self.metrics,
                            kind="grad_accum_h2d",
                            pinned_transfer_buffer=self.pinned_transfer_buffer,
                        )
                        accumulated.add_(grad_detached)
                    self._record_grad_norm(name, accumulated, norm_type=self.grad_norm_type)
                    target.grad = self._copy_to_local_shared_gradient(name, accumulated, kind="grad_d2h")
                elif target.grad is None:
                    target.grad = grad.detach().clone()
                else:
                    target.grad.add_(grad.detach())

    def synchronize_gradients(self) -> None:
        """Average accumulated offloaded gradients across torchrun ranks."""

        if self.local_device.type == "cuda":
            for name, parameter in self.parameters_by_name.items():
                if parameter.grad is None:
                    continue
                staged_grad = _copy_tensor_to_device(
                    parameter.grad.detach(),
                    self.local_device,
                    metrics=self.metrics,
                    kind="grad_reduce_h2d",
                    pinned_transfer_buffer=self.pinned_transfer_buffer,
                )
                reduced = _all_reduce_mean_(staged_grad, self.local_device, self.process_group).detach()
                self._record_grad_norm(name, reduced, norm_type=self.grad_norm_type)
                if _rank(self.process_group) == self.owner_rank:
                    parameter.grad = self._copy_to_shared_gradient(name, reduced, kind="grad_reduce_d2h")
                else:
                    parameter.grad = None
                self._release_local_shared_gradients((name,))
            return

        for name, parameter in self.parameters_by_name.items():
            if parameter.grad is None:
                continue
            reduced = _all_reduce_mean_(parameter.grad, self.local_device, self.process_group).detach().cpu()
            self._record_grad_norm(name, reduced, norm_type=self.grad_norm_type)
            if _rank(self.process_group) == self.owner_rank:
                parameter.grad = self._copy_to_shared_gradient(name, reduced, kind="grad_reduce_d2h")
            else:
                parameter.grad = None

    def offloaded_grad_norm(self, *, device: torch.device, norm_type: float) -> Optional[Tensor]:
        if norm_type == self.grad_norm_type:
            cached_norms: List[Tensor] = []
            for name, parameter in self.parameters_by_name.items():
                if parameter.grad is None and name not in self._grad_norms_by_name:
                    continue
                cached_norm = self._grad_norms_by_name.get(name)
                if cached_norm is None:
                    break
                cached_norms.append(cached_norm)
            else:
                return _combine_grad_norms(cached_norms, norm_type=norm_type, device=device)
        if device.type != "cuda":
            return _grad_total_norm((parameter.grad for parameter in self.parameters_by_name.values()), norm_type=norm_type, device=device)
        if norm_type == self.grad_norm_type:
            cached_norms: List[Tensor] = []
            for name, parameter in self.parameters_by_name.items():
                if parameter.grad is None:
                    continue
                cached_norm = self._grad_norms_by_name.get(name)
                if cached_norm is None:
                    break
                cached_norms.append(cached_norm)
            else:
                return _combine_grad_norms(cached_norms, norm_type=norm_type, device=device)
        staged_grads: List[Tensor] = []
        for name, parameter in self.parameters_by_name.items():
            if parameter.grad is None:
                continue
            staged_grads.append(
                _copy_tensor_to_device(
                    parameter.grad.detach(),
                    device,
                    metrics=self.metrics,
                    kind="grad_norm_h2d",
                    pinned_transfer_buffer=self.pinned_transfer_buffer,
                )
            )
        return _grad_total_norm(staged_grads, norm_type=norm_type, device=device)

    def scale_gradients_(self, clip_coef: Tensor) -> None:
        if _rank(self.process_group) != self.owner_rank:
            return
        coef = clip_coef.to(torch.device("cpu"))
        for parameter in self.parameters_by_name.values():
            if parameter.grad is not None:
                parameter.grad.mul_(coef)

    def evict_device(self, device: torch.device) -> None:
        """Drop temporary staged state owned by this prototype."""

        self.clear_prefetch()

    def optimizer_step(
        self,
        *,
        optimizer_cls: Type[torch.optim.Optimizer],
        optimizer_kwargs: Mapping[str, Any],
        clip_coef: Optional[Tensor],
    ) -> None:
        """Run a PyTorch optimizer for this stage on its owner rank.

        Optimizer state is sharded by stage owner: rank ``stage_index %
        world_size`` owns the state and performs the update.  The updated CPU
        master weights are then broadcast to the other ranks.  This keeps the
        optimizer algorithm delegated to PyTorch while avoiding replicated AdamW
        state for offloaded layers.
        """

        self.clear_prefetch()
        named_parameters = list(self.parameters_by_name.items())
        if not named_parameters:
            return

        if self.local_device.type != "cuda":
            rank = _rank(self.process_group)
            owner = self.owner_rank
            if rank == owner:
                staged_parameters: List[nn.Parameter] = []
                for name, cpu_parameter in named_parameters:
                    staged = nn.Parameter(
                        _copy_tensor_to_device(
                            cpu_parameter.detach(),
                            self.local_device,
                            metrics=self.metrics,
                            kind="optimizer_param_h2d",
                            pinned_transfer_buffer=self.pinned_transfer_buffer,
                        ),
                        requires_grad=cpu_parameter.requires_grad,
                    )
                    if cpu_parameter.grad is not None:
                        staged.grad = _copy_tensor_to_device(
                            cpu_parameter.grad.detach(),
                            self.local_device,
                            metrics=self.metrics,
                            kind="optimizer_grad_h2d",
                            pinned_transfer_buffer=self.pinned_transfer_buffer,
                        )
                    staged_parameters.append(staged)

                optimizer = optimizer_cls(staged_parameters, **dict(optimizer_kwargs))
                for (name, _), staged in zip(named_parameters, staged_parameters):
                    saved_state = self.optimizer_state.get(name)
                    if saved_state is not None:
                        optimizer.state[staged] = _Tree.to_device(
                            saved_state,
                            self.local_device,
                            metrics=self.metrics,
                            pinned_transfer_buffer=self.pinned_transfer_buffer,
                        )
                optimizer.step()
                for (name, cpu_parameter), staged in zip(named_parameters, staged_parameters):
                    cpu_parameter.data.copy_(staged.detach(), non_blocking=False)
                    if staged in optimizer.state:
                        self.optimizer_state[name] = _Tree.detach_cpu(
                            optimizer.state[staged],
                            metrics=self.metrics,
                            shared_cpu_store=self.shared_cpu_store,
                            shared_key=self._shared_key("optimizer", name),
                        )

            for _, cpu_parameter in named_parameters:
                cpu_parameter.grad = None
            dist.barrier(group=self.process_group)
            self._release_shared_gradients(name for name, _ in named_parameters)
            return

        rank = _rank(self.process_group)
        owner = self.owner_rank
        staged_parameters: List[nn.Parameter]
        optimizer: Optional[torch.optim.Optimizer]
        if rank == owner:
            prefetched = self._consume_optimizer_step(
                optimizer_cls=optimizer_cls,
                optimizer_kwargs=optimizer_kwargs,
                clip_coef=clip_coef,
            )
            named_parameters = prefetched.named_parameters
            staged_parameters = prefetched.staged_parameters
            optimizer = prefetched.optimizer
            if optimizer is None:
                raise RuntimeError("owner rank is missing prefetched optimizer")

            optimizer.step()
        else:
            staged_parameters = []
            optimizer = None

        if rank == owner:
            self._schedule_optimizer_commit(
                named_parameters=named_parameters,
                staged_parameters=staged_parameters,
                optimizer=optimizer,
            )
        for name, cpu_parameter in named_parameters:
            cpu_parameter.grad = None
        dist.barrier(group=self.process_group)
        self._release_shared_gradients(name for name, _ in named_parameters)
        del optimizer, staged_parameters
        self.evict_device(self.local_device)

    def materialize(self, device: torch.device) -> nn.Module:
        self.wait_optimizer_commit()
        return copy.deepcopy(self.module).to(device)


class _OffloadedStageFunction(torch.autograd.Function):
    """Autograd boundary that saves activations but not device parameter copies."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        *flat_inputs_and_static: Any,
    ) -> Tuple[Tensor, ...]:
        call_spec: _CallSpec = flat_inputs_and_static[-4]
        output_holder: _OutputHolder = flat_inputs_and_static[-3]
        handle: _OffloadedModuleHandle = flat_inputs_and_static[-2]
        previous_backward_handle: Optional[_OffloadedModuleHandle] = flat_inputs_and_static[-1]
        flat_inputs = tuple(flat_inputs_and_static[: -5])
        autograd_token = flat_inputs_and_static[-5]
        del autograd_token

        if not all(torch.is_tensor(t) for t in flat_inputs):
            raise TypeError("internal error: offloaded stage received a non-tensor flattened input")

        device = _tensor_device_for_call(flat_inputs, handle.local_device, handle.qualified_name)
        args, kwargs = call_spec.rebuild(flat_inputs)
        ctx.handle = handle
        ctx.previous_backward_handle = previous_backward_handle
        ctx.call_spec = call_spec
        ctx.rng_snapshot = _RNGSnapshot(device)
        ctx.device = device
        ctx.save_for_backward(*flat_inputs)

        with torch.no_grad():
            output = handle.forward_on_device(args, kwargs, device)

        flat_outputs = output_holder.flatten_output(output)
        handle.evict_device(device)
        return flat_outputs

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Optional[Tensor]) -> Tuple[Any, ...]:  # type: ignore[override]
        saved_inputs = ctx.saved_tensors
        handle: _OffloadedModuleHandle = ctx.handle
        previous_backward_handle: Optional[_OffloadedModuleHandle] = ctx.previous_backward_handle
        call_spec: _CallSpec = ctx.call_spec
        device: torch.device = ctx.device

        if previous_backward_handle is not None:
            previous_backward_handle.prefetch(device, requires_grad=True)

        replay_inputs: List[Tensor] = []
        for saved, requires_grad in zip(saved_inputs, call_spec.tensor_requires_grad):
            replay = saved.detach()
            if requires_grad and _is_differentiable_tensor(replay):
                replay.requires_grad_(True)
            replay_inputs.append(replay)
        args, kwargs = call_spec.rebuild(replay_inputs)

        with ctx.rng_snapshot.fork(), torch.enable_grad():
            output, params = handle.replay_for_backward(args, kwargs, device)
            flat_outputs, _ = _TensorTreeSpec.flatten(output, tensor_predicate=_is_differentiable_tensor)
            if len(flat_outputs) != len(grad_outputs):
                raise RuntimeError(
                    f"replayed stage {handle.qualified_name!r} returned {len(flat_outputs)} differentiable tensors, "
                    f"but the original forward returned {len(grad_outputs)}"
                )

            targets: List[Tensor] = []
            target_kinds: List[Tuple[str, Optional[int], Optional[str]]] = []
            for index, (tensor, requires_grad) in enumerate(zip(replay_inputs, call_spec.tensor_requires_grad)):
                if requires_grad and _is_differentiable_tensor(tensor):
                    targets.append(tensor)
                    target_kinds.append(("input", index, None))
            for name, parameter in params.items():
                if parameter.requires_grad:
                    targets.append(parameter)
                    target_kinds.append(("param", None, name))

            if targets:
                normalized_grad_outputs: List[Tensor] = []
                for output_tensor, grad in zip(flat_outputs, grad_outputs):
                    if grad is None:
                        normalized_grad_outputs.append(torch.zeros_like(output_tensor))
                    else:
                        normalized_grad_outputs.append(grad)
                grads = torch.autograd.grad(
                    outputs=tuple(flat_outputs),
                    inputs=targets,
                    grad_outputs=tuple(normalized_grad_outputs),
                    allow_unused=True,
                    retain_graph=False,
                    create_graph=False,
                )
            else:
                grads = tuple()

        grad_inputs: List[Optional[Tensor]] = [None for _ in saved_inputs]
        grads_by_name: Dict[str, Optional[Tensor]] = {name: None for name in handle.param_names}
        for (kind, index, name), grad in zip(target_kinds, grads):
            if kind == "input":
                assert index is not None
                grad_inputs[index] = grad
            else:
                assert name is not None
                grads_by_name[name] = grad

        handle.accumulate_local_gradients(grads_by_name)
        handle.evict_device(device)

        # One gradient slot is needed for every input to ``apply``: flattened
        # tensor inputs, the dummy autograd token, the call spec, output holder,
        # current handle, and previous handle.
        return tuple(grad_inputs) + (None, None, None, None, None)


class _StreamingStage(nn.Module):
    """One item inside a transformed ModuleList or Sequential."""

    def __init__(
        self,
        *,
        display_name: str,
        module: nn.Module,
        offloaded_handle: Optional[_OffloadedModuleHandle],
        local_device: torch.device,
    ) -> None:
        super().__init__()
        self.display_name = display_name
        self.local_device = local_device
        self.offloaded = offloaded_handle is not None
        self._token_by_device: Dict[str, Tensor] = {}
        self._next_forward_handle: Optional[_OffloadedModuleHandle] = None
        self._previous_backward_handle: Optional[_OffloadedModuleHandle] = None
        if offloaded_handle is None:
            self.module = module
            self._handle: Optional[_OffloadedModuleHandle] = None
        else:
            self._handle = offloaded_handle

    def train(self, mode: bool = True) -> "_StreamingStage":
        super().train(mode)
        if self._handle is not None:
            self._handle.train(mode)
        return self

    def _token(self, device: torch.device) -> Tensor:
        key = str(device)
        token = self._token_by_device.get(key)
        if token is None or token.device != device:
            token = torch.ones((), device=device, requires_grad=True)
            self._token_by_device[key] = token
        return token

    def _prefetch_self(self, device: torch.device, *, requires_grad: bool = False) -> None:
        if self._handle is not None:
            self._handle.prefetch(device, requires_grad=requires_grad)

    def _schedule_next_forward_prefetch(self, device: torch.device) -> None:
        if self._next_forward_handle is not None:
            self._next_forward_handle.prefetch(device, requires_grad=False)

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        flat_inputs, call_spec = _CallSpec.from_call(args, kwargs)
        device = _tensor_device_for_call(flat_inputs, self.local_device, self.display_name)
        self._schedule_next_forward_prefetch(device)

        if self._handle is not None:
            output_holder = _OutputHolder()
            result = _OffloadedStageFunction.apply(
                *flat_inputs,
                self._token(device),
                call_spec,
                output_holder,
                self._handle,
                self._previous_backward_handle,
            )
            return output_holder.reconstruct(result)
        return self.module(*args, **kwargs)  # type: ignore[attr-defined]

    def materialize(self, device: torch.device) -> nn.Module:
        if self._handle is not None:
            return self._handle.materialize(device)
        return copy.deepcopy(self.module).to(device)  # type: ignore[attr-defined]

    def offloaded_handles(self) -> Iterator[_OffloadedModuleHandle]:
        if self._handle is not None:
            yield self._handle

    def clear_prefetch(self) -> None:
        if self._handle is not None:
            self._handle.clear_prefetch()


class CPUStreamingModuleList(nn.ModuleList):
    """A drop-in iterable replacement for a transformed ``nn.ModuleList``.

    Standard ``ModuleList`` has no forward method.  This replacement remains
    iterable for existing user-written loops, and it also provides a convenience
    forward that chains the stages when called directly.
    """

    def __init__(self, modules: Iterable[_StreamingStage]) -> None:
        super().__init__(list(modules))
        self._link_prefetch_neighbors()

    def _link_prefetch_neighbors(self) -> None:
        stages = [stage for stage in self if isinstance(stage, _StreamingStage)]
        next_handle: Optional[_OffloadedModuleHandle] = None
        for stage in reversed(stages):
            stage._next_forward_handle = next_handle
            if stage._handle is not None:
                next_handle = stage._handle
        previous_handle: Optional[_OffloadedModuleHandle] = None
        for stage in stages:
            stage._previous_backward_handle = previous_handle
            if stage._handle is not None:
                previous_handle = stage._handle

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        if len(self) == 0:
            raise RuntimeError("cannot call an empty CPUStreamingModuleList")
        flat_inputs, _ = _CallSpec.from_call(args, kwargs)
        device = _tensor_device_for_call(flat_inputs, self[0].local_device, "ModuleList")
        if isinstance(self[0], _StreamingStage):
            self[0]._prefetch_self(device, requires_grad=False)
        output = self[0](*args, **kwargs)
        for layer in list(self)[1:]:
            output = layer(output)
        return output

    def offloaded_handles(self) -> Iterator[_OffloadedModuleHandle]:
        for stage in self:
            if isinstance(stage, _StreamingStage):
                yield from stage.offloaded_handles()

    def clear_prefetch(self) -> None:
        for stage in self:
            if isinstance(stage, _StreamingStage):
                stage.clear_prefetch()

    def materialize(self, device: torch.device) -> nn.ModuleList:
        return nn.ModuleList([stage.materialize(device) for stage in self])


class CPUStreamingSequential(nn.Sequential):
    """A callable replacement for a transformed ``nn.Sequential``.

    Unlike vanilla ``nn.Sequential``, this forward accepts arbitrary arguments for
    the first stage.  Each later stage receives the previous stage's output as its
    single positional argument, matching the usual sequential-composition rule.
    """

    def __init__(self, modules: Optional[OrderedDict[str, _StreamingStage]] = None) -> None:
        super().__init__(modules if modules is not None else OrderedDict())
        self._link_prefetch_neighbors()

    def _link_prefetch_neighbors(self) -> None:
        stages = [stage for stage in self if isinstance(stage, _StreamingStage)]
        next_handle: Optional[_OffloadedModuleHandle] = None
        for stage in reversed(stages):
            stage._next_forward_handle = next_handle
            if stage._handle is not None:
                next_handle = stage._handle
        previous_handle: Optional[_OffloadedModuleHandle] = None
        for stage in stages:
            stage._previous_backward_handle = previous_handle
            if stage._handle is not None:
                previous_handle = stage._handle

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        if len(self) == 0:
            raise RuntimeError("cannot call an empty CPUStreamingSequential")
        first = next(iter(self))
        flat_inputs, _ = _CallSpec.from_call(args, kwargs)
        device = _tensor_device_for_call(flat_inputs, first.local_device, "Sequential")
        if isinstance(first, _StreamingStage):
            first._prefetch_self(device, requires_grad=False)
        output = first(*args, **kwargs)
        first_seen = False
        for layer in self:
            if not first_seen:
                first_seen = True
                continue
            output = layer(output)
        return output

    def offloaded_handles(self) -> Iterator[_OffloadedModuleHandle]:
        for stage in self:
            if isinstance(stage, _StreamingStage):
                yield from stage.offloaded_handles()

    def clear_prefetch(self) -> None:
        for stage in self:
            if isinstance(stage, _StreamingStage):
                stage.clear_prefetch()

    def materialize(self, device: torch.device) -> nn.Sequential:
        modules: "OrderedDict[str, nn.Module]" = OrderedDict()
        for name, stage in self._modules.items():
            assert isinstance(stage, _StreamingStage)
            modules[name] = stage.materialize(device)
        return nn.Sequential(modules)


# ---------------------------------------------------------------------------
# Engine and transformation entry point.
# ---------------------------------------------------------------------------


class CPUStreamingEngine:
    """Companion object returned by the in-place transformation.

    The engine is not the model.  It owns optimizer state for offloaded layers, a
    normal PyTorch optimizer for resident parameters, and, when distributed
    training is active, the DDP-wrapped model exposed as ``.model``.  The training
    loop calls ``engine.zero_grad()``, computes ``loss.backward()`` through
    ``engine.model``, and calls ``engine.step()``.
    """

    def __init__(
        self,
        *,
        root_model: nn.Module,
        module_path: str,
        streaming_container: Union[CPUStreamingModuleList, CPUStreamingSequential],
        local_device: torch.device,
        optimizer_cls: Type[torch.optim.Optimizer],
        optimizer_kwargs: Mapping[str, Any],
        max_grad_norm: Optional[float],
        grad_norm_type: float,
        error_if_nonfinite: bool,
        process_group: Optional[Any],
        ddp_kwargs: Mapping[str, Any],
        close_rank: int,
        metrics: StreamingTransferMetrics,
        shared_cpu_store: _SharedCpuStore,
    ) -> None:
        self.root_model = root_model
        self.module_path = module_path
        self.streaming_container = streaming_container
        self.local_device = local_device
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = dict(optimizer_kwargs)
        self.max_grad_norm = max_grad_norm
        self.grad_norm_type = float(grad_norm_type)
        self.error_if_nonfinite = bool(error_if_nonfinite)
        self.process_group = process_group
        self.close_rank = int(close_rank)
        self.transfer_metrics = metrics
        self.shared_cpu_store = shared_cpu_store
        self.optimizer_prefetch_enabled = _env_flag("DTAI_PARALLEL_OPTIMIZER_PREFETCH", default=True)
        self._closed = False

        self.handles = list(streaming_container.offloaded_handles())
        self.shared_cpu_store.barrier()

        # Only resident parameters are registered in the module tree.  Offloaded
        # CPU masters live in hidden handles, so moving the root model cannot move
        # them off CPU.
        self.root_model.to(self.local_device)

        kwargs = dict(ddp_kwargs)
        if self.local_device.type == "cuda":
            index = self.local_device.index
            if index is None:
                index = torch.cuda.current_device()
            kwargs.setdefault("device_ids", [index])
            kwargs.setdefault("output_device", index)
        else:
            kwargs.setdefault("device_ids", None)
        kwargs.setdefault("process_group", process_group)
        self.model: nn.Module = DDP(self.root_model, **kwargs)
        self.ddp_enabled = True

        resident_parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.resident_optimizer: Optional[torch.optim.Optimizer]
        if resident_parameters:
            self.resident_optimizer = self.optimizer_cls(resident_parameters, **self.optimizer_kwargs)
        else:
            self.resident_optimizer = None

        self.config = StreamingConfig(
            module_path=module_path,
            device=local_device,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=self.optimizer_kwargs,
            max_grad_norm=max_grad_norm,
            grad_norm_type=self.grad_norm_type,
            ddp_enabled=self.ddp_enabled,
            world_size=_world_size(process_group),
            rank=_rank(process_group),
            shared_cpu_dir=self.shared_cpu_store.directory,
        )

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self._closed:
            raise RuntimeError("close() has been called; create a new engine for more training")
        self.streaming_container.clear_prefetch()
        if self.resident_optimizer is not None:
            self.resident_optimizer.zero_grad(set_to_none=set_to_none)
        else:
            self.model.zero_grad(set_to_none=set_to_none)
        for handle in self.handles:
            handle.zero_grad(set_to_none=set_to_none)

    def _cuda_total_grad_norm(self) -> Tensor:
        norms: List[Optional[Tensor]] = []
        for handle in self.handles:
            norms.append(handle.offloaded_grad_norm(device=self.local_device, norm_type=self.grad_norm_type))
        if self.resident_optimizer is not None:
            resident_gradients: List[Tensor] = []
            for group in self.resident_optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        resident_gradients.append(parameter.grad)
            norms.append(_grad_total_norm(resident_gradients, norm_type=self.grad_norm_type, device=self.local_device))
        return _combine_grad_norms(norms, norm_type=self.grad_norm_type, device=self.local_device)

    def _scale_resident_gradients_(self, clip_coef: Tensor) -> None:
        if self.resident_optimizer is None:
            return
        for group in self.resident_optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.mul_(clip_coef.to(parameter.grad.device))

    def _scale_offloaded_gradients_(self, clip_coef: Tensor) -> None:
        for handle in self.handles:
            handle.scale_gradients_(clip_coef)

    def step(self) -> Optional[Tensor]:
        """Synchronize offloaded gradients, optionally clip, and run optimizers."""

        if self._closed:
            raise RuntimeError("close() has been called; create a new engine for more training")

        for handle in self.handles:
            handle.synchronize_gradients()

        total_norm: Optional[Tensor]
        cuda_clip_coef: Optional[Tensor] = None
        if self.max_grad_norm is not None:
            total_norm = self._cuda_total_grad_norm()
            cuda_clip_coef = _clip_coef(
                total_norm,
                max_norm=float(self.max_grad_norm),
                error_if_nonfinite=self.error_if_nonfinite,
            )
            self._scale_resident_gradients_(cuda_clip_coef)
            if self.local_device.type != "cuda":
                self._scale_offloaded_gradients_(cuda_clip_coef)
        else:
            total_norm = None

        if self.resident_optimizer is not None:
            self.resident_optimizer.step()
            self.resident_optimizer.zero_grad(set_to_none=True)

        if self.handles and self.local_device.type == "cuda" and self.optimizer_prefetch_enabled:
            self.handles[0].prefetch_optimizer_step(
                optimizer_cls=self.optimizer_cls,
                optimizer_kwargs=self.optimizer_kwargs,
                clip_coef=cuda_clip_coef,
            )

        for index, handle in enumerate(self.handles):
            next_index = index + 1
            if next_index < len(self.handles) and self.local_device.type == "cuda" and self.optimizer_prefetch_enabled:
                self.handles[next_index].prefetch_optimizer_step(
                    optimizer_cls=self.optimizer_cls,
                    optimizer_kwargs=self.optimizer_kwargs,
                    clip_coef=cuda_clip_coef,
                )
            handle.optimizer_step(
                optimizer_cls=self.optimizer_cls,
                optimizer_kwargs=self.optimizer_kwargs,
                clip_coef=cuda_clip_coef,
            )
            if index > 0:
                self.handles[index - 1].wait_optimizer_commit()

        if self.handles:
            self.handles[-1].wait_optimizer_commit()

        self.streaming_container.clear_prefetch()
        return total_norm

    def transfer_timing_summary(self, *, reset: bool = False, synchronize: bool = True) -> Dict[str, Dict[str, float]]:
        """Return accumulated transfer timing counters.

        ``enqueue_ms`` is host-side time spent issuing copies.  ``cuda_ms`` is
        measured with CUDA events and therefore reflects device-stream copy time
        for transfers that ran on CUDA streams.
        """

        return self.transfer_metrics.summary(reset=reset, synchronize=synchronize)

    def offloaded_parameters(self) -> Iterator[nn.Parameter]:
        for handle in self.handles:
            yield from handle.parameters_by_name.values()

    def resident_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.model.parameters()

    def materialize(self, device: Optional[DeviceLike] = None) -> nn.Module:
        """Return a normal copy of the transformed model with no streaming wrappers.

        The hidden offload handles contain locks and optimizer state, so an
        ordinary ``copy.deepcopy`` of the live transformed model is not the right
        operation.  Instead the engine temporarily swaps the transformed submodule
        for an ordinary materialized container, deep-copies the resulting normal
        model, and then puts the streaming container back.
        """

        target_device = torch.device(device) if device is not None else self.local_device
        parent, child_name, current_child = _resolve_module_path(self.root_model, self.module_path)
        if current_child is not self.streaming_container:
            raise RuntimeError("the transformed submodule was replaced outside the streaming engine")
        materialized_child = self.streaming_container.materialize(target_device)
        _set_module_path(parent, child_name, materialized_child)
        try:
            model_copy = copy.deepcopy(self.root_model)
        finally:
            _set_module_path(parent, child_name, self.streaming_container)
        return model_copy.to(target_device)

    def close(self, *, return_on_all_ranks: bool = False, device: Optional[DeviceLike] = None) -> Optional[nn.Module]:
        """Evict temporary device state and return a normal model on rank 0.

        By default only ``close_rank`` returns the materialized model; other ranks
        return ``None``.  This mirrors typical checkpointing under ``torchrun``.
        Pass ``return_on_all_ranks=True`` when every rank should get a copy.
        """

        if self._closed:
            return None
        dist.barrier(group=self.process_group)
        for handle in self.handles:
            handle.evict_device(self.local_device)
        self._closed = True

        rank = _rank(self.process_group)
        result: Optional[nn.Module] = None
        if return_on_all_ranks or rank == self.close_rank:
            target = torch.device(device) if device is not None else self.local_device
            result = self.materialize(target)

        dist.barrier(group=self.process_group)
        self.shared_cpu_store.cleanup()
        dist.barrier(group=self.process_group)
        return result


def _build_streaming_container(
    *,
    container: nn.Module,
    module_path: str,
    offload_policy: OffloadPolicy,
    resident_suffix_count: int,
    local_device: torch.device,
    process_group: Optional[Any],
    metrics: StreamingTransferMetrics,
    grad_norm_type: float,
    track_grad_norms: bool,
    pinned_transfer_buffer: _PinnedTransferBuffer,
    shared_cpu_store: _SharedCpuStore,
) -> Union[CPUStreamingModuleList, CPUStreamingSequential]:
    if not isinstance(container, (nn.ModuleList, nn.Sequential)):
        raise TypeError(
            f"module_path {module_path!r} must name an nn.ModuleList or nn.Sequential; "
            f"got {type(container).__name__}"
        )

    world = _world_size(process_group)
    module_count = len(container)
    if resident_suffix_count < 0:
        raise ValueError("resident_suffix_count must be non-negative")
    if resident_suffix_count > module_count:
        raise ValueError(
            f"resident_suffix_count is {resident_suffix_count}, but {module_path!r} contains only {module_count} stages"
        )
    first_resident_suffix_index = module_count - resident_suffix_count

    stages: "OrderedDict[str, _StreamingStage]" = OrderedDict()
    for index, (name, child) in enumerate(container._modules.items()):
        offload = _policy_value(offload_policy, index, name, child) and index < first_resident_suffix_index
        qualified_name = f"{module_path}.{name}"
        if offload:
            handle = _OffloadedModuleHandle(
                qualified_name=qualified_name,
                module=child,
                stage_index=index,
                owner_rank=index % world,
                local_device=local_device,
                process_group=process_group,
                metrics=metrics,
                grad_norm_type=grad_norm_type,
                track_grad_norms=track_grad_norms,
                pinned_transfer_buffer=pinned_transfer_buffer,
                shared_cpu_store=shared_cpu_store,
            )
            stage = _StreamingStage(
                display_name=qualified_name,
                module=nn.Identity(),
                offloaded_handle=handle,
                local_device=local_device,
            )
        else:
            stage = _StreamingStage(
                display_name=qualified_name,
                module=copy.deepcopy(child),
                offloaded_handle=None,
                local_device=local_device,
            )
        stages[name] = stage

    if isinstance(container, nn.Sequential):
        return CPUStreamingSequential(stages)
    return CPUStreamingModuleList(stages.values())


def apply_cpu_streaming_(
    model: nn.Module,
    module_path: str,
    *,
    offload_policy: OffloadPolicy = True,
    resident_suffix_count: int = 0,
    optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.AdamW,
    optimizer_kwargs: Optional[Mapping[str, Any]] = None,
    max_grad_norm: Optional[float] = None,
    grad_norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    device: Optional[DeviceLike] = None,
    process_group: Optional[Any] = None,
    ddp_kwargs: Optional[Mapping[str, Any]] = None,
    close_rank: int = 0,
    collect_timing: Optional[bool] = None,
    shared_cpu_dir: Optional[str] = None,
) -> CPUStreamingEngine:
    """Transform ``model.<module_path>`` in place and return a training engine.

    Parameters
    ----------
    model:
        The existing model object.  The function mutates it by replacing one
        ordered submodule.  The surrounding embedding, normalization, and head
        modules remain ordinary resident PyTorch modules.
    module_path:
        Dotted path to an ``nn.ModuleList`` or ``nn.Sequential`` inside ``model``;
        examples include ``"layers"`` and ``"decoder.layers"``.
    offload_policy:
        ``True`` offloads every item in the list.  ``False`` keeps every item
        resident.  A sequence of booleans controls individual stages.  A callable
        receives ``(index, name, module)`` and returns a boolean.
    resident_suffix_count:
        Number of trailing items in the target ``ModuleList`` or ``Sequential``
        to keep resident, regardless of ``offload_policy``.  For example,
        ``offload_policy=True, resident_suffix_count=2`` offloads all but the
        final two stages.
    optimizer_cls, optimizer_kwargs:
        A real PyTorch optimizer class and its keyword arguments.  One optimizer
        is constructed for resident parameters.  Offloaded stages construct
        short-lived optimizers during ``step()`` and restore their opaque state.
    max_grad_norm:
        If set, ``step()`` clips the combined resident and offloaded gradients
        after all DDP/offloaded reductions and before any optimizer update.
    device:
        The single local device for this torchrun process.  If omitted, CUDA uses
        ``LOCAL_RANK`` when available; CPU is used otherwise.
    ddp_kwargs:
        Optional keyword arguments for the ``DistributedDataParallel`` wrapper
        used for resident parameters.
    collect_timing:
        When true, collect per-transfer enqueue time, byte counts, and CUDA event
        durations.  If omitted, ``DTAI_PARALLEL_TIMING=1`` enables collection.
    shared_cpu_dir:
        Optional directory for offloaded CPU state.  Offloaded CPU weights,
        gradients, and optimizer-state tensors are always file-backed shared
        memory.  This directory must resolve under ``/dev/shm``.  If omitted,
        rank 0 creates a temporary directory under ``/dev/shm`` and broadcasts it
        to the other same-host ranks.

    Returns
    -------
    CPUStreamingEngine
        The companion object that owns optimizer state, exposes the model to use
        in the training loop, and materializes a normal model in ``close()``.
    """

    local_device = _normalize_device(device) if device is not None else _infer_local_device()
    _init_process_group_from_torchrun(local_device)
    if collect_timing is None:
        collect_timing = os.environ.get("DTAI_PARALLEL_TIMING", "").lower() in {"1", "true", "yes", "on"}
    metrics = StreamingTransferMetrics(enabled=bool(collect_timing))
    pinned_transfer_buffer = _PinnedTransferBuffer()
    shared_cpu_store = _SharedCpuStore.create(requested_dir=shared_cpu_dir, process_group=process_group)

    parent, child_name, target = _resolve_module_path(model, module_path)
    streaming_container = _build_streaming_container(
        container=target,
        module_path=module_path,
        offload_policy=offload_policy,
        resident_suffix_count=int(resident_suffix_count),
        local_device=local_device,
        process_group=process_group,
        metrics=metrics,
        grad_norm_type=grad_norm_type,
        track_grad_norms=max_grad_norm is not None,
        pinned_transfer_buffer=pinned_transfer_buffer,
        shared_cpu_store=shared_cpu_store,
    )
    _set_module_path(parent, child_name, streaming_container)

    return CPUStreamingEngine(
        root_model=model,
        module_path=module_path,
        streaming_container=streaming_container,
        local_device=local_device,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=dict(optimizer_kwargs or {}),
        max_grad_norm=max_grad_norm,
        grad_norm_type=grad_norm_type,
        error_if_nonfinite=error_if_nonfinite,
        process_group=process_group,
        ddp_kwargs=dict(ddp_kwargs or {}),
        close_rank=close_rank,
        metrics=metrics,
        shared_cpu_store=shared_cpu_store,
    )


# Backwards-compatible alias retained for the first prototype and for readers who
# want the target type spelled out.  The shorter name is the preferred API.
apply_cpu_streaming_to_modulelist_ = apply_cpu_streaming_
apply_cpu_streaming = apply_cpu_streaming_


__all__ = [
    "CPUStreamingEngine",
    "CPUStreamingModuleList",
    "CPUStreamingSequential",
    "StreamingConfig",
    "StreamingTransferMetrics",
    "apply_cpu_streaming",
    "apply_cpu_streaming_",
    "apply_cpu_streaming_to_modulelist_",
]
