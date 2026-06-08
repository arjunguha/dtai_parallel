"""CPU-master streaming data parallelism for layer-by-layer training.

This module is written as a compact research prototype rather than as a
production runtime.  The idea is deliberately close to the algorithm sketched in
DeltaAI/GH200 discussions:

1.  The authoritative copy of every *offloaded* parameter lives on CPU.
2.  During the forward pass, a layer is copied to each participating device,
    used for that device's local mini-batch, and then allowed to die.
3.  During the backward pass, the layer is copied back to the device, gradients
    are computed from the saved activations, local gradients are averaged across
    replicas, and the averaged gradient is written into the CPU master
    parameter's ``.grad`` field.
4.  During ``step()``, the CPU master parameters, their gradients, and their
    optimizer state are streamed to an optimizer device one layer at a time.
    A real ``torch.optim.Optimizer`` class performs the update; the updated
    weights and state are copied back to CPU.  AdamW is therefore not reimplemented
    here.

The implementation uses ``torch.func.functional_call`` because it lets a module
be evaluated with a temporary dictionary of parameters and buffers.  PyTorch
states that functional_call runs a module after replacing its parameters and
buffers with the supplied values.  That is exactly the primitive needed for
streaming: the layer object stays on CPU, while one call receives temporary GPU
copies.

The central technical point is the custom autograd boundary.  If we used a normal
``module(x)`` call, PyTorch autograd would keep references to the GPU parameter
copies produced during forward.  That would defeat the purpose of evicting
weights.  Instead, ``_OffloadedStageFunction`` saves only the input activation
and replays the stage during backward with fresh streamed-in parameters.  This is
similar in spirit to activation checkpointing, except that we retain activations
and recompute with re-fetched weights.

Scope of the prototype
----------------------

The public class, ``CPUStreamingDataParallel``, expects the model to be written
as an ordered list of stages, like ``nn.Sequential``.  A stage may be offloaded or
resident.  Resident stages are kept on each device and synchronized with the same
mean-gradient rule as DDP inside this single-process prototype.  In a production
one-process-per-GPU job, such resident modules are the natural place to use
``torch.nn.parallel.DistributedDataParallel`` directly; the tests include a real
DDP reference so the gradient semantics are checked against PyTorch DDP.

The code is intentionally conservative: no closures, no optimizer algorithms
whose updates depend on cross-parameter global statistics, and no mutation of
running buffers inside offloaded layers.  Adam, AdamW, and SGD are the intended
optimizers.  Stochastic layers such as Dropout are supported by saving and
restoring RNG state across the backward replay, but deterministic layers are much
easier to reason about and are what the equivalence tests use.
"""

from __future__ import annotations

import copy
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, Union

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_

try:  # PyTorch >= 2.0
    from torch.func import functional_call
except Exception:  # pragma: no cover - compatibility for older PyTorch builds
    from torch.nn.utils.stateless import functional_call  # type: ignore


DeviceLike = Union[str, torch.device]


@dataclass(frozen=True)
class StageSpec:
    """Describe one sequential stage.

    Parameters
    ----------
    name:
        Stable name used when reconstructing the model in ``close()``.  The name
        is also useful in assertion failures and in optimizer-state sharding.
    module:
        A PyTorch module whose ``forward`` accepts one tensor and returns one
        tensor.  This restriction keeps the prototype readable.  Extending the
        same idea to tensor pytrees mostly requires flattening inputs and outputs
        around the autograd function.
    offload:
        If true, parameters and buffers for this stage live on CPU and are
        streamed to devices on demand.  If false, one resident replica is kept on
        each device and synchronized with mean-gradient semantics.
    """

    name: str
    module: nn.Module
    offload: bool = True


class _Tree:
    """Small helpers for copying PyTorch optimizer state between devices.

    PyTorch optimizers store state in nested dictionaries that usually contain
    tensors, Python scalars, and occasionally lists or tuples.  We do not inspect
    AdamW internals.  We just move every tensor and deepcopy everything else.
    """

    @staticmethod
    def to_device(value: Any, device: torch.device) -> Any:
        if torch.is_tensor(value):
            return value.detach().to(device, non_blocking=True).clone()
        if isinstance(value, dict):
            return {k: _Tree.to_device(v, device) for k, v in value.items()}
        if isinstance(value, list):
            return [_Tree.to_device(v, device) for v in value]
        if isinstance(value, tuple):
            return tuple(_Tree.to_device(v, device) for v in value)
        return copy.deepcopy(value)

    @staticmethod
    def detach_cpu(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: _Tree.detach_cpu(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_Tree.detach_cpu(v) for v in value]
        if isinstance(value, tuple):
            return tuple(_Tree.detach_cpu(v) for v in value)
        return copy.deepcopy(value)


class _RNGSnapshot:
    """RNG state used to replay stochastic layers during custom backward."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.cpu_state = torch.random.get_rng_state()
        self.cuda_state: Optional[Tensor]
        if device.type == "cuda":
            self.cuda_state = torch.cuda.get_rng_state(device)
        else:
            self.cuda_state = None

    def fork(self) -> Iterator[None]:
        """Temporarily restore the saved state and then put the caller's state back."""

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


class _OffloadedStageHandle:
    """CPU-resident master copy plus optimizer state for one offloaded stage."""

    def __init__(
        self,
        name: str,
        module: nn.Module,
        replica_count: int,
        reduce_device: torch.device,
    ) -> None:
        self.name = name
        self.module = copy.deepcopy(module).cpu()
        self.replica_count = int(replica_count)
        self.reduce_device = reduce_device
        self.lock = threading.Lock()

        self.param_names = [name for name, _ in self.module.named_parameters(recurse=True)]
        self.buffer_names = [name for name, _ in self.module.named_buffers(recurse=True)]
        self.optimizer_state: Dict[str, Dict[str, Any]] = {}

    @property
    def parameters_by_name(self) -> "OrderedDict[str, nn.Parameter]":
        return OrderedDict(self.module.named_parameters(recurse=True))

    @property
    def buffers_by_name(self) -> "OrderedDict[str, Tensor]":
        return OrderedDict(self.module.named_buffers(recurse=True))

    def _snapshot_master_tensors(self) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        params = {name: parameter.detach().cpu().clone() for name, parameter in self.parameters_by_name.items()}
        buffers = {name: buffer.detach().cpu().clone() for name, buffer in self.buffers_by_name.items()}
        return params, buffers

    def _restore_master_tensors(self, snapshot: Tuple[Dict[str, Tensor], Dict[str, Tensor]]) -> None:
        param_snapshot, buffer_snapshot = snapshot
        for name, parameter in self.parameters_by_name.items():
            parameter.data = param_snapshot[name]
        for name, buffer in self.buffers_by_name.items():
            buffer.data = buffer_snapshot[name]

    def train(self, mode: bool = True) -> None:
        self.module.train(mode)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters_by_name.values():
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter, memory_format=torch.preserve_format)
            else:
                parameter.grad.zero_()

    def _state_for_call(self, device: torch.device, requires_grad: bool) -> "OrderedDict[str, Tensor]":
        """Build a parameter/buffer dictionary for one functional_call."""

        state: "OrderedDict[str, Tensor]" = OrderedDict()
        for name, parameter in self.parameters_by_name.items():
            streamed = parameter.detach().to(device, non_blocking=True).clone()
            if requires_grad and parameter.requires_grad:
                streamed.requires_grad_(True)
            state[name] = streamed
        for name, buffer in self.buffers_by_name.items():
            state[name] = buffer.detach().to(device, non_blocking=True).clone()
        return state

    def forward_on_device(self, x: Tensor, device: torch.device) -> Tensor:
        with self.lock:
            snapshot = self._snapshot_master_tensors()
            state = self._state_for_call(device=device, requires_grad=False)
            try:
                return functional_call(self.module, state, (x,), strict=False)
            finally:
                self._restore_master_tensors(snapshot)

    def replay_for_backward(self, x: Tensor, device: torch.device) -> Tuple[Tensor, "OrderedDict[str, Tensor]"]:
        with self.lock:
            snapshot = self._snapshot_master_tensors()
            state = self._state_for_call(device=device, requires_grad=True)
            try:
                output = functional_call(self.module, state, (x,), strict=False)
                params = OrderedDict((name, state[name]) for name in self.param_names)
                return output, params
            finally:
                self._restore_master_tensors(snapshot)

    def accumulate_mean_gradients(self, grads_by_name: Mapping[str, Optional[Tensor]]) -> None:
        """Accumulate one replica's local gradients into CPU master ``.grad``.

        DDP semantics are mean-gradient semantics: every participating replica
        computes a local gradient, and the optimizer sees the average.  The test
        training loop therefore sums local losses and lets this method divide by
        ``replica_count``.
        """

        with self.lock:
            parameters = self.parameters_by_name
            for name, grad in grads_by_name.items():
                if grad is None:
                    continue
                target = parameters[name]
                reduced = grad.detach()
                if self.reduce_device.type == "cuda" and reduced.device != self.reduce_device:
                    # This mirrors the intended design point: local gradients can
                    # be staged to GPU 0 for the reduction before the result is
                    # written back to CPU.  In CPU-only tests this path is not used.
                    reduced = reduced.to(self.reduce_device, non_blocking=True)
                reduced_cpu = reduced.to("cpu", non_blocking=True) / float(self.replica_count)
                if target.grad is None:
                    target.grad = reduced_cpu.clone()
                else:
                    target.grad.add_(reduced_cpu)

    def evict_device(self, device: torch.device) -> None:
        """Drop persistent GPU references.

        The prototype does not keep a device cache, so there is nothing to clear
        besides giving CUDA's caching allocator a chance after temporary tensors
        go out of scope.  The method exists because a production implementation
        would release stream-owned buffers here.
        """

        if device.type == "cuda":
            torch.cuda.empty_cache()

    def optimizer_step(
        self,
        optimizer_cls: Type[torch.optim.Optimizer],
        optimizer_kwargs: Mapping[str, Any],
        optimizer_device: torch.device,
    ) -> None:
        """Stream this stage to an optimizer device and dispatch to PyTorch.

        This method constructs temporary device parameters, attaches the CPU
        gradients and previously saved optimizer state, calls ``optimizer.step()``,
        then copies the new parameters and optimizer state back to CPU.  The
        method intentionally does not know AdamW equations.
        """

        named_parameters = list(self.parameters_by_name.items())
        if not named_parameters:
            return

        staged_parameters: List[nn.Parameter] = []
        for _, cpu_parameter in named_parameters:
            staged = nn.Parameter(
                cpu_parameter.detach().to(optimizer_device, non_blocking=True).clone(),
                requires_grad=cpu_parameter.requires_grad,
            )
            if cpu_parameter.grad is not None:
                staged.grad = cpu_parameter.grad.detach().to(optimizer_device, non_blocking=True).clone()
            staged_parameters.append(staged)

        optimizer = optimizer_cls(staged_parameters, **dict(optimizer_kwargs))
        for (name, _), staged in zip(named_parameters, staged_parameters):
            saved_state = self.optimizer_state.get(name)
            if saved_state is not None:
                optimizer.state[staged] = _Tree.to_device(saved_state, optimizer_device)

        optimizer.step()

        for (name, cpu_parameter), staged in zip(named_parameters, staged_parameters):
            cpu_parameter.data.copy_(staged.detach().cpu())
            if staged in optimizer.state:
                self.optimizer_state[name] = _Tree.detach_cpu(optimizer.state[staged])
            cpu_parameter.grad = None

        del optimizer, staged_parameters
        self.evict_device(optimizer_device)


class _OffloadedStageFunction(torch.autograd.Function):
    """Autograd boundary that saves activations but not GPU parameter copies."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x: Tensor,
        autograd_token: Tensor,
        handle: _OffloadedStageHandle,
        replica_index: int,
    ) -> Tensor:
        del autograd_token  # The token merely forces a grad_fn when x is data.
        device = x.device
        ctx.handle = handle
        ctx.replica_index = replica_index
        ctx.input_requires_grad = bool(x.requires_grad)
        ctx.rng_snapshot = _RNGSnapshot(device)
        ctx.save_for_backward(x)

        with torch.no_grad():
            y = handle.forward_on_device(x, device)

        # No persistent parameter copy is kept after this point.
        handle.evict_device(device)
        return y

    @staticmethod
    def backward(ctx: Any, grad_y: Tensor) -> Tuple[Optional[Tensor], None, None, None]:  # type: ignore[override]
        (saved_x,) = ctx.saved_tensors
        handle: _OffloadedStageHandle = ctx.handle
        device = saved_x.device

        with ctx.rng_snapshot.fork(), torch.enable_grad():
            x = saved_x.detach()
            if ctx.input_requires_grad:
                x.requires_grad_(True)
            y, params = handle.replay_for_backward(x, device)

            targets: List[Tensor] = []
            target_kinds: List[Tuple[str, Optional[str]]] = []
            if ctx.input_requires_grad:
                targets.append(x)
                target_kinds.append(("input", None))
            for name, parameter in params.items():
                if parameter.requires_grad:
                    targets.append(parameter)
                    target_kinds.append(("param", name))

            if targets:
                grads = torch.autograd.grad(
                    outputs=y,
                    inputs=targets,
                    grad_outputs=grad_y,
                    allow_unused=True,
                    retain_graph=False,
                    create_graph=False,
                )
            else:
                grads = tuple()

        grad_x: Optional[Tensor] = None
        grads_by_name: Dict[str, Optional[Tensor]] = {name: None for name in handle.param_names}
        for (kind, name), grad in zip(target_kinds, grads):
            if kind == "input":
                grad_x = grad
            else:
                assert name is not None
                grads_by_name[name] = grad

        handle.accumulate_mean_gradients(grads_by_name)
        handle.evict_device(device)
        return grad_x, None, None, None


class CPUStreamingDataParallel(nn.Module):
    """Layer-streaming data parallel wrapper with CPU master parameters.

    Parameters
    ----------
    stages:
        Ordered stage specifications.  The order is the forward order.
    devices:
        Participating devices.  For CPU tests, pass ``["cpu", "cpu"]`` to
        emulate two DDP replicas.  For a GH200-like node, pass something like
        ``["cuda:0", "cuda:1", "cuda:2", "cuda:3"]``.
    optimizer_cls, optimizer_kwargs:
        A real PyTorch optimizer class and its keyword arguments.  ``step()``
        dispatches to this class for both offloaded and resident parameters.
        The intended optimizers are Adam, AdamW, and SGD without closures.
    optimizer_devices:
        Devices used for streamed optimizer steps.  If omitted, ``devices`` are
        used.  Stage ``i`` is assigned to ``optimizer_devices[i % n]``.  This is
        the prototype's stand-in for sharding optimizer work and state across the
        CPU/GPU complexes in the node.
    max_grad_norm:
        If set, ``step()`` first averages resident gradients, then applies
        ``torch.nn.utils.clip_grad_norm_`` to the complete set of offloaded CPU
        master parameters and resident master parameters.

    Training-loop contract
    ----------------------

    The wrapper does not move user inputs.  The caller must place the ``i``-th
    input tensor on ``devices[i]``.  The wrapper also assumes DDP-like loss
    scaling: use the sum of local losses and let the wrapper average gradients.

    Example
    -------

    >>> stages = [
    ...     StageSpec("embed", nn.Linear(16, 32), offload=False),
    ...     StageSpec("block0", nn.Sequential(nn.GELU(), nn.Linear(32, 32)), offload=True),
    ...     StageSpec("head", nn.Linear(32, 4), offload=True),
    ... ]
    >>> model = CPUStreamingDataParallel(
    ...     stages,
    ...     devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
    ...     optimizer_cls=torch.optim.AdamW,
    ...     optimizer_kwargs={"lr": 1e-4, "weight_decay": 0.01, "foreach": False},
    ...     max_grad_norm=1.0,
    ... )
    >>> model.zero_grad()
    >>> outputs = model([batch0, batch1, batch2, batch3])
    >>> loss_device = outputs[0].device
    >>> loss = sum(criterion(out, target).to(loss_device) for out, target in zip(outputs, targets))
    >>> loss.backward()
    >>> total_norm = model.step()
    >>> final_model_on_gpu0 = model.close()
    """

    def __init__(
        self,
        stages: Sequence[StageSpec],
        devices: Optional[Sequence[DeviceLike]] = None,
        *,
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.AdamW,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        optimizer_devices: Optional[Sequence[DeviceLike]] = None,
        max_grad_norm: Optional[float] = None,
        grad_norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
    ) -> None:
        super().__init__()
        if not stages:
            raise ValueError("CPUStreamingDataParallel needs at least one stage")

        if devices is None:
            if torch.cuda.is_available():
                devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
            else:
                devices = [torch.device("cpu")]
        self.devices = [torch.device(device) for device in devices]
        if not self.devices:
            raise ValueError("devices must not be empty")

        if optimizer_devices is None:
            optimizer_devices = self.devices
        self.optimizer_devices = [torch.device(device) for device in optimizer_devices]
        if not self.optimizer_devices:
            raise ValueError("optimizer_devices must not be empty")

        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        self.max_grad_norm = max_grad_norm
        self.grad_norm_type = grad_norm_type
        self.error_if_nonfinite = error_if_nonfinite
        self._closed = False

        self.stage_specs = list(stages)
        self._offloaded: Dict[str, _OffloadedStageHandle] = {}
        self._resident_replicas: Dict[str, List[nn.Module]] = {}
        self._resident_optimizers: Dict[str, torch.optim.Optimizer] = {}
        self._stage_order: List[Tuple[str, bool]] = []

        # Register CPU offloaded modules and resident-master modules so state
        # appears in reprs and ordinary module traversal can still find them.
        self.offloaded_cpu_modules = nn.ModuleDict()
        self.resident_master_modules = nn.ModuleDict()

        names_seen = set()
        for stage_index, spec in enumerate(self.stage_specs):
            if spec.name in names_seen:
                raise ValueError(f"duplicate stage name: {spec.name!r}")
            names_seen.add(spec.name)
            self._stage_order.append((spec.name, spec.offload))

            if spec.offload:
                handle = _OffloadedStageHandle(
                    name=spec.name,
                    module=spec.module,
                    replica_count=len(self.devices),
                    reduce_device=self.devices[0],
                )
                self._offloaded[spec.name] = handle
                self.offloaded_cpu_modules[spec.name] = handle.module
            else:
                replicas = [copy.deepcopy(spec.module).to(device) for device in self.devices]
                self._resident_replicas[spec.name] = replicas
                self.resident_master_modules[spec.name] = replicas[0]
                params = list(replicas[0].parameters())
                if params:
                    self._resident_optimizers[spec.name] = self.optimizer_cls(params, **self.optimizer_kwargs)
                self._broadcast_resident_stage(spec.name)

        self._tokens: List[Tensor] = [torch.ones((), device=device, requires_grad=True) for device in self.devices]

    def train(self, mode: bool = True) -> "CPUStreamingDataParallel":
        super().train(mode)
        for handle in self._offloaded.values():
            handle.train(mode)
        for replicas in self._resident_replicas.values():
            for replica in replicas:
                replica.train(mode)
        return self

    def forward(self, inputs: Sequence[Tensor]) -> List[Tensor]:  # type: ignore[override]
        """Run all local replicas.

        ``inputs[i]`` is processed on ``devices[i]``.  The method deliberately
        rejects misplaced tensors rather than silently copying them, because the
        surrounding training loop is responsible for data placement.
        """

        if self._closed:
            raise RuntimeError("close() has been called; create a new wrapper for more training")
        if len(inputs) != len(self.devices):
            raise ValueError(f"expected {len(self.devices)} input tensors, got {len(inputs)}")

        outputs: List[Tensor] = []
        for replica_index, (x, device) in enumerate(zip(inputs, self.devices)):
            if x.device != device:
                raise ValueError(
                    f"input {replica_index} is on {x.device}, but this replica expects {device}. "
                    "Move inputs in the training loop."
                )
            y = x
            for stage_name, offload in self._stage_order:
                if offload:
                    y = _OffloadedStageFunction.apply(
                        y,
                        self._tokens[replica_index],
                        self._offloaded[stage_name],
                        replica_index,
                    )
                else:
                    y = self._resident_replicas[stage_name][replica_index](y)
            outputs.append(y)
        return outputs

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        for handle in self._offloaded.values():
            handle.zero_grad(set_to_none=set_to_none)
        for replicas in self._resident_replicas.values():
            for replica in replicas:
                for parameter in replica.parameters():
                    if set_to_none:
                        parameter.grad = None
                    elif parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                    else:
                        parameter.grad.zero_()

    def _broadcast_resident_stage(self, stage_name: str) -> None:
        replicas = self._resident_replicas[stage_name]
        if len(replicas) <= 1:
            return
        source = replicas[0]
        with torch.no_grad():
            source_state = source.state_dict()
            for replica in replicas[1:]:
                copied = {k: v.detach().to(next(replica.parameters(), torch.empty((), device=self.devices[0])).device) if torch.is_tensor(v) else v for k, v in source_state.items()}
                # load_state_dict handles device copies, but the explicit copy
                # above avoids surprises for buffers on unusual modules.
                replica.load_state_dict(copied, strict=True)

    def _average_resident_gradients(self) -> None:
        """Average resident replica gradients into replica 0, then mirror them.

        This is the single-process analog of DDP's all-reduce.  The optimizer for
        resident parameters only owns replica 0, so replica 0 is the master.  All
        other replicas receive the averaged gradient only for inspection; they do
        not perform an optimizer step.
        """

        for stage_name, replicas in self._resident_replicas.items():
            if not replicas:
                continue
            named_parameters_per_replica = [OrderedDict(replica.named_parameters()) for replica in replicas]
            master_named = named_parameters_per_replica[0]
            for name, master_parameter in master_named.items():
                grads: List[Tensor] = []
                for replica_named in named_parameters_per_replica:
                    grad = replica_named[name].grad
                    if grad is not None:
                        grads.append(grad.detach().to(master_parameter.device, non_blocking=True))
                if grads:
                    mean_grad = torch.stack(grads, dim=0).mean(dim=0)
                    master_parameter.grad = mean_grad.clone()
                else:
                    master_parameter.grad = None

            # Optional mirroring helps debugging and makes every replica look DDP-like.
            for replica in replicas[1:]:
                replica_named = OrderedDict(replica.named_parameters())
                for name, master_parameter in master_named.items():
                    if master_parameter.grad is None:
                        replica_named[name].grad = None
                    else:
                        replica_named[name].grad = master_parameter.grad.detach().to(replica_named[name].device).clone()

    def _parameters_for_clipping(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for handle in self._offloaded.values():
            params.extend(handle.parameters_by_name.values())
        for stage_name in self._resident_replicas:
            params.extend(self._resident_replicas[stage_name][0].parameters())
        return params

    def step(self) -> Optional[Tensor]:
        """Average resident grads, optionally clip, then run optimizer updates.

        Returns the total gradient norm reported by ``clip_grad_norm_`` when
        clipping is enabled; otherwise returns ``None``.
        """

        self._average_resident_gradients()

        total_norm: Optional[Tensor]
        if self.max_grad_norm is not None:
            total_norm = clip_grad_norm_(
                self._parameters_for_clipping(),
                max_norm=float(self.max_grad_norm),
                norm_type=float(self.grad_norm_type),
                error_if_nonfinite=bool(self.error_if_nonfinite),
            )
        else:
            total_norm = None

        for stage_index, (stage_name, offload) in enumerate(self._stage_order):
            if offload:
                optimizer_device = self.optimizer_devices[stage_index % len(self.optimizer_devices)]
                self._offloaded[stage_name].optimizer_step(
                    optimizer_cls=self.optimizer_cls,
                    optimizer_kwargs=self.optimizer_kwargs,
                    optimizer_device=optimizer_device,
                )
            else:
                optimizer = self._resident_optimizers.get(stage_name)
                if optimizer is not None:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                self._broadcast_resident_stage(stage_name)

        # Gradients on non-master resident replicas are inspection artifacts only.
        for replicas in self._resident_replicas.values():
            for replica in replicas[1:]:
                for parameter in replica.parameters():
                    parameter.grad = None

        return total_norm

    def offloaded_parameters(self) -> Iterator[nn.Parameter]:
        for handle in self._offloaded.values():
            yield from handle.parameters_by_name.values()

    def resident_master_parameters(self) -> Iterator[nn.Parameter]:
        for stage_name in self._resident_replicas:
            yield from self._resident_replicas[stage_name][0].parameters()

    def materialize(self, device: Optional[DeviceLike] = None) -> nn.Sequential:
        """Build a normal ``nn.Sequential`` model from the current master state."""

        target_device = torch.device(device) if device is not None else torch.device("cpu")
        modules: "OrderedDict[str, nn.Module]" = OrderedDict()
        for stage_name, offload in self._stage_order:
            if offload:
                modules[stage_name] = copy.deepcopy(self._offloaded[stage_name].module).to(target_device)
            else:
                modules[stage_name] = copy.deepcopy(self._resident_replicas[stage_name][0]).to(target_device)
        return nn.Sequential(modules)

    def close(self) -> nn.Sequential:
        """Evict device-side state and return a normal model on device 0.

        Offloaded stages have no persistent GPU tensors to clear.  Resident stages
        are collapsed to replica 0; other replicas are dropped.  The returned
        model is a deep copy, so the caller can save it or continue with ordinary
        PyTorch training.
        """

        result = self.materialize(self.devices[0])
        for handle in self._offloaded.values():
            for device in self.devices:
                handle.evict_device(device)
        self._resident_replicas.clear()
        self._resident_optimizers.clear()
        self._closed = True
        return result
