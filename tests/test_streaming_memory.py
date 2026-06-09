"""Peak memory characterization for CPU-streamed CUDA training.

CPU footprint is measured with process RSS (``VmRSS`` from ``/proc/self/status``),
sampled at several points each iteration.  CUDA footprint uses
``torch.cuda.max_memory_allocated()`` reset at the start of each iteration.

Run with verbose output to see the cross-configuration comparison table::

    pytest tests/test_streaming_memory.py -v -s

The ``-s`` flag disables output capture so the table is printed to the terminal.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from dtai_parallel import apply_cpu_streaming_

INPUT_DIM = 128
OUTPUT_DIM = 128
BATCH_SIZE = 2
TRAIN_STEPS = 3
ADAMW_FACTOR = 4

# Calibrated to the original four-layer reference (width 7936): ~500M offloaded params.
REFERENCE_LAYER_WIDTH = 7936
REFERENCE_NUM_LAYERS = 4
TARGET_OFFLOADED_PARAMS = REFERENCE_NUM_LAYERS * 2 * (
    REFERENCE_LAYER_WIDTH * REFERENCE_LAYER_WIDTH + REFERENCE_LAYER_WIDTH
)

MEMORY_STUDY_CASES = [
    (4, 1.0),
    (8, 1.0),
    (16, 1.0),
    (8, 0.5),
    (16, 0.5),
]

MEASURED_CUDA_PEAK_GIB: Dict[Tuple[int, float], float] = {}
MEASURED_CPU_PEAK_GIB: Dict[Tuple[int, float], float] = {}


@dataclass(frozen=True)
class MemoryStudyRow:
    num_layers: int
    layer_param_scale: float
    layer_width: int
    parameter_count: int
    offloaded_total_gib: float
    largest_layer_gib: float
    resident_gib: float
    expected_cpu_peak_gib: float
    measured_cpu_peak_gib: float
    expected_cuda_peak_gib: float
    measured_cuda_peak_gib: float
    measured_transfer_h2d_gib: float
    measured_transfer_d2h_gib: float
    measured_transfer_cuda_ms: float

    @property
    def scale_label(self) -> str:
        return "full" if self.layer_param_scale == 1.0 else "half layer"

    @property
    def cuda_ratio(self) -> float:
        return self.measured_cuda_peak_gib / self.expected_cuda_peak_gib

    @property
    def sort_key(self) -> Tuple[int, float]:
        return (self.num_layers, -self.layer_param_scale)


def layer_width_for_num_layers(num_layers: int) -> int:
    target_per_layer = TARGET_OFFLOADED_PARAMS // num_layers
    return int((math.sqrt(1 + 4 * (target_per_layer // 2)) - 1) / 2)


def layer_width_for_config(num_layers: int, layer_param_scale: float) -> int:
    base_width = layer_width_for_num_layers(num_layers)
    if layer_param_scale == 1.0:
        return base_width
    return max(1, int(base_width * math.sqrt(layer_param_scale)))


class LargeResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(width, width)
        self.act = nn.GELU()
        self.lin2 = nn.Linear(width, width)

    def forward(self, x: Tensor) -> Tensor:
        return x + 0.25 * self.lin2(self.act(self.lin1(x)))


class LargeStreamingSandwich(nn.Module):
    """Embedding and head around a large streamed ModuleList."""

    def __init__(self, num_layers: int, layer_width: int) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.layer_width = int(layer_width)
        self.embed = nn.Linear(INPUT_DIM, layer_width)
        self.layers = nn.ModuleList(LargeResidualBlock(layer_width) for _ in range(num_layers))
        self.norm = nn.LayerNorm(layer_width)
        self.unembed = nn.Linear(layer_width, OUTPUT_DIM)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.unembed(self.norm(x))


def read_rss_bytes() -> int:
    with open("/proc/self/status", encoding="ascii") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS not found in /proc/self/status")


def parameter_bytes(module: nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())


def gib(bytes_value: int | float) -> float:
    return float(bytes_value) / (1024**3)


def offloaded_layer_bytes(model: LargeStreamingSandwich) -> Tuple[int, int]:
    layer_bytes = [parameter_bytes(layer) for layer in model.layers]
    return sum(layer_bytes), max(layer_bytes)


def resident_layer_bytes(model: LargeStreamingSandwich) -> int:
    return parameter_bytes(model.embed) + parameter_bytes(model.norm) + parameter_bytes(model.unembed)


def _format_table_cell(text: str, width: int, *, align: str = "right") -> str:
    if align == "left":
        return text.ljust(width)
    if align == "center":
        return text.center(width)
    return text.rjust(width)


def print_memory_comparison_table(rows: List[MemoryStudyRow]) -> None:
    if not rows:
        return

    columns: List[Tuple[str, int, str]] = [
        ("layers", 6, "right"),
        ("layer scale", 11, "left"),
        ("width", 6, "right"),
        ("params (B)", 10, "right"),
        ("largest (GiB)", 13, "right"),
        ("exp CUDA", 10, "right"),
        ("meas CUDA", 10, "right"),
        ("CUDA ratio", 10, "right"),
        ("H2D GiB", 9, "right"),
        ("D2H GiB", 9, "right"),
        ("copy ms", 9, "right"),
        ("exp CPU", 10, "right"),
        ("meas CPU", 10, "right"),
    ]

    ordered = sorted(rows, key=lambda row: row.sort_key)
    header = "  ".join(_format_table_cell(title, width, align=align) for title, width, align in columns)
    rule = "  ".join("-" * width for _, width, _ in columns)

    print("\n--- streaming memory comparison ---")
    print(header)
    print(rule)
    for row in ordered:
        values = [
            f"{row.num_layers:d}",
            row.scale_label,
            f"{row.layer_width:d}",
            f"{row.parameter_count / 1e9:.3f}",
            f"{row.largest_layer_gib:.3f}",
            f"{row.expected_cuda_peak_gib:.3f}",
            f"{row.measured_cuda_peak_gib:.3f}",
            f"{row.cuda_ratio:.2f}x",
            f"{row.measured_transfer_h2d_gib:.3f}",
            f"{row.measured_transfer_d2h_gib:.3f}",
            f"{row.measured_transfer_cuda_ms:.1f}",
            f"{row.expected_cpu_peak_gib:.3f}",
            f"{row.measured_cpu_peak_gib:.3f}",
        ]
        print(
            "  ".join(
                _format_table_cell(value, width, align=align)
                for value, (_, width, align) in zip(values, columns)
            )
        )
    print("--- end comparison ---\n")


@pytest.fixture(scope="session")
def memory_study_results() -> List[MemoryStudyRow]:
    rows: List[MemoryStudyRow] = []
    yield rows
    print_memory_comparison_table(rows)


class IterationMemoryTracker:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.cpu_rss_peak = 0
        self.cuda_peak = 0

    def begin_iteration(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def sample(self) -> None:
        self.cpu_rss_peak = max(self.cpu_rss_peak, read_rss_bytes())
        if self.device.type == "cuda":
            self.cuda_peak = max(self.cuda_peak, int(torch.cuda.max_memory_allocated(self.device)))


def make_batch(device: torch.device) -> Tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(20260609)
    x = torch.randn(BATCH_SIZE, INPUT_DIM, generator=generator, dtype=torch.float32).to(device)
    y = torch.randn(BATCH_SIZE, OUTPUT_DIM, generator=generator, dtype=torch.float32).to(device)
    return x, y


def _streaming_memory_worker(
    rank: int,
    world_size: int,
    init_file: str,
    result_file: str,
    initial_state: Mapping[str, Tensor],
    num_layers: int,
    layer_width: int,
) -> None:
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, init_method=f"file://{init_file}")
    try:
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        model = LargeStreamingSandwich(num_layers, layer_width)
        model.load_state_dict(initial_state, strict=True)
        engine = apply_cpu_streaming_(
            model,
            "layers",
            offload_policy=True,
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs={"lr": 3e-4, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.01, "foreach": False},
            max_grad_norm=1.0,
            device=device,
            auto_init_process_group=False,
            wrap_ddp=True,
            collect_timing=True,
        )
        assert isinstance(engine.model, DDP)

        criterion = nn.MSELoss()
        tracker = IterationMemoryTracker(device)
        iteration_records: List[Dict[str, int]] = []

        for _ in range(TRAIN_STEPS):
            tracker.begin_iteration()
            engine.zero_grad(set_to_none=True)
            tracker.sample()

            x, y = make_batch(device)
            loss = criterion(engine.model(x), y)
            tracker.sample()

            loss.backward()
            tracker.sample()

            engine.step()
            tracker.sample()
            transfer_timing = engine.transfer_timing_summary(reset=True, synchronize=True)

            iteration_records.append(
                {
                    "cpu_rss_peak_bytes": tracker.cpu_rss_peak,
                    "cuda_peak_bytes": tracker.cuda_peak,
                    "transfer_timing": transfer_timing,
                }
            )

        engine.close(device=torch.device("cpu"))
        if rank == 0:
            payload = {"iteration_records": iteration_records}
            with open(result_file, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
    finally:
        dist.destroy_process_group()


@pytest.mark.cuda
@pytest.mark.parametrize(
    "num_layers,layer_param_scale",
    MEMORY_STUDY_CASES,
    ids=["4-full", "8-full", "16-full", "8-half-layer", "16-half-layer"],
)
def test_streaming_peak_memory_is_bounded_by_layerwise_offload(
    num_layers: int,
    layer_param_scale: float,
    memory_study_results: List[MemoryStudyRow],
) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs at least two CUDA devices")
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")

    layer_width = layer_width_for_config(num_layers, layer_param_scale)
    world_size = 2
    initial_model = LargeStreamingSandwich(num_layers, layer_width)
    parameter_count = sum(parameter.numel() for parameter in initial_model.parameters())
    if layer_param_scale == 1.0:
        assert parameter_count >= 500_000_000
    else:
        assert parameter_count >= 100_000_000

    offloaded_total_bytes, largest_layer_bytes = offloaded_layer_bytes(initial_model)
    resident_bytes = resident_layer_bytes(initial_model)
    expected_cpu_peak = ADAMW_FACTOR * offloaded_total_bytes
    expected_cuda_peak = ADAMW_FACTOR * (largest_layer_bytes + resident_bytes)

    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init")
        result_file = os.path.join(tmpdir, "memory.json")
        if os.path.exists(init_file):
            os.unlink(init_file)

        mp.start_processes(
            _streaming_memory_worker,
            args=(
                world_size,
                init_file,
                result_file,
                initial_model.state_dict(),
                num_layers,
                layer_width,
            ),
            nprocs=world_size,
            start_method="spawn",
            join=True,
        )

        with open(result_file, encoding="utf-8") as handle:
            payload = json.load(handle)

    records = payload["iteration_records"]
    assert len(records) == TRAIN_STEPS

    cpu_peak = max(record["cpu_rss_peak_bytes"] for record in records)
    cuda_peak = max(record["cuda_peak_bytes"] for record in records)
    transfer_h2d_bytes = sum(
        int(timing["bytes"])
        for record in records
        for kind, timing in record["transfer_timing"].items()
        if kind.endswith("_h2d")
    )
    transfer_d2h_bytes = sum(
        int(timing["bytes"])
        for record in records
        for kind, timing in record["transfer_timing"].items()
        if kind.endswith("_d2h")
    )
    transfer_cuda_ms = sum(
        float(timing["cuda_ms"])
        for record in records
        for timing in record["transfer_timing"].values()
    )

    measured_cpu_peak_gib = gib(cpu_peak)
    measured_cuda_peak_gib = gib(cuda_peak)
    config_key = (num_layers, layer_param_scale)
    MEASURED_CPU_PEAK_GIB[config_key] = measured_cpu_peak_gib
    MEASURED_CUDA_PEAK_GIB[config_key] = measured_cuda_peak_gib

    memory_study_results.append(
        MemoryStudyRow(
            num_layers=num_layers,
            layer_param_scale=layer_param_scale,
            layer_width=layer_width,
            parameter_count=parameter_count,
            offloaded_total_gib=gib(offloaded_total_bytes),
            largest_layer_gib=gib(largest_layer_bytes),
            resident_gib=gib(resident_bytes),
            expected_cpu_peak_gib=gib(expected_cpu_peak),
            measured_cpu_peak_gib=measured_cpu_peak_gib,
            expected_cuda_peak_gib=gib(expected_cuda_peak),
            measured_cuda_peak_gib=measured_cuda_peak_gib,
            measured_transfer_h2d_gib=gib(transfer_h2d_bytes),
            measured_transfer_d2h_gib=gib(transfer_d2h_bytes),
            measured_transfer_cuda_ms=transfer_cuda_ms,
        )
    )

    assert transfer_h2d_bytes > 0
    assert transfer_d2h_bytes > 0

    # Offloaded AdamW state lives on CPU: weights, transient gradients, and two moments.
    assert cpu_peak >= int(0.85 * expected_cpu_peak)
    if layer_param_scale == 1.0:
        assert cpu_peak <= int(1.35 * expected_cpu_peak)
    else:
        full_cpu_peak_gib = MEASURED_CPU_PEAK_GIB[(num_layers, 1.0)]
        assert measured_cpu_peak_gib < full_cpu_peak_gib

    post_first_step = records[1:]
    assert any(record["cpu_rss_peak_bytes"] >= int(0.85 * expected_cpu_peak) for record in post_first_step)

    # CUDA peak is dominated by optimizer.step(): staged weights, gradients, and AdamW
    # state for the largest offloaded layer, plus the resident modules that already
    # live on device with their own AdamW state.
    assert cuda_peak >= int(0.85 * expected_cuda_peak)

    if layer_param_scale == 0.5:
        full_cuda_peak_gib = MEASURED_CUDA_PEAK_GIB[(num_layers, 1.0)]
        assert measured_cuda_peak_gib < full_cuda_peak_gib
