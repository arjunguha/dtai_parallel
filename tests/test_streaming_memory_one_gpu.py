"""One-GPU peak memory characterization for CPU-streamed CUDA training.

This is a single-process port of ``test_streaming_memory.py``.  It deliberately
requires exactly one visible CUDA device and does not initialize distributed
training, so it can validate the streaming path on a one-GPU machine without
exercising the two-GPU DDP tests.

Run with verbose output to see the cross-configuration comparison table::

    pytest tests/test_streaming_memory_one_gpu.py -v -s
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")

import pytest
import torch
from torch import Tensor, nn

from dtai_parallel import apply_cpu_streaming_

INPUT_DIM = 128
OUTPUT_DIM = 128
BATCH_SIZE = 2
TRAIN_STEPS = 3
ADAMW_FACTOR = 4
PIN_CPU_MASTERS_MODE = os.environ.get("DTAI_PIN_CPU_MASTERS", "eager")

# Calibrated to the original four-layer reference (width 7936): ~500M
# offloaded params.  Keeping this identical to the two-GPU memory test makes the
# one-GPU measurements directly comparable.
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
    pin_mode: str
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
    average_step_seconds: float

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
        ("pin mode", 8, "left"),
        ("layer scale", 11, "left"),
        ("width", 6, "right"),
        ("params (B)", 10, "right"),
        ("largest (GiB)", 13, "right"),
        ("exp CUDA", 10, "right"),
        ("meas CUDA", 10, "right"),
        ("CUDA ratio", 10, "right"),
        ("avg step", 9, "right"),
        ("exp CPU", 10, "right"),
        ("meas CPU", 10, "right"),
    ]

    ordered = sorted(rows, key=lambda row: row.sort_key)
    header = "  ".join(_format_table_cell(title, width, align=align) for title, width, align in columns)
    rule = "  ".join("-" * width for _, width, _ in columns)

    print("\n--- one-GPU streaming memory comparison ---")
    print(header)
    print(rule)
    for row in ordered:
        values = [
            f"{row.num_layers:d}",
            row.pin_mode,
            row.scale_label,
            f"{row.layer_width:d}",
            f"{row.parameter_count / 1e9:.3f}",
            f"{row.largest_layer_gib:.3f}",
            f"{row.expected_cuda_peak_gib:.3f}",
            f"{row.measured_cuda_peak_gib:.3f}",
            f"{row.cuda_ratio:.2f}x",
            f"{row.average_step_seconds:.3f}",
            f"{row.expected_cpu_peak_gib:.3f}",
            f"{row.measured_cpu_peak_gib:.3f}",
        ]
        print(
            "  ".join(
                _format_table_cell(value, width, align=align)
                for value, (_, width, align) in zip(values, columns)
            )
        )
    print("--- end one-GPU comparison ---\n")


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
        torch.cuda.reset_peak_memory_stats(self.device)

    def sample(self) -> None:
        self.cpu_rss_peak = max(self.cpu_rss_peak, read_rss_bytes())
        self.cuda_peak = max(self.cuda_peak, int(torch.cuda.max_memory_allocated(self.device)))


def make_batch(device: torch.device) -> Tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(20260609)
    x = torch.randn(BATCH_SIZE, INPUT_DIM, generator=generator, dtype=torch.float32).to(device)
    y = torch.randn(BATCH_SIZE, OUTPUT_DIM, generator=generator, dtype=torch.float32).to(device)
    return x, y


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_one_gpu_streaming_memory_case(num_layers: int, layer_width: int) -> List[Dict[str, int]]:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    model = LargeStreamingSandwich(num_layers, layer_width)
    engine = apply_cpu_streaming_(
        model,
        "layers",
        offload_policy=True,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 3e-4, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.01, "foreach": False},
        max_grad_norm=1.0,
        device=device,
        auto_init_process_group=False,
        wrap_ddp=False,
        pin_cpu_masters=PIN_CPU_MASTERS_MODE,
    )

    criterion = nn.MSELoss()
    tracker = IterationMemoryTracker(device)
    iteration_records: List[Dict[str, int]] = []

    for _ in range(TRAIN_STEPS):
        synchronize_if_cuda(device)
        step_start = time.perf_counter()

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

        synchronize_if_cuda(device)
        iteration_records.append(
            {
                "cpu_rss_peak_bytes": tracker.cpu_rss_peak,
                "cuda_peak_bytes": tracker.cuda_peak,
                "step_seconds": time.perf_counter() - step_start,
            }
        )

    closed = engine.close(return_on_all_ranks=True, device=torch.device("cpu"))
    assert closed is not None
    torch.cuda.empty_cache()
    return iteration_records


@pytest.mark.parametrize(
    "num_layers,layer_param_scale",
    MEMORY_STUDY_CASES,
    ids=["4-full", "8-full", "16-full", "8-half-layer", "16-half-layer"],
)
def test_one_gpu_streaming_peak_memory_is_bounded_by_layerwise_offload(
    num_layers: int,
    layer_param_scale: float,
    memory_study_results: List[MemoryStudyRow],
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs exactly one CUDA device")
    if torch.cuda.device_count() != 1:
        pytest.skip("needs exactly one visible CUDA device")

    layer_width = layer_width_for_config(num_layers, layer_param_scale)
    model = LargeStreamingSandwich(num_layers, layer_width)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if layer_param_scale == 1.0:
        assert parameter_count >= 500_000_000
    else:
        assert parameter_count >= 100_000_000

    offloaded_total_bytes, largest_layer_bytes = offloaded_layer_bytes(model)
    resident_bytes = resident_layer_bytes(model)
    expected_cpu_peak = ADAMW_FACTOR * offloaded_total_bytes
    expected_cuda_peak = ADAMW_FACTOR * (largest_layer_bytes + resident_bytes)
    del model

    records = run_one_gpu_streaming_memory_case(num_layers, layer_width)
    assert len(records) == TRAIN_STEPS

    cpu_peak = max(record["cpu_rss_peak_bytes"] for record in records)
    cuda_peak = max(record["cuda_peak_bytes"] for record in records)

    measured_cpu_peak_gib = gib(cpu_peak)
    measured_cuda_peak_gib = gib(cuda_peak)
    average_step_seconds = sum(float(record["step_seconds"]) for record in records[1:]) / len(records[1:])
    config_key = (num_layers, layer_param_scale)
    MEASURED_CPU_PEAK_GIB[config_key] = measured_cpu_peak_gib
    MEASURED_CUDA_PEAK_GIB[config_key] = measured_cuda_peak_gib

    memory_study_results.append(
        MemoryStudyRow(
            pin_mode=PIN_CPU_MASTERS_MODE,
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
            average_step_seconds=average_step_seconds,
        )
    )

    # Offloaded AdamW state lives on CPU: weights, transient gradients, and two
    # moments.  VmRSS includes interpreter, library, and allocator memory, so the
    # one-GPU port uses lower-bound and cross-configuration checks instead of the
    # two-process test's absolute RSS upper bound.
    assert cpu_peak >= int(0.85 * expected_cpu_peak)

    post_first_step = records[1:]
    assert any(record["cpu_rss_peak_bytes"] >= int(0.85 * expected_cpu_peak) for record in post_first_step)

    # CUDA peak is dominated by optimizer.step(): staged weights, gradients, and
    # AdamW state for the largest offloaded layer, plus resident modules.
    assert cuda_peak >= int(0.85 * expected_cuda_peak)

    if layer_param_scale == 0.5:
        full_cuda_peak_gib = MEASURED_CUDA_PEAK_GIB[(num_layers, 1.0)]
        assert measured_cuda_peak_gib < full_cuda_peak_gib
