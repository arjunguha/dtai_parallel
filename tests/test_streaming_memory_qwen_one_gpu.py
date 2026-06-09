"""One-GPU Qwen2.5-Coder-14B streaming memory benchmark.

This is a single-process port of ``test_streaming_memory_qwen.py``.  It is
guarded to run only on the Vista cluster and only when exactly one CUDA device is
visible.  The model is loaded by repo ID so Hugging Face resolves the cached
snapshot normally.

Run with verbose output to see the cross-sequence-length comparison table::

    pytest tests/test_streaming_memory_qwen_one_gpu.py -v -s
"""

from __future__ import annotations

import os
import socket
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
from transformers import AutoModelForCausalLM

from dtai_parallel import apply_cpu_streaming_

BATCH_SIZE = 1
TRAIN_STEPS = 3
STREAM_MODULE_PATH = "model.layers"
QWEN_MODEL_NAME = "Qwen/Qwen2.5-Coder-14B-Instruct"
SEQUENCE_LENGTHS = [1000, 2000, 4000, 8000]
VISTA_HOST_SUFFIX = ".vista.tacc.utexas.edu"
RESIDENT_SUFFIX_COUNT = int(os.environ.get("DTAI_QWEN_RESIDENT_SUFFIX_COUNT", os.environ.get("DTAI_QWEN_RESIDENT_SUFFIX", "0")))


@dataclass(frozen=True)
class QwenMemoryRow:
    model_name: str
    seq_len: int
    num_layers: int
    resident_suffix_count: int
    parameter_count: int
    offloaded_total_gib: float
    largest_layer_gib: float
    resident_gib: float
    measured_cpu_peak_gib: float
    measured_cuda_peak_gib: float
    step_seconds: Tuple[float, ...]
    average_step_seconds: float

    @property
    def sort_key(self) -> int:
        return self.seq_len


def running_on_vista() -> bool:
    return socket.gethostname().endswith(VISTA_HOST_SUFFIX)


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


def offloaded_layer_bytes(model: nn.Module, resident_suffix_count: int = 0) -> Tuple[int, int, int]:
    layers = model.get_submodule(STREAM_MODULE_PATH)
    layer_bytes = [parameter_bytes(layer) for layer in layers]
    if resident_suffix_count:
        layer_bytes = layer_bytes[:-resident_suffix_count]
    return len(layers), sum(layer_bytes), max(layer_bytes) if layer_bytes else 0


def resident_layer_bytes(model: nn.Module, resident_suffix_count: int = 0) -> int:
    layers = model.get_submodule(STREAM_MODULE_PATH)
    offloaded_layers = list(layers)
    if resident_suffix_count:
        offloaded_layers = offloaded_layers[:-resident_suffix_count]
    layer_param_ids = {id(parameter) for layer in offloaded_layers for parameter in layer.parameters()}
    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
        if id(parameter) not in layer_param_ids
    )


def _format_table_cell(text: str, width: int, *, align: str = "right") -> str:
    if align == "left":
        return text.ljust(width)
    if align == "center":
        return text.center(width)
    return text.rjust(width)


def print_qwen_memory_table(rows: List[QwenMemoryRow]) -> None:
    if not rows:
        return

    columns: List[Tuple[str, int, str]] = [
        ("seq len", 8, "right"),
        ("layers", 6, "right"),
        ("resident tail", 13, "right"),
        ("params (B)", 10, "right"),
        ("offloaded (GiB)", 15, "right"),
        ("largest (GiB)", 13, "right"),
        ("resident (GiB)", 14, "right"),
        ("CPU RSS (GiB)", 15, "right"),
        ("CUDA peak (GiB)", 16, "right"),
        ("step 1 (s)", 10, "right"),
        ("step 2 (s)", 10, "right"),
        ("step 3 (s)", 10, "right"),
        ("avg step (s)", 12, "right"),
    ]

    ordered = sorted(rows, key=lambda row: row.sort_key)
    header = "  ".join(_format_table_cell(title, width, align=align) for title, width, align in columns)
    rule = "  ".join("-" * width for _, width, _ in columns)

    print("\n--- one-GPU Qwen streaming memory ---")
    print(f"model: {ordered[0].model_name}")
    print(header)
    print(rule)
    for row in ordered:
        values = [
            f"{row.seq_len:d}",
            f"{row.num_layers:d}",
            f"{row.resident_suffix_count:d}",
            f"{row.parameter_count / 1e9:.3f}",
            f"{row.offloaded_total_gib:.3f}",
            f"{row.largest_layer_gib:.3f}",
            f"{row.resident_gib:.3f}",
            f"{row.measured_cpu_peak_gib:.3f}",
            f"{row.measured_cuda_peak_gib:.3f}",
            f"{row.step_seconds[0]:.3f}",
            f"{row.step_seconds[1]:.3f}",
            f"{row.step_seconds[2]:.3f}",
            f"{row.average_step_seconds:.3f}",
        ]
        print(
            "  ".join(
                _format_table_cell(value, width, align=align)
                for value, (_, width, align) in zip(values, columns)
            )
        )
    print("--- end one-GPU Qwen table ---\n")


@pytest.fixture(scope="session")
def qwen_memory_rows() -> List[QwenMemoryRow]:
    rows: List[QwenMemoryRow] = []
    yield rows
    print_qwen_memory_table(rows)


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


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_batch(device: torch.device, vocab_size: int, seq_len: int) -> Tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(20260609 + seq_len)
    input_ids = torch.randint(
        0,
        vocab_size,
        (BATCH_SIZE, seq_len),
        generator=generator,
        dtype=torch.long,
    ).to(device)
    return input_ids, input_ids.clone()


def run_one_gpu_qwen_memory_case(seq_len: int) -> QwenMemoryRow:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    model_load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    print(f"loaded {QWEN_MODEL_NAME} for seq_len={seq_len} in {time.perf_counter() - model_load_start:.2f}s", flush=True)
    num_layers, offloaded_total_bytes, largest_layer_bytes = offloaded_layer_bytes(model, RESIDENT_SUFFIX_COUNT)
    resident_bytes = resident_layer_bytes(model, RESIDENT_SUFFIX_COUNT)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    engine = apply_cpu_streaming_(
        model,
        STREAM_MODULE_PATH,
        offload_policy=True,
        resident_suffix_count=RESIDENT_SUFFIX_COUNT,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 1e-5, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.01, "foreach": False},
        max_grad_norm=1.0,
        device=device,
        auto_init_process_group=False,
        wrap_ddp=False,
        close_rank=1,
        pin_cpu_masters="lazy",
    )

    vocab_size = model.config.vocab_size
    tracker = IterationMemoryTracker(device)
    iteration_records: List[Dict[str, int]] = []
    step_seconds: List[float] = []

    for _ in range(TRAIN_STEPS):
        synchronize_if_cuda(device)
        step_start = time.perf_counter()

        tracker.begin_iteration()
        engine.zero_grad(set_to_none=True)
        tracker.sample()

        input_ids, labels = make_batch(device, vocab_size, seq_len)
        loss = engine.model(input_ids=input_ids, labels=labels).loss
        tracker.sample()

        loss.backward()
        tracker.sample()

        engine.step()
        tracker.sample()

        synchronize_if_cuda(device)
        step_seconds.append(time.perf_counter() - step_start)

        iteration_records.append(
            {
                "cpu_rss_peak_bytes": tracker.cpu_rss_peak,
                "cuda_peak_bytes": tracker.cuda_peak,
            }
        )

    closed = engine.close(device=torch.device("cpu"))
    assert closed is None
    torch.cuda.empty_cache()

    cpu_peak = max(record["cpu_rss_peak_bytes"] for record in iteration_records)
    cuda_peak = max(record["cuda_peak_bytes"] for record in iteration_records)
    average_step_seconds = sum(step_seconds[1:]) / len(step_seconds[1:])

    return QwenMemoryRow(
        model_name=QWEN_MODEL_NAME,
        seq_len=seq_len,
        num_layers=num_layers,
        resident_suffix_count=RESIDENT_SUFFIX_COUNT,
        parameter_count=parameter_count,
        offloaded_total_gib=gib(offloaded_total_bytes),
        largest_layer_gib=gib(largest_layer_bytes),
        resident_gib=gib(resident_bytes),
        measured_cpu_peak_gib=gib(cpu_peak),
        measured_cuda_peak_gib=gib(cuda_peak),
        step_seconds=tuple(step_seconds),
        average_step_seconds=average_step_seconds,
    )


@pytest.mark.parametrize(
    "seq_len",
    SEQUENCE_LENGTHS,
    ids=[f"seq{value}" for value in SEQUENCE_LENGTHS],
)
def test_one_gpu_qwen14b_streaming_peak_memory(
    seq_len: int,
    qwen_memory_rows: List[QwenMemoryRow],
) -> None:
    if not running_on_vista():
        pytest.skip("runs only on Vista cluster hosts")
    if not torch.cuda.is_available():
        pytest.skip("needs exactly one CUDA device")
    if torch.cuda.device_count() != 1:
        pytest.skip("needs exactly one visible CUDA device")

    row = run_one_gpu_qwen_memory_case(seq_len)
    qwen_memory_rows.append(row)

    assert row.parameter_count >= 13_000_000_000
    assert row.measured_cpu_peak_gib > 0
    assert row.measured_cuda_peak_gib > 0
