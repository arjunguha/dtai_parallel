"""Peak memory characterization for CPU-streamed CUDA training on Qwen2.5-Coder-14B.

Uses the same measurement principles as ``test_streaming_memory.py``: CPU RSS from
``/proc/self/status`` and CUDA peak from ``torch.cuda.max_memory_allocated()``.

Sequence length is varied across 1k, 2k, 4k, and 8k tokens to show how activation
memory affects the CUDA peak while CPU offload footprint stays fixed.  Step time is
averaged over all training steps except the first warmup step.

Run with verbose output to see the cross-sequence-length comparison table::

    pytest tests/test_streaming_memory_qwen.py -v -s

The ``-s`` flag disables output capture so the table is printed to the terminal.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
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
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM

from dtai_parallel import apply_cpu_streaming_

BATCH_SIZE = 1
TRAIN_STEPS = 4
STREAM_MODULE_PATH = "model.layers"
SEQUENCE_LENGTHS = [1000, 2000, 4000, 8000]
QWEN_MODEL_DIRNAMES = (
    "qwen2p5_coder_14b_instruct",
    "Qwen2.5-Coder-14B-Instruct",
)


@dataclass(frozen=True)
class QwenMemoryRow:
    model_name: str
    seq_len: int
    num_layers: int
    parameter_count: int
    offloaded_total_gib: float
    largest_layer_gib: float
    resident_gib: float
    measured_cpu_peak_gib: float
    measured_cuda_peak_gib: float
    avg_step_seconds: float

    @property
    def sort_key(self) -> int:
        return self.seq_len


def resolve_qwen_14b_model_path() -> Path | None:
    search_roots = [
        Path.home() / "Models",
        Path.home() / "models",
        Path("/mnt/ssd/arjun/Models"),
        Path("/mnt/ssd/arjun/models"),
    ]
    for root in search_roots:
        if not root.is_dir():
            continue
        for dirname in QWEN_MODEL_DIRNAMES:
            candidate = root / dirname
            if (candidate / "config.json").is_file():
                return candidate
    return None


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


def offloaded_layer_bytes(model: nn.Module) -> Tuple[int, int, int]:
    layers = model.get_submodule(STREAM_MODULE_PATH)
    layer_bytes = [parameter_bytes(layer) for layer in layers]
    return len(layers), sum(layer_bytes), max(layer_bytes)


def resident_layer_bytes(model: nn.Module) -> int:
    layer_param_ids = {id(parameter) for parameter in model.get_submodule(STREAM_MODULE_PATH).parameters()}
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
        ("CUDA peak (GiB)", 16, "right"),
        ("CPU RSS (GiB)", 15, "right"),
        ("avg step (s)", 12, "right"),
    ]

    ordered = sorted(rows, key=lambda row: row.sort_key)
    header = "  ".join(_format_table_cell(title, width, align=align) for title, width, align in columns)
    rule = "  ".join("-" * width for _, width, _ in columns)

    print("\n--- Qwen streaming memory ---")
    print(f"model: {ordered[0].model_name}")
    print(header)
    print(rule)
    for row in ordered:
        values = [
            f"{row.seq_len:d}",
            f"{row.measured_cuda_peak_gib:.3f}",
            f"{row.measured_cpu_peak_gib:.3f}",
            f"{row.avg_step_seconds:.3f}",
        ]
        print(
            "  ".join(
                _format_table_cell(value, width, align=align)
                for value, (_, width, align) in zip(values, columns)
            )
        )
    print("--- end Qwen table ---\n")


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


def _qwen_memory_worker(
    rank: int,
    world_size: int,
    init_file: str,
    result_file: str,
    model_path: str,
    seq_len: int,
) -> None:
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, init_method=f"file://{init_file}")
    try:
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        num_layers, offloaded_total_bytes, largest_layer_bytes = offloaded_layer_bytes(model)
        resident_bytes = resident_layer_bytes(model)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        engine = apply_cpu_streaming_(
            model,
            STREAM_MODULE_PATH,
            offload_policy=True,
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs={"lr": 1e-5, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.01, "foreach": False},
            max_grad_norm=1.0,
            device=device,
            auto_init_process_group=False,
            wrap_ddp=True,
        )
        assert isinstance(engine.model, DDP)

        vocab_size = model.config.vocab_size
        iteration_records: List[Dict[str, float]] = []
        step_seconds: List[float] = []
        tracker = IterationMemoryTracker(device)

        for _ in range(TRAIN_STEPS):
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

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            step_seconds.append(time.perf_counter() - step_start)
            iteration_records.append(
                {
                    "cpu_rss_peak_bytes": tracker.cpu_rss_peak,
                    "cuda_peak_bytes": tracker.cuda_peak,
                    "step_seconds": step_seconds[-1],
                }
            )

        engine.close(device=torch.device("cpu"))
        if rank == 0:
            payload = {
                "model_name": Path(model_path).name,
                "seq_len": seq_len,
                "num_layers": num_layers,
                "parameter_count": parameter_count,
                "offloaded_total_bytes": offloaded_total_bytes,
                "largest_layer_bytes": largest_layer_bytes,
                "resident_bytes": resident_bytes,
                "iteration_records": iteration_records,
                "step_seconds": step_seconds,
            }
            with open(result_file, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope="session")
def qwen_memory_rows() -> List[QwenMemoryRow]:
    rows: List[QwenMemoryRow] = []
    yield rows
    print_qwen_memory_table(rows)


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.parametrize(
    "seq_len",
    SEQUENCE_LENGTHS,
    ids=[f"seq{value}" for value in SEQUENCE_LENGTHS],
)
def test_qwen14b_streaming_peak_memory(
    seq_len: int,
    qwen_memory_rows: List[QwenMemoryRow],
) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs at least two CUDA devices")
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")

    model_path = resolve_qwen_14b_model_path()
    if model_path is None:
        pytest.skip("Qwen2.5-Coder-14B model not found under ~/Models or /mnt/ssd/arjun/models")

    world_size = 2
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init")
        result_file = os.path.join(tmpdir, "memory.json")
        if os.path.exists(init_file):
            os.unlink(init_file)

        mp.start_processes(
            _qwen_memory_worker,
            args=(
                world_size,
                init_file,
                result_file,
                str(model_path),
                seq_len,
            ),
            nprocs=world_size,
            start_method="spawn",
            join=True,
        )

        with open(result_file, encoding="utf-8") as handle:
            payload = json.load(handle)

    records = payload["iteration_records"]
    assert len(records) == TRAIN_STEPS

    offloaded_total_bytes = int(payload["offloaded_total_bytes"])
    largest_layer_bytes = int(payload["largest_layer_bytes"])
    resident_bytes = int(payload["resident_bytes"])
    parameter_count = int(payload["parameter_count"])
    assert parameter_count >= 13_000_000_000

    cpu_peak = max(int(record["cpu_rss_peak_bytes"]) for record in records)
    cuda_peak = max(int(record["cuda_peak_bytes"]) for record in records)
    step_seconds = [float(value) for value in payload["step_seconds"]]
    assert len(step_seconds) == TRAIN_STEPS
    avg_step_seconds = sum(step_seconds[1:]) / len(step_seconds[1:])

    qwen_memory_rows.append(
        QwenMemoryRow(
            model_name=payload["model_name"],
            seq_len=seq_len,
            num_layers=int(payload["num_layers"]),
            parameter_count=parameter_count,
            offloaded_total_gib=gib(offloaded_total_bytes),
            largest_layer_gib=gib(largest_layer_bytes),
            resident_gib=gib(resident_bytes),
            measured_cpu_peak_gib=gib(cpu_peak),
            measured_cuda_peak_gib=gib(cuda_peak),
            avg_step_seconds=avg_step_seconds,
        )
    )

    assert cpu_peak > 0
    assert cuda_peak > 0
    assert avg_step_seconds > 0
