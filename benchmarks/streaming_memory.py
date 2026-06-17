from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from dtai_parallel import apply_cpu_streaming_
from tests.models import LargeStreamingSandwich, STREAM_INPUT_DIM, STREAM_OUTPUT_DIM
from tests.support.torchrun import run_torchrun

INPUT_DIM = STREAM_INPUT_DIM
OUTPUT_DIM = STREAM_OUTPUT_DIM
BATCH_SIZE = 2
TRAIN_STEPS = 3
ADAMW_FACTOR = 4
REFERENCE_LAYER_WIDTH = 7936
REFERENCE_NUM_LAYERS = 4
TARGET_OFFLOADED_PARAMS = REFERENCE_NUM_LAYERS * 2 * (
    REFERENCE_LAYER_WIDTH * REFERENCE_LAYER_WIDTH + REFERENCE_LAYER_WIDTH
)

BENCHMARK_CASES = [(4, 1.0), (8, 1.0), (16, 1.0), (8, 0.5), (16, 0.5)]
TEST_CASES = [(2, 0.002), (4, 0.002)]


@dataclass(frozen=True)
class MemoryRow:
    """Summarized result for one synthetic streaming memory benchmark case."""
    mode: str
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
    average_step_seconds: float

    @property
    def scale_label(self) -> str:
        """Return a compact label for the layer-size scale."""
        return "full" if self.layer_param_scale == 1.0 else f"{self.layer_param_scale:g}x"

    @property
    def cuda_ratio(self) -> float:
        """Return measured CUDA peak as a multiple of the expected lower bound."""
        return self.measured_cuda_peak_gib / self.expected_cuda_peak_gib

    @property
    def sort_key(self) -> Tuple[str, int, float]:
        """Sort benchmark rows by mode and configuration."""
        return (self.mode, self.num_layers, -self.layer_param_scale)


class IterationMemoryTracker:
    """Track per-iteration CPU and CUDA memory peaks while training runs."""

    def __init__(self, device: torch.device, *, distributed: bool) -> None:
        self.device = device
        self.distributed = distributed
        self.cpu_peak = 0
        self.cuda_peak = 0

    def begin_iteration(self) -> None:
        """Reset CUDA peak accounting at the start of a measured step."""
        torch.cuda.reset_peak_memory_stats(self.device)

    def sample(self) -> None:
        """Update remembered memory peaks at a measurement point."""
        self.cpu_peak = max(self.cpu_peak, read_cpu_memory_bytes(self.distributed))
        self.cuda_peak = max(self.cuda_peak, int(torch.cuda.max_memory_allocated(self.device)))


def layer_width_for_num_layers(num_layers: int) -> int:
    """Choose a width that keeps the full-scale benchmark near the reference size."""
    target_per_layer = TARGET_OFFLOADED_PARAMS // num_layers
    return int((math.sqrt(1 + 4 * (target_per_layer // 2)) - 1) / 2)


def layer_width_for_config(num_layers: int, layer_param_scale: float) -> int:
    """Choose the layer width for a benchmark layer-count and scale pair."""
    base_width = layer_width_for_num_layers(num_layers)
    if layer_param_scale == 1.0:
        return base_width
    return max(1, int(base_width * math.sqrt(layer_param_scale)))


def read_cpu_memory_bytes(prefer_pss: bool) -> int:
    """Read process memory with PSS when distributed shared pages matter."""
    if prefer_pss:
        try:
            with open("/proc/self/smaps_rollup", encoding="ascii") as handle:
                for line in handle:
                    if line.startswith("Pss:"):
                        return int(line.split()[1]) * 1024
        except OSError:
            pass
    with open("/proc/self/status", encoding="ascii") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("neither Pss nor VmRSS memory accounting is available")


def parameter_bytes(module: nn.Module) -> int:
    """Count parameter storage for model-footprint expectations."""
    return sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())


def gib(bytes_value: int | float) -> float:
    """Convert a byte count to gibibytes for reporting."""
    return float(bytes_value) / (1024**3)


def offloaded_layer_bytes(model: LargeStreamingSandwich) -> Tuple[int, int]:
    """Compute total and largest offloaded layer byte counts."""
    layer_bytes = [parameter_bytes(layer) for layer in model.layers]
    return sum(layer_bytes), max(layer_bytes)


def resident_layer_bytes(model: LargeStreamingSandwich) -> int:
    """Compute bytes for modules that remain resident on device."""
    return parameter_bytes(model.embed) + parameter_bytes(model.norm) + parameter_bytes(model.unembed)


def make_batch(device: torch.device) -> Tuple[Tensor, Tensor]:
    """Create a deterministic synthetic regression batch on the worker device."""
    generator = torch.Generator().manual_seed(20260609)
    x = torch.randn(BATCH_SIZE, INPUT_DIM, generator=generator, dtype=torch.float32).to(device)
    y = torch.randn(BATCH_SIZE, OUTPUT_DIM, generator=generator, dtype=torch.float32).to(device)
    return x, y


def _local_rank() -> int:
    """Read the torchrun local rank for device selection."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def _rank() -> int:
    """Read the torchrun global rank for rank-zero output."""
    return int(os.environ.get("RANK", "0"))


def _world_size() -> int:
    """Read the torchrun world size for distributed setup."""
    return int(os.environ.get("WORLD_SIZE", "1"))


def run_worker_case(
    *,
    mode: str,
    num_layers: int,
    layer_width: int,
    result_file: Path,
) -> None:
    """Run one benchmark case inside the torchrun worker process."""
    world_size = _world_size()
    distributed = world_size > 1
    device = torch.device("cuda", _local_rank())
    torch.cuda.set_device(device)
    try:
        dist.init_process_group(backend="nccl", device_id=device)
    except TypeError:
        dist.init_process_group(backend="nccl")
    try:
        torch.cuda.empty_cache()

        model = LargeStreamingSandwich(num_layers, layer_width)
        offloaded_total_bytes, largest_layer_bytes = offloaded_layer_bytes(model)
        resident_bytes = resident_layer_bytes(model)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        expected_cpu_peak = ADAMW_FACTOR * offloaded_total_bytes / float(world_size)
        expected_cuda_peak = ADAMW_FACTOR * (largest_layer_bytes + resident_bytes)

        engine = apply_cpu_streaming_(
            model,
            "layers",
            offload_policy=True,
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs={"lr": 3e-4, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.01, "foreach": False},
            max_grad_norm=1.0,
            device=device,
            collect_timing=True,
        )
        assert isinstance(engine.model, DDP)

        criterion = nn.MSELoss()
        tracker = IterationMemoryTracker(device, distributed=distributed)
        records: List[Dict[str, object]] = []

        for _ in range(TRAIN_STEPS):
            torch.cuda.synchronize(device)
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
            timing = engine.transfer_timing_summary(reset=True, synchronize=True)
            torch.cuda.synchronize(device)
            records.append(
                {
                    "cpu_peak_bytes": tracker.cpu_peak,
                    "cuda_peak_bytes": tracker.cuda_peak,
                    "transfer_timing": timing,
                    "step_seconds": time.perf_counter() - step_start,
                }
            )

        closed = engine.close(return_on_all_ranks=True, device=torch.device("cpu"))
        assert closed is not None
        torch.cuda.empty_cache()

        if _rank() == 0:
            payload = {
                "mode": mode,
                "num_layers": num_layers,
                "layer_width": layer_width,
                "parameter_count": parameter_count,
                "offloaded_total_bytes": offloaded_total_bytes,
                "largest_layer_bytes": largest_layer_bytes,
                "resident_bytes": resident_bytes,
                "expected_cpu_peak_bytes": expected_cpu_peak,
                "expected_cuda_peak_bytes": expected_cuda_peak,
                "iteration_records": records,
            }
            result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def row_from_payload(payload: Dict[str, object], layer_param_scale: float) -> MemoryRow:
    """Convert a worker JSON payload into a report row."""
    records = list(payload["iteration_records"])  # type: ignore[index]
    cpu_peak = max(int(record["cpu_peak_bytes"]) for record in records)  # type: ignore[index]
    cuda_peak = max(int(record["cuda_peak_bytes"]) for record in records)  # type: ignore[index]
    transfer_h2d_bytes = sum(
        int(timing["bytes"])
        for record in records  # type: ignore[assignment]
        for kind, timing in record["transfer_timing"].items()
        if kind.endswith("_h2d")
    )
    transfer_d2h_bytes = sum(
        int(timing["bytes"])
        for record in records  # type: ignore[assignment]
        for kind, timing in record["transfer_timing"].items()
        if kind.endswith("_d2h")
    )
    transfer_cuda_ms = sum(
        float(timing["cuda_ms"])
        for record in records  # type: ignore[assignment]
        for timing in record["transfer_timing"].values()
    )
    step_seconds = [float(record["step_seconds"]) for record in records]  # type: ignore[index]
    warm_steps = step_seconds[1:] or step_seconds
    return MemoryRow(
        mode=str(payload["mode"]),
        num_layers=int(payload["num_layers"]),
        layer_param_scale=layer_param_scale,
        layer_width=int(payload["layer_width"]),
        parameter_count=int(payload["parameter_count"]),
        offloaded_total_gib=gib(float(payload["offloaded_total_bytes"])),
        largest_layer_gib=gib(float(payload["largest_layer_bytes"])),
        resident_gib=gib(float(payload["resident_bytes"])),
        expected_cpu_peak_gib=gib(float(payload["expected_cpu_peak_bytes"])),
        measured_cpu_peak_gib=gib(cpu_peak),
        expected_cuda_peak_gib=gib(float(payload["expected_cuda_peak_bytes"])),
        measured_cuda_peak_gib=gib(cuda_peak),
        measured_transfer_h2d_gib=gib(transfer_h2d_bytes),
        measured_transfer_d2h_gib=gib(transfer_d2h_bytes),
        measured_transfer_cuda_ms=transfer_cuda_ms,
        average_step_seconds=sum(warm_steps) / len(warm_steps),
    )


def assert_memory_invariants(row: MemoryRow, *, require_cpu_lower_bound: bool = True) -> None:
    """Assert the benchmark still exercised CPU offload and CUDA streaming paths."""
    assert row.measured_transfer_h2d_gib > 0
    assert row.measured_transfer_d2h_gib > 0
    assert row.measured_cuda_peak_gib >= 0.85 * row.expected_cuda_peak_gib
    if require_cpu_lower_bound:
        assert row.measured_cpu_peak_gib >= 0.50 * row.expected_cpu_peak_gib
    assert row.average_step_seconds > 0


def run_case_with_torchrun(
    *,
    mode: str,
    num_layers: int,
    layer_param_scale: float,
    output_dir: Path,
) -> MemoryRow:
    """Launch one benchmark case through torchrun and load its result row."""
    layer_width = layer_width_for_config(num_layers, layer_param_scale)
    nproc = 2 if mode == "two-gpu" else 1
    result_file = output_dir / f"{mode}-{num_layers}-{layer_param_scale:g}.json"
    run_torchrun(
        "benchmarks.streaming_memory",
        [
            "worker",
            "--mode",
            mode,
            "--num-layers",
            str(num_layers),
            "--layer-width",
            str(layer_width),
            "--result-file",
            str(result_file),
        ],
        nproc_per_node=nproc,
    )
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    return row_from_payload(payload, layer_param_scale)


def _format_table_cell(text: str, width: int, *, align: str = "right") -> str:
    """Format one fixed-width table cell for console output."""
    return text.ljust(width) if align == "left" else text.rjust(width)


def print_table(rows: Iterable[MemoryRow]) -> None:
    """Print the benchmark summary table for human inspection."""
    ordered = sorted(rows, key=lambda row: row.sort_key)
    if not ordered:
        return
    columns = [
        ("mode", 8, "left"),
        ("layers", 6, "right"),
        ("scale", 8, "left"),
        ("width", 6, "right"),
        ("params B", 8, "right"),
        ("exp CUDA", 10, "right"),
        ("meas CUDA", 10, "right"),
        ("CUDA x", 8, "right"),
        ("H2D GiB", 9, "right"),
        ("D2H GiB", 9, "right"),
        ("copy ms", 9, "right"),
        ("avg step", 9, "right"),
        ("exp CPU", 10, "right"),
        ("meas CPU", 10, "right"),
    ]
    print("\n--- streaming memory benchmark ---")
    print("  ".join(_format_table_cell(title, width, align=align) for title, width, align in columns))
    print("  ".join("-" * width for _, width, _ in columns))
    for row in ordered:
        values = [
            row.mode,
            f"{row.num_layers}",
            row.scale_label,
            f"{row.layer_width}",
            f"{row.parameter_count / 1e9:.3f}",
            f"{row.expected_cuda_peak_gib:.3f}",
            f"{row.measured_cuda_peak_gib:.3f}",
            f"{row.cuda_ratio:.2f}",
            f"{row.measured_transfer_h2d_gib:.3f}",
            f"{row.measured_transfer_d2h_gib:.3f}",
            f"{row.measured_transfer_cuda_ms:.1f}",
            f"{row.average_step_seconds:.3f}",
            f"{row.expected_cpu_peak_gib:.3f}",
            f"{row.measured_cpu_peak_gib:.3f}",
        ]
        print("  ".join(_format_table_cell(value, width, align=align) for value, (_, width, align) in zip(values, columns)))
    print("--- end benchmark ---\n")


def parse_cases(values: List[str] | None) -> List[Tuple[int, float]]:
    """Parse CLI case selectors into benchmark layer-count and scale pairs."""
    if not values:
        return BENCHMARK_CASES
    cases = []
    for value in values:
        layers, scale = value.split(":", 1)
        cases.append((int(layers), float(scale)))
    return cases


def run_benchmark(args: argparse.Namespace) -> None:
    """Run selected benchmark cases and write machine-readable summaries."""
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in args.mode:
        for num_layers, layer_param_scale in parse_cases(args.case):
            row = run_case_with_torchrun(
                mode=mode,
                num_layers=num_layers,
                layer_param_scale=layer_param_scale,
                output_dir=output_dir,
            )
            assert_memory_invariants(row, require_cpu_lower_bound=mode == "two-gpu")
            rows.append(row)
    (output_dir / "summary.json").write_text(
        json.dumps([asdict(row) for row in rows], indent=2),
        encoding="utf-8",
    )
    print_table(rows)


def main() -> None:
    """Dispatch the synthetic streaming memory benchmark CLI."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--mode", choices=["one-gpu", "two-gpu"], required=True)
    worker.add_argument("--num-layers", type=int, required=True)
    worker.add_argument("--layer-width", type=int, required=True)
    worker.add_argument("--result-file", type=Path, required=True)

    bench = subparsers.add_parser("run")
    bench.add_argument("--gpus", help="Physical GPU list for CUDA_VISIBLE_DEVICES, for example 2,4")
    bench.add_argument("--mode", action="append", choices=["one-gpu", "two-gpu"], default=None)
    bench.add_argument("--case", action="append", help="Case as layers:scale, for example 8:0.5")
    bench.add_argument("--output-dir", default="benchmark-results/streaming-memory")

    args = parser.parse_args()
    if args.command == "worker":
        run_worker_case(
            mode=args.mode,
            num_layers=args.num_layers,
            layer_width=args.layer_width,
            result_file=args.result_file,
        )
    else:
        if args.mode is None:
            args.mode = ["two-gpu"]
        run_benchmark(args)


if __name__ == "__main__":
    main()
