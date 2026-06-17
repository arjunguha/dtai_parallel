from __future__ import annotations

import argparse
import json
import os
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM
from tqdm.auto import tqdm

from dtai_parallel import apply_cpu_streaming_
from tests.support.torchrun import run_torchrun

BATCH_SIZE = 1
TRAIN_STEPS = 3
STREAM_MODULE_PATH = "model.layers"
QWEN_MODEL_NAME = "Qwen/Qwen2.5-Coder-14B-Instruct"
QWEN_MODEL_DIRNAMES = ("qwen2p5_coder_14b_instruct", "Qwen2.5-Coder-14B-Instruct")
SEQUENCE_LENGTHS = [1000, 2000, 4000, 8000]
VISTA_HOST_SUFFIX = ".vista.tacc.utexas.edu"


@dataclass(frozen=True)
class QwenMemoryRow:
    """Summarized result for one Qwen streaming memory benchmark case."""
    mode: str
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
    def sort_key(self) -> Tuple[str, int]:
        """Sort Qwen benchmark rows by mode and sequence length."""
        return (self.mode, self.seq_len)


class IterationMemoryTracker:
    """Track per-iteration CPU and CUDA memory peaks for Qwen workers."""

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


def running_on_vista() -> bool:
    """Detect the cluster where the one-GPU Qwen cache assumptions apply."""
    return socket.gethostname().endswith(VISTA_HOST_SUFFIX)


def resolve_qwen_14b_model_path() -> Path | None:
    """Find a local Qwen2.5-Coder-14B checkout used by skip-friendly tests."""
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
    """Count parameter storage for model-footprint reporting."""
    return sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())


def gib(bytes_value: int | float) -> float:
    """Convert a byte count to gibibytes for reporting."""
    return float(bytes_value) / (1024**3)


def offloaded_layer_bytes(model: nn.Module, resident_suffix_count: int) -> Tuple[int, int, int]:
    """Compute Qwen decoder-layer bytes assigned to the streaming path."""
    layers = model.get_submodule(STREAM_MODULE_PATH)
    layer_bytes = [parameter_bytes(layer) for layer in layers]
    offloaded = layer_bytes[:-resident_suffix_count] if resident_suffix_count else layer_bytes
    return len(layers), sum(offloaded), max(offloaded) if offloaded else 0


def resident_layer_bytes(model: nn.Module, resident_suffix_count: int) -> int:
    """Compute Qwen parameter bytes left resident outside offloaded layers."""
    layers = list(model.get_submodule(STREAM_MODULE_PATH))
    offloaded_layers = layers[:-resident_suffix_count] if resident_suffix_count else layers
    offloaded_param_ids = {id(parameter) for layer in offloaded_layers for parameter in layer.parameters()}
    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
        if id(parameter) not in offloaded_param_ids
    )


def make_batch(device: torch.device, vocab_size: int, seq_len: int) -> Tuple[Tensor, Tensor]:
    """Create deterministic token and label tensors for one Qwen step."""
    generator = torch.Generator().manual_seed(20260609 + seq_len)
    input_ids = torch.randint(0, vocab_size, (BATCH_SIZE, seq_len), generator=generator, dtype=torch.long).to(device)
    return input_ids, input_ids.clone()


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
    model_ref: str,
    seq_len: int,
    resident_suffix_count: int,
    result_file: Path,
) -> None:
    """Run one Qwen benchmark case inside a torchrun worker process."""
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

        model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=not Path(model_ref).exists(),
        )
        num_layers, offloaded_total_bytes, largest_layer_bytes = offloaded_layer_bytes(model, resident_suffix_count)
        resident_bytes = resident_layer_bytes(model, resident_suffix_count)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        engine = apply_cpu_streaming_(
            model,
            STREAM_MODULE_PATH,
            offload_policy=True,
            resident_suffix_count=resident_suffix_count,
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs={"lr": 1e-5, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.01, "foreach": False},
            max_grad_norm=1.0,
            device=device,
            close_rank=0,
        )
        assert isinstance(engine.model, DDP)

        tracker = IterationMemoryTracker(device, distributed=distributed)
        records: List[Dict[str, float]] = []
        step_seconds: List[float] = []
        vocab_size = model.config.vocab_size
        for _ in range(TRAIN_STEPS):
            torch.cuda.synchronize(device)
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
            torch.cuda.synchronize(device)
            step_seconds.append(time.perf_counter() - step_start)
            records.append(
                {
                    "cpu_peak_bytes": tracker.cpu_peak,
                    "cuda_peak_bytes": tracker.cuda_peak,
                    "step_seconds": step_seconds[-1],
                }
            )

        engine.close(device=torch.device("cpu"))
        torch.cuda.empty_cache()
        if _rank() == 0:
            payload = {
                "mode": mode,
                "model_name": Path(model_ref).name if Path(model_ref).exists() else model_ref,
                "seq_len": seq_len,
                "num_layers": num_layers,
                "resident_suffix_count": resident_suffix_count,
                "parameter_count": parameter_count,
                "offloaded_total_bytes": offloaded_total_bytes,
                "largest_layer_bytes": largest_layer_bytes,
                "resident_bytes": resident_bytes,
                "iteration_records": records,
                "step_seconds": step_seconds,
            }
            result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def row_from_payload(payload: Dict[str, object]) -> QwenMemoryRow:
    """Convert a worker JSON payload into a Qwen report row."""
    records = list(payload["iteration_records"])  # type: ignore[index]
    step_seconds = tuple(float(value) for value in payload["step_seconds"])  # type: ignore[index]
    warm_steps = step_seconds[1:] or step_seconds
    return QwenMemoryRow(
        mode=str(payload["mode"]),
        model_name=str(payload["model_name"]),
        seq_len=int(payload["seq_len"]),
        num_layers=int(payload["num_layers"]),
        resident_suffix_count=int(payload["resident_suffix_count"]),
        parameter_count=int(payload["parameter_count"]),
        offloaded_total_gib=gib(float(payload["offloaded_total_bytes"])),
        largest_layer_gib=gib(float(payload["largest_layer_bytes"])),
        resident_gib=gib(float(payload["resident_bytes"])),
        measured_cpu_peak_gib=gib(max(int(record["cpu_peak_bytes"]) for record in records)),  # type: ignore[index]
        measured_cuda_peak_gib=gib(max(int(record["cuda_peak_bytes"]) for record in records)),  # type: ignore[index]
        step_seconds=step_seconds,
        average_step_seconds=sum(warm_steps) / len(warm_steps),
    )


def assert_qwen_invariants(row: QwenMemoryRow) -> None:
    """Assert the Qwen run exercised a real model and recorded nonzero metrics."""
    assert row.parameter_count >= 13_000_000_000
    assert row.measured_cpu_peak_gib > 0
    assert row.measured_cuda_peak_gib > 0
    assert row.average_step_seconds > 0


def run_case_with_torchrun(
    *,
    mode: str,
    model_ref: str,
    seq_len: int,
    output_dir: Path,
    resident_suffix_count: int = 0,
) -> QwenMemoryRow:
    """Launch one Qwen case through torchrun and load its result row."""
    nproc = 2 if mode == "two-gpu" else 1
    result_file = output_dir / f"qwen-{mode}-{seq_len}.json"
    run_torchrun(
        "benchmarks.qwen_memory",
        [
            "worker",
            "--mode",
            mode,
            "--model-ref",
            model_ref,
            "--seq-len",
            str(seq_len),
            "--resident-suffix-count",
            str(resident_suffix_count),
            "--result-file",
            str(result_file),
        ],
        nproc_per_node=nproc,
    )
    return row_from_payload(json.loads(result_file.read_text(encoding="utf-8")))


def _format_table_cell(text: str, width: int, *, align: str = "right") -> str:
    """Format one fixed-width table cell for console output."""
    return text.ljust(width) if align == "left" else text.rjust(width)


def print_table(rows: Iterable[QwenMemoryRow]) -> None:
    """Print the Qwen benchmark summary table for human inspection."""
    ordered = sorted(rows, key=lambda row: row.sort_key)
    if not ordered:
        return
    columns = [
        ("mode", 8, "left"),
        ("seq len", 8, "right"),
        ("layers", 6, "right"),
        ("resident", 8, "right"),
        ("params B", 8, "right"),
        ("CPU GiB", 10, "right"),
        ("CUDA GiB", 10, "right"),
        ("avg step", 9, "right"),
    ]
    print("\n--- Qwen streaming memory benchmark ---")
    print(f"model: {ordered[0].model_name}")
    print("  ".join(_format_table_cell(title, width, align=align) for title, width, align in columns))
    print("  ".join("-" * width for _, width, _ in columns))
    for row in ordered:
        values = [
            row.mode,
            f"{row.seq_len}",
            f"{row.num_layers}",
            f"{row.resident_suffix_count}",
            f"{row.parameter_count / 1e9:.3f}",
            f"{row.measured_cpu_peak_gib:.3f}",
            f"{row.measured_cuda_peak_gib:.3f}",
            f"{row.average_step_seconds:.3f}",
        ]
        print("  ".join(_format_table_cell(value, width, align=align) for value, (_, width, align) in zip(values, columns)))
    print("--- end Qwen benchmark ---\n")


def run_benchmark(args: argparse.Namespace) -> None:
    """Run selected Qwen benchmark cases and write summaries."""
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    model_ref = args.model_ref
    if model_ref is None:
        resolved = resolve_qwen_14b_model_path()
        model_ref = str(resolved) if resolved is not None else QWEN_MODEL_NAME
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cases = [(mode, seq_len) for mode in args.mode for seq_len in args.seq_len]
    for mode, seq_len in tqdm(cases, desc="Qwen memory benchmarks", unit="case"):
        row = run_case_with_torchrun(
            mode=mode,
            model_ref=model_ref,
            seq_len=seq_len,
            output_dir=output_dir,
            resident_suffix_count=args.resident_suffix_count,
        )
        assert_qwen_invariants(row)
        rows.append(row)
    (output_dir / "summary.json").write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")
    print_table(rows)


def main() -> None:
    """Dispatch the Qwen streaming memory benchmark CLI."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--mode", choices=["one-gpu", "two-gpu"], required=True)
    worker.add_argument("--model-ref", required=True)
    worker.add_argument("--seq-len", type=int, required=True)
    worker.add_argument("--resident-suffix-count", type=int, default=0)
    worker.add_argument("--result-file", type=Path, required=True)

    bench = subparsers.add_parser("run")
    bench.add_argument("--gpus")
    bench.add_argument("--mode", action="append", choices=["one-gpu", "two-gpu"], default=None)
    bench.add_argument("--model-ref")
    bench.add_argument("--seq-len", action="append", type=int, default=None)
    bench.add_argument("--resident-suffix-count", type=int, default=int(os.environ.get("DTAI_QWEN_RESIDENT_SUFFIX_COUNT", "0")))
    bench.add_argument("--output-dir", default="benchmark-results/qwen-memory")

    args = parser.parse_args()
    if args.command == "worker":
        run_worker_case(
            mode=args.mode,
            model_ref=args.model_ref,
            seq_len=args.seq_len,
            resident_suffix_count=args.resident_suffix_count,
            result_file=args.result_file,
        )
    else:
        if args.mode is None:
            args.mode = ["two-gpu"]
        if args.seq_len is None:
            args.seq_len = SEQUENCE_LENGTHS
        run_benchmark(args)


if __name__ == "__main__":
    main()
