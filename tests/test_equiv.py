import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")

import pytest
import torch

from tests.support.training import assert_state_dicts_close, assert_state_dicts_equal
from tests.support.torchrun import run_torchrun


def _load_state(path: str):
    """Load a rank-zero state dict written by a torchrun worker."""
    return torch.load(path, map_location="cpu")


def _run_state(case: str, result_file: str, *, nproc_per_node: int, extra_args: list[str] | None = None) -> None:
    """Run one worker case that writes a state dict for pytest-side comparison."""
    args = [case, "--result-file", result_file]
    if extra_args:
        args.extend(extra_args)
    run_torchrun("tests.torchrun_streaming_worker", args, nproc_per_node=nproc_per_node)


@pytest.mark.parametrize("container_kind", ["modulelist", "sequential"])
@pytest.mark.parametrize("optimizer_name", ["adamw", "sgd"])
@pytest.mark.parametrize("max_grad_norm", [None, 0.35])
def test_single_gpu_streaming_matches_single_process_reference(
    container_kind: str,
    optimizer_name: str,
    max_grad_norm: Optional[float],
) -> None:
    """Compare one-GPU streamed training against ordinary one-GPU training."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        pytest.skip("needs at least one CUDA device")

    extra_args = ["--container-kind", container_kind, "--optimizer-name", optimizer_name]
    if max_grad_norm is not None:
        extra_args.extend(["--max-grad-norm", str(max_grad_norm)])

    with tempfile.TemporaryDirectory() as tmpdir:
        ref_file = os.path.join(tmpdir, "ref.pt")
        stream_file = os.path.join(tmpdir, "stream.pt")
        _run_state("single-gpu-reference-state", ref_file, nproc_per_node=1, extra_args=extra_args)
        _run_state("single-gpu-streaming-state", stream_file, nproc_per_node=1, extra_args=extra_args)

        ref_state = _load_state(ref_file)
        stream_state = _load_state(stream_file)

    assert_state_dicts_close(stream_state, ref_state, dtype=torch.float32)


def test_single_gpu_resident_suffix_matches_single_process_reference() -> None:
    """Compare resident-suffix streamed training against ordinary one-GPU training."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        pytest.skip("needs at least one CUDA device")

    with tempfile.TemporaryDirectory() as tmpdir:
        ref_file = os.path.join(tmpdir, "ref.pt")
        stream_file = os.path.join(tmpdir, "stream.pt")
        _run_state(
            "single-gpu-reference-state",
            ref_file,
            nproc_per_node=1,
            extra_args=["--container-kind", "modulelist", "--optimizer-name", "adamw"],
        )
        _run_state(
            "single-gpu-streaming-state",
            stream_file,
            nproc_per_node=1,
            extra_args=[
                "--container-kind",
                "modulelist",
                "--optimizer-name",
                "adamw",
                "--resident-suffix-count",
                "2",
            ],
        )

        ref_state = _load_state(ref_file)
        stream_state = _load_state(stream_file)

    assert_state_dicts_equal(stream_state, ref_state)


def test_single_gpu_gradient_accumulation_matches_single_process_reference() -> None:
    """Compare streamed gradient accumulation against ordinary one-GPU training."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        pytest.skip("needs at least one CUDA device")

    with tempfile.TemporaryDirectory() as tmpdir:
        ref_file = os.path.join(tmpdir, "ref.pt")
        stream_file = os.path.join(tmpdir, "stream.pt")
        _run_state("gradient-accumulation-reference-state", ref_file, nproc_per_node=1)
        _run_state("gradient-accumulation-streaming-state", stream_file, nproc_per_node=1)

        ref_state = _load_state(ref_file)
        stream_state = _load_state(stream_file)

    assert_state_dicts_close(stream_state, ref_state, dtype=torch.float32)


def test_kwarg_streaming_matches_single_process_reference() -> None:
    """Compare streamed kwargs and nested outputs against ordinary one-GPU training."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        pytest.skip("needs at least one CUDA device")

    with tempfile.TemporaryDirectory() as tmpdir:
        ref_file = os.path.join(tmpdir, "ref.pt")
        stream_file = os.path.join(tmpdir, "stream.pt")
        _run_state("kwarg-reference-state", ref_file, nproc_per_node=1)
        _run_state("kwarg-streaming-state", stream_file, nproc_per_node=1)

        ref_state = _load_state(ref_file)
        stream_state = _load_state(stream_file)

    assert_state_dicts_close(stream_state, ref_state, dtype=torch.float32)


def test_streamed_modulelist_inside_larger_model_matches_standard_training_loop() -> None:
    """Compare two-rank streamed training against an equivalent one-rank CUDA batch."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs at least two CUDA devices")

    with tempfile.TemporaryDirectory() as tmpdir:
        ref_file = os.path.join(tmpdir, "ref.pt")
        stream_file = os.path.join(tmpdir, "stream.pt")
        _run_state("standard-state", ref_file, nproc_per_node=1)
        _run_state("streaming-ddp-state", stream_file, nproc_per_node=2)

        ref_state = _load_state(ref_file)
        stream_state = _load_state(stream_file)

    assert_state_dicts_close(stream_state, ref_state, dtype=torch.float32)


def test_two_rank_gradient_accumulation_matches_standard_training_loop() -> None:
    """Compare two-rank streamed gradient accumulation against full-batch CUDA."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs at least two CUDA devices")

    with tempfile.TemporaryDirectory() as tmpdir:
        ref_file = os.path.join(tmpdir, "ref.pt")
        stream_file = os.path.join(tmpdir, "stream.pt")
        _run_state("standard-accumulation-state", ref_file, nproc_per_node=1)
        _run_state("streaming-ddp-accumulation-state", stream_file, nproc_per_node=2)

        ref_state = _load_state(ref_file)
        stream_state = _load_state(stream_file)

    assert_state_dicts_close(stream_state, ref_state, dtype=torch.float32)
