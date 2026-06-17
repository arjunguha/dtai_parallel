import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Torchrun-backed CUDA tests launch multiple Python processes.  Keeping math to
# one thread per process prevents small CI machines from oversubscribing cores.
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

from tests.support.training import assert_state_dicts_close
from tests.support.torchrun import run_torchrun


@pytest.mark.parametrize("container_kind", ["modulelist", "sequential"])
@pytest.mark.parametrize("optimizer_name", ["adamw", "sgd"])
@pytest.mark.parametrize("max_grad_norm", [None, 0.35])
def test_in_place_transform_matches_single_process_reference(container_kind: str, optimizer_name: str, max_grad_norm: Optional[float]) -> None:
    """Compare streamed and ordinary single-process training on one CUDA rank."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        pytest.skip("needs at least one CUDA device")

    with tempfile.TemporaryDirectory() as tmpdir:
        result_file = os.path.join(tmpdir, "result.pt")
        args = [
            "single-gpu-equivalence",
            "--result-file",
            result_file,
            "--container-kind",
            container_kind,
            "--optimizer-name",
            optimizer_name,
        ]
        if max_grad_norm is not None:
            args.extend(["--max-grad-norm", str(max_grad_norm)])
        run_torchrun("tests.torchrun_streaming_worker", args, nproc_per_node=1)
        assert torch.load(result_file, map_location="cpu")["ok"]


@pytest.mark.parametrize("pin_cpu_masters", ["eager", "lazy", False])
def test_one_gpu_cuda_streaming_matches_single_process_reference(pin_cpu_masters) -> None:
    """Check pinning modes through the one-GPU streaming equivalence path."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        pytest.skip("needs at least one CUDA device")

    pin_arg = "false" if pin_cpu_masters is False else str(pin_cpu_masters)
    with tempfile.TemporaryDirectory() as tmpdir:
        result_file = os.path.join(tmpdir, "result.pt")
        run_torchrun(
            "tests.torchrun_streaming_worker",
            ["single-gpu-reference", "--result-file", result_file, "--pin-cpu-masters", pin_arg],
            nproc_per_node=1,
        )
        assert torch.load(result_file, map_location="cpu")["ok"]


def test_one_gpu_cuda_resident_suffix_matches_single_process_reference() -> None:
    """Check resident suffix handling through a one-GPU torchrun worker."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        pytest.skip("needs at least one CUDA device")

    with tempfile.TemporaryDirectory() as tmpdir:
        result_file = os.path.join(tmpdir, "result.pt")
        run_torchrun(
            "tests.torchrun_streaming_worker",
            ["single-gpu-resident-suffix", "--result-file", result_file],
            nproc_per_node=1,
        )
        assert torch.load(result_file, map_location="cpu")["ok"]


def test_cuda_api_behaviors() -> None:
    """Exercise transformation API, kwargs replay, and prefetch scheduling on CUDA."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        pytest.skip("needs at least one CUDA device")

    with tempfile.TemporaryDirectory() as tmpdir:
        result_file = os.path.join(tmpdir, "result.pt")
        run_torchrun(
            "tests.torchrun_streaming_worker",
            ["api-behaviors", "--result-file", result_file],
            nproc_per_node=1,
        )
        assert torch.load(result_file, map_location="cpu")["ok"]


@pytest.mark.cuda
def test_cuda_transfer_timing_records_streamed_copies() -> None:
    """Verify transfer metrics record CUDA streaming copies through torchrun."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        pytest.skip("needs CUDA")

    with tempfile.TemporaryDirectory() as tmpdir:
        result_file = os.path.join(tmpdir, "result.pt")
        run_torchrun(
            "tests.torchrun_streaming_worker",
            ["transfer-timing", "--result-file", result_file],
            nproc_per_node=1,
        )
        assert torch.load(result_file, map_location="cpu")["ok"]


def test_distributed_offloaded_parameters_use_shared_cpu_storage() -> None:
    """Verify CUDA torchrun ranks use shared /dev/shm CPU storage for offloaded state."""
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    if not os.path.isdir("/dev/shm"):
        pytest.skip("/dev/shm is unavailable")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs at least two CUDA devices")

    with tempfile.TemporaryDirectory() as tmpdir:
        result_file = os.path.join(tmpdir, "shared_result.pt")
        run_torchrun(
            "tests.torchrun_streaming_worker",
            ["shared-cpu-storage", "--result-file", result_file],
            nproc_per_node=2,
        )
        results = [torch.load(f"{result_file}.{rank}", map_location="cpu") for rank in range(2)]

    assert {result["rank"] for result in results} == {0, 1}
    assert all(result["observed"] == 123.0 for result in results)
    assert all(result["is_shm"] for result in results)
    assert all(not result["gradient_files_remaining"] for result in results)
    assert all(not result["local_gradient_files_remaining"] for result in results)


def test_streamed_modulelist_inside_larger_model_matches_real_ddp() -> None:
    """Compare streamed DDP training against real DDP using two CUDA ranks."""
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs at least two CUDA devices")

    with tempfile.TemporaryDirectory() as tmpdir:
        result_ref = os.path.join(tmpdir, "ref.pt")
        result_stream = os.path.join(tmpdir, "stream.pt")
        run_torchrun(
            "tests.torchrun_streaming_worker",
            ["reference-ddp-state", "--result-file", result_ref],
            nproc_per_node=2,
        )
        run_torchrun(
            "tests.torchrun_streaming_worker",
            ["streaming-ddp-state", "--result-file", result_stream],
            nproc_per_node=2,
        )
        ref_state = torch.load(result_ref, map_location="cpu")
        stream_state = torch.load(result_stream, map_location="cpu")

    assert_state_dicts_close(stream_state, ref_state, dtype=torch.float32)


def test_streamed_modulelist_inside_larger_model_matches_standard_training_loop() -> None:
    """Compare two-rank streamed training against an equivalent one-rank CUDA batch."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs at least two CUDA devices")

    with tempfile.TemporaryDirectory() as tmpdir:
        result_ref = os.path.join(tmpdir, "ref.pt")
        result_stream = os.path.join(tmpdir, "stream.pt")
        run_torchrun(
            "tests.torchrun_streaming_worker",
            ["standard-state", "--result-file", result_ref],
            nproc_per_node=1,
        )
        run_torchrun(
            "tests.torchrun_streaming_worker",
            ["streaming-ddp-state", "--result-file", result_stream],
            nproc_per_node=2,
        )
        ref_state = torch.load(result_ref, map_location="cpu")
        stream_state = torch.load(result_stream, map_location="cpu")

    assert_state_dicts_close(stream_state, ref_state, dtype=torch.float32)
