import os
import sys
from pathlib import Path

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

from tests.support.torchrun import run_torchrun


@pytest.mark.cuda
def test_cuda_transfer_timing_records_streamed_copies() -> None:
    """Verify transfer metrics record CUDA streaming copies through torchrun."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        pytest.skip("needs CUDA")

    import tempfile

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

    import tempfile

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
