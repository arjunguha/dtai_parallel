from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from benchmarks.qwen_memory import (
    QWEN_MODEL_NAME,
    assert_qwen_invariants,
    resolve_qwen_14b_model_path,
    run_case_with_torchrun as run_qwen_case_with_torchrun,
    running_on_vista,
)
from benchmarks.streaming_memory import (
    TEST_CASES,
    assert_memory_invariants,
    run_case_with_torchrun as run_streaming_case_with_torchrun,
)


@pytest.mark.cuda
@pytest.mark.parametrize("num_layers,layer_param_scale", TEST_CASES, ids=["2-tiny", "4-tiny"])
def test_two_gpu_streaming_memory_invariants(num_layers: int, layer_param_scale: float) -> None:
    """Check the fast two-GPU synthetic memory case keeps core benchmark invariants."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs at least two CUDA devices")

    with tempfile.TemporaryDirectory() as tmpdir:
        row = run_streaming_case_with_torchrun(
            mode="two-gpu",
            num_layers=num_layers,
            layer_param_scale=layer_param_scale,
            output_dir=Path(tmpdir),
        )

    assert row.parameter_count > 0
    assert_memory_invariants(row)


@pytest.mark.cuda
def test_one_gpu_streaming_memory_invariants() -> None:
    """Check the fast one-GPU synthetic memory case through a one-process torchrun launch."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        pytest.skip("needs at least one CUDA device")

    with tempfile.TemporaryDirectory() as tmpdir:
        row = run_streaming_case_with_torchrun(
            mode="one-gpu",
            num_layers=2,
            layer_param_scale=0.002,
            output_dir=Path(tmpdir),
        )

    assert row.parameter_count > 0
    assert_memory_invariants(row, require_cpu_lower_bound=False)


@pytest.mark.cuda
@pytest.mark.slow
def test_qwen14b_two_gpu_streaming_memory_invariants() -> None:
    """Run the smallest two-GPU Qwen memory invariant check when the model is local."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs at least two CUDA devices")

    model_path = resolve_qwen_14b_model_path()
    if model_path is None:
        pytest.skip("Qwen2.5-Coder-14B model not found under ~/Models or /mnt/ssd/arjun/models")

    with tempfile.TemporaryDirectory() as tmpdir:
        row = run_qwen_case_with_torchrun(
            mode="two-gpu",
            model_ref=str(model_path),
            seq_len=1000,
            output_dir=Path(tmpdir),
        )

    assert_qwen_invariants(row)


@pytest.mark.cuda
@pytest.mark.slow
def test_qwen14b_one_gpu_streaming_memory_invariants() -> None:
    """Run the smallest one-GPU Qwen memory invariant check on the expected cluster."""
    if not running_on_vista():
        pytest.skip("runs only on Vista cluster hosts")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        pytest.skip("needs at least one CUDA device")

    with tempfile.TemporaryDirectory() as tmpdir:
        row = run_qwen_case_with_torchrun(
            mode="one-gpu",
            model_ref=QWEN_MODEL_NAME,
            seq_len=1000,
            output_dir=Path(tmpdir),
            resident_suffix_count=0,
        )

    assert_qwen_invariants(row)
