from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_torchrun(
    module: str,
    args: Sequence[str],
    *,
    nproc_per_node: int,
    env: Mapping[str, str] | None = None,
) -> None:
    """Launch a Python module through torchrun with repo-local defaults."""
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={nproc_per_node}",
        "-m",
        module,
        *args,
    ]
    merged_env = os.environ.copy()
    merged_env.setdefault("OMP_NUM_THREADS", "1")
    merged_env.setdefault("MKL_NUM_THREADS", "1")
    merged_env.setdefault("TORCH_NUM_THREADS", "1")
    if env:
        merged_env.update(env)
    subprocess.run(command, cwd=PROJECT_ROOT, env=merged_env, check=True)
