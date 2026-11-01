# -*- coding: utf-8 -*-
"""torch device and VRAM management helpers (paper-agnostic)."""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

log = logging.getLogger("multiverse.torch_utils")

_DEVICE = None


def get_device():
    """Return the CUDA device; fall back to CPU when unavailable. Result is cached."""
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    import torch
    if torch.cuda.is_available():
        _DEVICE = torch.device("cuda")
        log.info("torch CUDA available: %s", torch.cuda.get_device_name(0))
    else:
        _DEVICE = torch.device("cpu")
        log.warning("torch CUDA unavailable, estimation falls back to CPU.")
    return _DEVICE


def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def reset_peak_memory() -> None:
    import torch
    if cuda_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mib() -> float:
    import torch
    if cuda_available():
        return float(torch.cuda.max_memory_allocated()) / 1048576.0
    return 0.0


def free_memory_mib() -> float:
    import torch
    if cuda_available():
        free, _total = torch.cuda.mem_get_info()
        return float(free) / 1048576.0
    return 0.0


def gpu_utilization_pct() -> float:
    """Instantaneous GPU utilization per torch's metric (sampling semantics)."""
    import torch
    if not cuda_available():
        return 0.0
    try:
        return float(torch.cuda.utilization())
    except Exception:
        return 0.0


def nvidia_smi_utilization() -> float:
    """GPU utilization percentage per nvidia-smi's metric; returns -1 when unavailable."""
    import shutil
    import subprocess
    if not shutil.which("nvidia-smi"):
        return -1.0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        ).strip().splitlines()[0]
        util, used = [x.strip() for x in out.split(",")]
        return float(util)
    except Exception:
        return -1.0


def estimate_batch_mem_bytes(n: int, k: int, batch: int, float_nbytes: int = 8) -> int:
    """Rough estimate of VRAM bytes needed for one WLS batch (Xb/yb/wb/mask + GPU-side intermediates, with 2x headroom).

    The engine actually uses float64 (8-byte) tensors; after grouping, n/k align to the same-batch maximum.
    """
    # X: batch*n*k, W*X intermediate, A: batch*k*k, B: batch*k, residual meat: batch*n*k
    elems = batch * (n * k + k * k + n * k + n * k)
    return elems * float_nbytes * 2


def adaptive_batch_size(n: int, k: int, budget_bytes: int = 8 * 1024 ** 3,
                        start: int = 512, min_batch: int = 1) -> int:
    """Adaptive initial batch size based on the budget (8GB budget by default)."""
    b = start
    while b > min_batch and estimate_batch_mem_bytes(n, k, b) > budget_bytes:
        b = max(min_batch, b // 2)
    return b
