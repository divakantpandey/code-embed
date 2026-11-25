"""Process environment: ``.env`` loading and the CPU budget this container really has."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# GPU selection belongs to the launcher; a CUDA_VISIBLE_DEVICES copied in from another project's .env
# would silently train on one GPU of a multi-GPU pod.
NOT_FROM_DOTENV = frozenset({"CUDA_VISIBLE_DEVICES"})


def load_dotenv(path: str | Path = ".env") -> list[str]:
    """Set ``KEY=VALUE`` lines from ``path`` that are not already in the environment.

    Real environment variables (RunPod secrets, shell exports) win over the file. Returns the names set.
    """
    path = Path(path)
    if not path.is_file():
        return []
    loaded: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key in NOT_FROM_DOTENV:
            logger.warning("ignoring %s from %s; pick GPUs with the launcher instead", key, path)
            continue
        if key in os.environ:
            continue
        os.environ[key] = value
        loaded.append(key)
    return loaded


def available_cpus() -> int:
    """CPUs this process may use. ``os.cpu_count()`` reports the host, which on a RunPod container can be
    several times the cgroup quota and would oversubscribe dataloader workers."""
    quota: float | None = None
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    if cpu_max.is_file():
        limit, period = cpu_max.read_text().split()[:2]
        if limit != "max":
            quota = int(limit) / int(period)
    else:
        v1_quota = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        v1_period = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if v1_quota.is_file() and v1_period.is_file():
            q = int(v1_quota.read_text())
            if q > 0:
                quota = q / int(v1_period.read_text())
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    # Round up: a 7.65-CPU quota (RunPod A40) leaves 0.65 of a core idle with 7 workers; 8 cost only a
    # little throttling.
    return max(1, min(affinity, math.ceil(quota)) if quota else affinity)
