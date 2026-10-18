"""Learning rate scheduler factory for the training pipeline.

Wraps standard ``torch`` / ``transformers`` scheduler implementations behind a
single factory function so the training config can select a scheduler by name.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

SUPPORTED_SCHEDULERS = frozenset(
    {"cosine", "cosine_with_restarts", "linear", "constant_with_warmup"}
)


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    *,
    num_cycles: int = 1,
) -> LambdaLR:
    """Create a learning-rate scheduler by name.

    All schedulers include a linear warm-up phase over ``num_warmup_steps``.

    Args:
        name: One of ``'cosine'``, ``'cosine_with_restarts'``, ``'linear'``,
            or ``'constant_with_warmup'``.
        optimizer: The optimizer whose learning rate will be scheduled.
        num_warmup_steps: Number of warm-up steps (linear ramp from 0 → base LR).
        num_training_steps: Total number of training steps (including warm-up).
        num_cycles: Number of cosine-annealing cycles.  Only used by
            ``'cosine_with_restarts'``. Defaults to ``1``.

    Returns:
        A ``LambdaLR`` scheduler instance.

    Raises:
        ValueError: If ``name`` is not a supported scheduler.
    """
    if name not in SUPPORTED_SCHEDULERS:
        raise ValueError(f"Unknown scheduler '{name}'. Choose from {sorted(SUPPORTED_SCHEDULERS)}")

    schedule_fn = _SCHEDULE_FACTORIES[name]
    lr_lambda = schedule_fn(
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )

    scheduler = LambdaLR(optimizer, lr_lambda)
    logger.info(
        "Created scheduler '%s' (warmup=%d, total=%d)",
        name,
        num_warmup_steps,
        num_training_steps,
    )
    return scheduler


# ---------------------------------------------------------------------------
# Schedule lambda factories
# ---------------------------------------------------------------------------


def _linear_schedule(
    num_warmup_steps: int,
    num_training_steps: int,
    **_: object,
) -> Callable[[int], float]:
    """Linear warm-up followed by linear decay to 0."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return lr_lambda


def _cosine_schedule(
    num_warmup_steps: int,
    num_training_steps: int,
    **_: object,
) -> Callable[[int], float]:
    """Linear warm-up followed by cosine decay to 0."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return lr_lambda


def _cosine_with_restarts_schedule(
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,
    **_: object,
) -> Callable[[int], float]:
    """Linear warm-up followed by cosine decay with hard restarts."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((progress * num_cycles) % 1.0))))

    return lr_lambda


def _constant_with_warmup_schedule(
    num_warmup_steps: int,
    **_: object,
) -> Callable[[int], float]:
    """Linear warm-up followed by constant LR."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return lr_lambda


# Mapping from name → factory
_SCHEDULE_FACTORIES: dict[str, Any] = {
    "linear": _linear_schedule,
    "cosine": _cosine_schedule,
    "cosine_with_restarts": _cosine_with_restarts_schedule,
    "constant_with_warmup": _constant_with_warmup_schedule,
}
