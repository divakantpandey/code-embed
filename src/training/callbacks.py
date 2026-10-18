"""Callbacks for the training loop: logging, checkpointing, early stopping, and alerts."""

from __future__ import annotations

import logging
import math
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.training.trainer import CodeEmbeddingTrainer

logger = logging.getLogger(__name__)


class Callback:
    """Base callback class for training events."""

    def on_train_begin(self, trainer: CodeEmbeddingTrainer) -> None:
        pass

    def on_train_end(self, trainer: CodeEmbeddingTrainer) -> None:
        pass

    def on_epoch_begin(self, trainer: CodeEmbeddingTrainer, epoch: int) -> None:
        pass

    def on_epoch_end(self, trainer: CodeEmbeddingTrainer, epoch: int) -> None:
        pass

    def on_step_begin(self, trainer: CodeEmbeddingTrainer, step: int) -> None:
        pass

    def on_step_end(self, trainer: CodeEmbeddingTrainer, step: int, logs: dict[str, float]) -> None:
        pass

    def on_eval_end(self, trainer: CodeEmbeddingTrainer, metrics: dict[str, float]) -> None:
        pass


class WandBCallback(Callback):
    """Logs training and evaluation metrics to Weights & Biases."""

    def __init__(
        self,
        project: str = "code-embed",
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        log_every_n_steps: int = 10,
        enabled: bool = True,
    ) -> None:
        self.project = project
        self.run_name = run_name
        self.config = config or {}
        self.log_every_n_steps = log_every_n_steps
        self.enabled = enabled
        self._wandb: Any = None

    def on_train_begin(self, trainer: CodeEmbeddingTrainer) -> None:
        if not self.enabled:
            return
        try:
            import wandb

            self._wandb = wandb
            wandb.init(
                project=self.project,
                name=self.run_name,
                config=self.config,
                reinit=True,
            )
            logger.info("Initialized Weights & Biases tracking in project %s", self.project)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not initialize W&B: %s. Disabling W&B logging.", exc)
            self.enabled = False

    def on_step_end(self, trainer: CodeEmbeddingTrainer, step: int, logs: dict[str, float]) -> None:
        if not self.enabled or self._wandb is None:
            return
        if step % self.log_every_n_steps == 0:
            prefixed = {f"train/{k}": v for k, v in logs.items()}
            self._wandb.log(prefixed, step=step)

    def on_eval_end(self, trainer: CodeEmbeddingTrainer, metrics: dict[str, float]) -> None:
        if not self.enabled or self._wandb is None:
            return
        prefixed = {f"eval/{k}": v for k, v in metrics.items()}
        self._wandb.log(prefixed, step=trainer.global_step)

    def on_train_end(self, trainer: CodeEmbeddingTrainer) -> None:
        if self.enabled and self._wandb is not None:
            self._wandb.finish()


class TensorBoardCallback(Callback):
    """Logs scalar metrics to TensorBoard."""

    def __init__(self, log_dir: str | Path = "./runs", enabled: bool = True) -> None:
        self.log_dir = Path(log_dir)
        self.enabled = enabled
        self.writer: Any = None

    def on_train_begin(self, trainer: CodeEmbeddingTrainer) -> None:
        if not self.enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            logger.info("TensorBoard logging enabled at %s", self.log_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not initialize TensorBoard writer: %s", exc)
            self.enabled = False

    def on_step_end(self, trainer: CodeEmbeddingTrainer, step: int, logs: dict[str, float]) -> None:
        if not self.enabled or self.writer is None:
            return
        for k, v in logs.items():
            self.writer.add_scalar(f"train/{k}", v, step)

    def on_eval_end(self, trainer: CodeEmbeddingTrainer, metrics: dict[str, float]) -> None:
        if not self.enabled or self.writer is None:
            return
        for k, v in metrics.items():
            self.writer.add_scalar(f"eval/{k}", v, trainer.global_step)

    def on_train_end(self, trainer: CodeEmbeddingTrainer) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


class EarlyStoppingCallback(Callback):
    """Early stops training when a monitored metric ceases to improve."""

    def __init__(
        self,
        metric: str = "eval/MRR@10",
        patience: int = 3,
        mode: str = "max",
        min_delta: float = 1e-4,
    ) -> None:
        self.metric = metric
        self.patience = patience
        self.mode = mode.lower()
        self.min_delta = min_delta
        self.best_score: float | None = None
        self.bad_epochs: int = 0

        if self.mode not in ("min", "max"):
            raise ValueError(f"Mode must be 'min' or 'max', got {self.mode}")

    def on_eval_end(self, trainer: CodeEmbeddingTrainer, metrics: dict[str, float]) -> None:
        # Check raw metric or with/without eval/ prefix
        val = metrics.get(self.metric)
        if val is None:
            short_key = self.metric.removeprefix("eval/")
            val = metrics.get(short_key)

        if val is None:
            return

        if self.best_score is None:
            self.best_score = val
            return

        improved = (
            val > self.best_score + self.min_delta
            if self.mode == "max"
            else val < self.best_score - self.min_delta
        )

        if improved:
            self.best_score = val
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            logger.info(
                "EarlyStopping: metric %s did not improve (patience %d/%d).",
                self.metric,
                self.bad_epochs,
                self.patience,
            )
            if self.bad_epochs >= self.patience:
                logger.warning("EarlyStopping: Patience reached. Stopping training.")
                trainer.should_stop = True


class AlertCallback(Callback):
    """Detects anomalies (loss spikes, gradient explosions, NaN loss)."""

    def __init__(
        self,
        loss_spike_multiplier: float = 3.0,
        grad_norm_threshold: float = 10.0,
        nan_halt: bool = True,
        rolling_window_size: int = 20,
    ) -> None:
        self.loss_spike_multiplier = loss_spike_multiplier
        self.grad_norm_threshold = grad_norm_threshold
        self.nan_halt = nan_halt
        self.loss_history: deque[float] = deque(maxlen=rolling_window_size)

    def on_step_end(self, trainer: CodeEmbeddingTrainer, step: int, logs: dict[str, float]) -> None:
        loss = logs.get("loss")
        if loss is not None:
            if math.isnan(loss) or math.isinf(loss):
                msg = f"ALERT: Loss became {loss} at step {step}!"
                logger.error(msg)
                if self.nan_halt:
                    trainer.should_stop = True
                return

            if len(self.loss_history) >= 5:
                avg_loss = sum(self.loss_history) / len(self.loss_history)
                if loss > self.loss_spike_multiplier * avg_loss:
                    logger.warning(
                        "ALERT: Loss spike detected! Step %d: loss=%.4f (avg=%.4f, multiplier=%.1fx)",
                        step,
                        loss,
                        avg_loss,
                        loss / max(avg_loss, 1e-8),
                    )

            self.loss_history.append(loss)

        grad_norm = logs.get("grad_norm")
        if grad_norm is not None and grad_norm > self.grad_norm_threshold:
            logger.warning(
                "ALERT: Large gradient norm detected! Step %d: grad_norm=%.4f > %.1f",
                step,
                grad_norm,
                self.grad_norm_threshold,
            )


class CheckpointCallback(Callback):
    """Saves best and periodic checkpoints to disk."""

    def __init__(
        self,
        checkpoint_dir: str | Path = "./checkpoints",
        save_every_n_steps: int = 1000,
        save_best: bool = True,
        save_last: bool = True,
        metric: str = "eval/MRR@10",
        mode: str = "max",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_every_n_steps = save_every_n_steps
        self.save_best = save_best
        self.save_last = save_last
        self.metric = metric
        self.mode = mode.lower()
        self.best_metric: float | None = None

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, trainer: CodeEmbeddingTrainer, step: int, logs: dict[str, float]) -> None:
        if step > 0 and step % self.save_every_n_steps == 0:
            step_dir = self.checkpoint_dir / f"checkpoint_step_{step}"
            trainer.save_checkpoint(step_dir)
            logger.info("Saved periodic checkpoint to %s", step_dir)

    def on_eval_end(self, trainer: CodeEmbeddingTrainer, metrics: dict[str, float]) -> None:
        if not self.save_best:
            return

        score = metrics.get(self.metric)
        if score is None:
            short_key = self.metric.removeprefix("eval/")
            score = metrics.get(short_key)

        if score is None:
            return

        is_better = False
        if (
            self.best_metric is None
            or self.mode == "max"
            and score > self.best_metric
            or self.mode == "min"
            and score < self.best_metric
        ):
            is_better = True

        if is_better:
            self.best_metric = score
            best_dir = self.checkpoint_dir / "best_model"
            trainer.save_checkpoint(best_dir)
            logger.info("New best model (%s=%.4f) saved to %s", self.metric, score, best_dir)

    def on_train_end(self, trainer: CodeEmbeddingTrainer) -> None:
        if self.save_last:
            last_dir = self.checkpoint_dir / "last_model"
            trainer.save_checkpoint(last_dir)
            logger.info("Saved final checkpoint to %s", last_dir)
