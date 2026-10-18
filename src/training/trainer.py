"""Custom trainer for contrastive code embedding models.

Implements the 3-stage training loop:
- Stage 1: SimCSE unsupervised contrastive learning
- Stage 2: Supervised fine-tuning with MNRL
- Stage 3: Matryoshka Representation Learning (MRL)

Supports mixed precision (AMP), gradient accumulation, gradient clipping,
device detection (MPS / CUDA / CPU), periodic evaluation, and callbacks.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.training.callbacks import Callback
from src.training.scheduler import get_scheduler

logger = logging.getLogger(__name__)


def get_default_device(preference: str = "auto") -> torch.device:
    """Resolve compute device based on preference and hardware availability."""
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if preference == "cpu":
        return torch.device("cpu")

    # Auto-detection
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CodeEmbeddingTrainer:
    """Trainer managing optimizer, scheduler, forward/backward passes, and callbacks.

    Parameters
    ----------
    model : nn.Module
        The CodeEncoder or MatryoshkaEncoder instance.
    loss_fn : nn.Module
        InfoNCELoss, MultipleNegativesRankingLoss, or MatryoshkaLoss.
    train_dataloader : DataLoader
        DataLoader yielding training batches.
    eval_dataloader : DataLoader | None
        Optional DataLoader for evaluation.
    eval_fn : Callable[[], dict[str, float]] | None
        Optional custom evaluation function returning a metrics dict.
    learning_rate : float
        Peak learning rate.
    weight_decay : float
        L2 weight decay for AdamW.
    epochs : int
        Total training epochs.
    gradient_accumulation_steps : int
        Number of forward batches before an optimizer step.
    max_grad_norm : float
        Gradient norm clipping threshold.
    warmup_ratio : float
        Fraction of total steps devoted to linear warm-up.
    scheduler_name : str
        One of 'cosine', 'cosine_with_restarts', 'linear', 'constant_with_warmup'.
    fp16 : bool
        Enable mixed precision training (AMP).
    device : str | torch.device
        Target device ('auto', 'mps', 'cuda', 'cpu').
    eval_every_n_steps : int
        Run evaluation every N training steps.
    callbacks : Sequence[Callback] | None
        List of Callback instances.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_dataloader: DataLoader[Any],
        eval_dataloader: DataLoader[Any] | None = None,
        eval_fn: Callable[[], dict[str, float]] | None = None,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        epochs: int = 3,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        warmup_ratio: float = 0.1,
        scheduler_name: str = "cosine",
        fp16: bool = True,
        device: str | torch.device = "auto",
        eval_every_n_steps: int = 500,
        callbacks: Sequence[Callback] | None = None,
    ) -> None:
        self.device = get_default_device(device) if isinstance(device, str) else device
        self.model: Any = model.to(self.device)
        self.loss_fn: Any = loss_fn.to(self.device)

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.eval_fn = eval_fn

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.max_grad_norm = max_grad_norm
        self.warmup_ratio = warmup_ratio
        self.scheduler_name = scheduler_name
        self.fp16 = fp16
        self.eval_every_n_steps = eval_every_n_steps
        self.callbacks = list(callbacks or [])

        # Optimizer setup: separate weight decay for bias / LayerNorm
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

        # Scheduler setup
        self.total_training_steps = max(
            1, (len(self.train_dataloader) // self.gradient_accumulation_steps) * self.epochs
        )
        self.num_warmup_steps = int(self.total_training_steps * self.warmup_ratio)
        self.scheduler = get_scheduler(
            self.scheduler_name,
            self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.total_training_steps,
        )

        # Mixed precision scaler (CUDA only; MPS uses autocast without GradScaler)
        self.use_amp = self.fp16 and self.device.type in ("cuda", "mps", "cpu")
        self.amp_dtype = (
            torch.float16
            if self.device.type == "cuda"
            else torch.bfloat16
            if self.device.type in ("mps", "cpu")
            else torch.float32
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.fp16 and self.device.type == "cuda")
        )

        # State
        self.global_step = 0
        self.current_epoch = 0
        self.should_stop = False

    # ------------------------------------------------------------------
    # Core training loop
    # ------------------------------------------------------------------

    def train(self) -> dict[str, float]:
        """Execute the full training pipeline across configured epochs."""
        logger.info(
            "Starting training on %s: %d epochs, %d total steps, lr=%.2e",
            self.device,
            self.epochs,
            self.total_training_steps,
            self.learning_rate,
        )

        for cb in self.callbacks:
            cb.on_train_begin(self)

        step_loss = 0.0
        start_time = time.time()

        for epoch in range(self.epochs):
            if self.should_stop:
                break

            self.current_epoch = epoch
            self.model.train()

            for cb in self.callbacks:
                cb.on_epoch_begin(self, epoch)

            for batch_idx, batch in enumerate(self.train_dataloader):
                if self.should_stop:
                    break

                for cb in self.callbacks:
                    cb.on_step_begin(self, self.global_step)

                loss = self._training_step(batch)
                loss_value = loss.item()
                step_loss += loss_value

                # Backward pass with gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Clip gradients
                    if self.scaler.is_enabled():
                        self.scaler.unscale_(self.optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    ).item()

                    # Step optimizer
                    if self.scaler.is_enabled():
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    avg_loss = step_loss / self.gradient_accumulation_steps
                    step_loss = 0.0

                    logs = {
                        "loss": avg_loss,
                        "learning_rate": current_lr,
                        "grad_norm": grad_norm,
                        "epoch": epoch + (batch_idx + 1) / len(self.train_dataloader),
                    }

                    for cb in self.callbacks:
                        cb.on_step_end(self, self.global_step, logs)

                    # Periodic evaluation
                    if self.global_step % self.eval_every_n_steps == 0:
                        eval_metrics = self.evaluate()
                        for cb in self.callbacks:
                            cb.on_eval_end(self, eval_metrics)
                        self.model.train()

            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch)

            if (
                self.eval_dataloader is not None or self.eval_fn is not None
            ) and not self.should_stop:
                eval_metrics = self.evaluate()
                for cb in self.callbacks:
                    cb.on_eval_end(self, eval_metrics)
                self.model.train()

        elapsed = time.time() - start_time
        logger.info("Training completed in %.2f seconds.", elapsed)

        final_eval = self.evaluate()
        for cb in self.callbacks:
            cb.on_eval_end(self, final_eval)
            cb.on_train_end(self)

        return final_eval

    # ------------------------------------------------------------------
    # Forward & loss dispatch
    # ------------------------------------------------------------------

    def _training_step(self, batch: Any) -> torch.Tensor:
        """Dispatch batch to model and loss function based on batch structure."""
        # Case A: Pair dictionary (queries, codes) from PairCollator
        if isinstance(batch, dict) and "query_input_ids" in batch:
            q_ids = batch["query_input_ids"].to(self.device)
            q_mask = batch["query_attention_mask"].to(self.device)
            c_ids = batch["code_input_ids"].to(self.device)
            c_mask = batch["code_attention_mask"].to(self.device)

            with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                q_embeds = self.model(input_ids=q_ids, attention_mask=q_mask)
                c_embeds = self.model(input_ids=c_ids, attention_mask=c_mask)

                # Optional hard negatives
                neg_embeds = None
                if "neg_input_ids" in batch:
                    neg_ids = batch["neg_input_ids"].to(self.device)
                    neg_mask = batch["neg_attention_mask"].to(self.device)
                    neg_embeds = self.model(input_ids=neg_ids, attention_mask=neg_mask)

                if neg_embeds is not None:
                    loss = self.loss_fn(q_embeds, c_embeds, neg_embeds)
                else:
                    loss = self.loss_fn(q_embeds, c_embeds)

            return loss

        # Case B: SimCSE single-text batch (pass twice with dropout)
        if isinstance(batch, dict) and "input_ids" in batch:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                # Pass 1
                embeds1 = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # Pass 2 (different dropout mask)
                embeds2 = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.loss_fn(embeds1, embeds2)

            return loss

        raise ValueError(f"Unrecognized batch format: {type(batch)}")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> dict[str, float]:
        """Run evaluation using either custom eval_fn or eval_dataloader."""
        self.model.eval()

        if self.eval_fn is not None:
            metrics = self.eval_fn()
            if "loss" not in metrics:
                metrics["loss"] = metrics.get("eval_loss", 0.0)
            return metrics

        if self.eval_dataloader is None:
            return {"loss": 0.0}

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.eval_dataloader:
                loss = self._training_step(batch)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        metrics = {"eval_loss": avg_loss}
        logger.info("Evaluation results: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model checkpoint to target directory."""
        dest = Path(path)
        dest.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(dest)
        else:
            torch.save(self.model.state_dict(), dest / "model.pt")

        torch.save(self.optimizer.state_dict(), dest / "optimizer.pt")
        torch.save(self.scheduler.state_dict(), dest / "scheduler.pt")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model checkpoint from disk."""
        src = Path(path)
        model_weights = src / "model.pt"
        if model_weights.exists():
            self.model.load_state_dict(torch.load(model_weights, map_location=self.device))
        elif (src / "code_embed_config.json").exists() or (src / "pycode_embed_config.json").exists():
            from src.model.encoder import CodeEncoder

            loaded = CodeEncoder.from_pretrained(src).to(self.device)
            if hasattr(self.model, "encoder") and isinstance(self.model.encoder, CodeEncoder):
                self.model.encoder = loaded
            else:
                self.model = loaded

        opt_path = src / "optimizer.pt"
        if opt_path.exists():
            self.optimizer.load_state_dict(torch.load(opt_path, map_location=self.device))
        sched_path = src / "scheduler.pt"
        if sched_path.exists():
            self.scheduler.load_state_dict(torch.load(sched_path, map_location=self.device))
