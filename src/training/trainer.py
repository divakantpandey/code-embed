"""Custom trainer for contrastive code embedding models.

Implements the 3-stage training loop:
- Stage 1: SimCSE unsupervised contrastive learning
- Stage 2: Supervised fine-tuning with MNRL
- Stage 3: Matryoshka Representation Learning (MRL)

Supports mixed precision (AMP), gradient accumulation, gradient clipping,
device detection (MPS / CUDA / CPU), periodic evaluation, and callbacks.
"""

from __future__ import annotations

import contextlib
import json
import logging
import random
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
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
        bf16: bool = False,
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
        self.bf16 = bf16
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

        # Mixed precision and accelerator configuration
        if self.device.type == "cuda":
            # Enable TF32 by default on Ampere / Ada / Hopper architecture
            if torch.cuda.get_device_capability(0)[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            if self.bf16:
                self.use_amp = True
                self.amp_dtype = torch.bfloat16
                self.scaler = torch.amp.GradScaler("cuda", enabled=False)
            elif self.fp16:
                self.use_amp = True
                self.amp_dtype = torch.float16
                self.scaler = torch.amp.GradScaler("cuda", enabled=True)
            else:
                self.use_amp = False
                self.amp_dtype = torch.float32
                self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        elif self.device.type in ("mps", "cpu"):
            self.use_amp = self.fp16 or self.bf16
            self.amp_dtype = torch.bfloat16 if self.use_amp else torch.float32
            self.scaler = torch.amp.GradScaler("cpu", enabled=False)
        else:
            self.use_amp = False
            self.amp_dtype = torch.float32
            self.scaler = torch.amp.GradScaler("cpu", enabled=False)

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

            if epoch < self.current_epoch:
                continue

            self.current_epoch = epoch
            self.model.train()

            for cb in self.callbacks:
                cb.on_epoch_begin(self, epoch)

            steps_per_epoch = len(self.train_dataloader) // self.gradient_accumulation_steps
            steps_already_done_in_epoch = max(0, self.global_step - (epoch * steps_per_epoch))
            batches_to_skip = steps_already_done_in_epoch * self.gradient_accumulation_steps

            if batches_to_skip > 0:
                logger.info(
                    "Resuming epoch %d: fast-forwarding %d micro-batches (%d global steps already completed).",
                    epoch,
                    batches_to_skip,
                    steps_already_done_in_epoch,
                )

            for batch_idx, batch in enumerate(self.train_dataloader):
                if self.should_stop:
                    break

                if batch_idx < batches_to_skip:
                    continue

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
            q_ids = batch["query_input_ids"].to(self.device, non_blocking=True)
            q_mask = batch["query_attention_mask"].to(self.device, non_blocking=True)
            c_ids = batch["code_input_ids"].to(self.device, non_blocking=True)
            c_mask = batch["code_attention_mask"].to(self.device, non_blocking=True)

            with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                q_embeds = self.model(input_ids=q_ids, attention_mask=q_mask)
                c_embeds = self.model(input_ids=c_ids, attention_mask=c_mask)

                # Optional hard negatives
                neg_embeds = None
                if "neg_input_ids" in batch:
                    neg_ids = batch["neg_input_ids"].to(self.device, non_blocking=True)
                    neg_mask = batch["neg_attention_mask"].to(self.device, non_blocking=True)
                    neg_embeds = self.model(input_ids=neg_ids, attention_mask=neg_mask)

                if neg_embeds is not None:
                    loss = self.loss_fn(q_embeds, c_embeds, neg_embeds)
                else:
                    loss = self.loss_fn(q_embeds, c_embeds)

            return loss

        # Case B: SimCSE single-text batch (pass twice with dropout)
        if isinstance(batch, dict) and "input_ids" in batch:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)

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
        """Run evaluation using custom eval_fn and/or eval_dataloader."""
        self.model.eval()
        metrics: dict[str, float] = {}

        if self.eval_dataloader is not None:
            total_loss = 0.0
            num_batches = 0
            with torch.no_grad():
                for batch in self.eval_dataloader:
                    loss = self._training_step(batch)
                    total_loss += loss.item()
                    num_batches += 1
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                metrics["eval_loss"] = avg_loss
                metrics["loss"] = avg_loss

        if self.eval_fn is not None:
            custom_metrics = self.eval_fn()
            metrics.update(custom_metrics)

        if not metrics:
            return {"loss": 0.0}

        if "loss" not in metrics and "eval_loss" in metrics:
            metrics["loss"] = metrics["eval_loss"]
        elif "eval_loss" not in metrics and "loss" in metrics:
            metrics["eval_loss"] = metrics["loss"]

        logger.info("Evaluation results: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path) -> None:
        """Save complete Hugging Face compatible checkpoint to target directory."""
        dest = Path(path)
        dest.mkdir(parents=True, exist_ok=True)

        # 1. Save model weights and configuration
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(dest)
        else:
            torch.save(self.model.state_dict(), dest / "model.pt")

        # 2. Save SentenceTransformers format for seamless ecosystem integration
        try:
            if hasattr(self.model, "save_as_sentence_transformer"):
                self.model.save_as_sentence_transformer(dest / "sentence_transformers")
            elif hasattr(self.model, "encoder") and hasattr(
                self.model.encoder, "save_as_sentence_transformer"
            ):
                self.model.encoder.save_as_sentence_transformer(dest / "sentence_transformers")
        except Exception as exc:  # noqa: BLE001
            logger.debug("SentenceTransformers export skipped during checkpoint: %s", exc)

        # 3. Save optimizer, scheduler, and scaler states
        torch.save(self.optimizer.state_dict(), dest / "optimizer.pt")
        torch.save(self.scheduler.state_dict(), dest / "scheduler.pt")
        if hasattr(self, "scaler") and self.scaler.is_enabled():
            torch.save(self.scaler.state_dict(), dest / "scaler.pt")

        # 4. Save trainer metadata and progress
        trainer_state = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "total_training_steps": self.total_training_steps,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "scheduler_name": self.scheduler_name,
            "fp16": self.fp16,
            "bf16": self.bf16,
        }
        with open(dest / "trainer_state.json", "w", encoding="utf-8") as f:
            json.dump(trainer_state, f, indent=2)

        # 5. Save RNG states for deterministic resumption
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(rng_state, dest / "rng_state.pt")
        logger.info("Saved complete HF checkpoint to %s (step=%d)", dest, self.global_step)

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Load complete model checkpoint and training state from disk."""
        src = Path(path)
        logger.info("Loading checkpoint from %s ...", src)

        # 1. Load model weights
        model_weights = src / "model.pt"
        if model_weights.exists():
            self.model.load_state_dict(torch.load(model_weights, map_location=self.device))
        elif (src / "code_embed_config.json").exists() or (
            src / "pycode_embed_config.json"
        ).exists():
            target_encoder = (
                self.model.encoder
                if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "projection_head")
                else self.model
            )
            proj_pt = src / "projection_head.pt"
            if (
                proj_pt.exists()
                and hasattr(target_encoder, "projection_head")
                and target_encoder.projection_head is not None
            ):
                with contextlib.suppress(Exception):
                    target_encoder.projection_head.load_state_dict(
                        torch.load(proj_pt, map_location=self.device)
                    )
            if (src / "backbone").exists() and hasattr(target_encoder, "backbone"):
                try:
                    from transformers import AutoModel

                    target_encoder.backbone = AutoModel.from_pretrained(str(src / "backbone")).to(
                        self.device
                    )
                except Exception:
                    pass
        else:
            logger.warning("No model weight files found in checkpoint %s", src)

        # 2. Load optimizer, scheduler, and scaler states
        opt_path = src / "optimizer.pt"
        if opt_path.exists():
            self.optimizer.load_state_dict(torch.load(opt_path, map_location=self.device))
        sched_path = src / "scheduler.pt"
        if sched_path.exists():
            self.scheduler.load_state_dict(torch.load(sched_path, map_location=self.device))
        scaler_path = src / "scaler.pt"
        if scaler_path.exists() and hasattr(self, "scaler"):
            self.scaler.load_state_dict(torch.load(scaler_path, map_location=self.device))

        # 3. Load RNG state if available
        rng_path = src / "rng_state.pt"
        if rng_path.exists():
            try:
                rng_state = torch.load(rng_path, map_location="cpu")
                if "python" in rng_state:
                    random.setstate(rng_state["python"])
                if "numpy" in rng_state:
                    np.random.set_state(rng_state["numpy"])
                if "torch" in rng_state:
                    torch.set_rng_state(rng_state["torch"])
                if (
                    "cuda" in rng_state
                    and rng_state["cuda"] is not None
                    and torch.cuda.is_available()
                ):
                    torch.cuda.set_rng_state_all(rng_state["cuda"])
            except Exception as rng_err:  # noqa: BLE001
                logger.debug("Could not restore RNG states: %s", rng_err)

        # 4. Load trainer state
        state_path = src / "trainer_state.json"
        trainer_state: dict[str, Any] = {}
        if state_path.exists():
            with open(state_path, encoding="utf-8") as f:
                trainer_state = json.load(f)
            self.global_step = trainer_state.get("global_step", self.global_step)
            self.current_epoch = trainer_state.get("current_epoch", self.current_epoch)
            logger.info(
                "Restored training state: resumed from global_step=%d, epoch=%d",
                self.global_step,
                self.current_epoch,
            )

        return trainer_state
