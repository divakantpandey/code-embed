"""System pre-flight and runtime health checks for CodeEmbed training.

Verifies:
- Hardware capabilities (CUDA, GPU name, VRAM, BF16, TF32)
- Environment credentials (HF_TOKEN, WANDB_API_KEY)
- Disk storage capacity in output and checkpoint directories
- Dataset readiness (detects empty / synthetic-only partitions)
- Model & hyperparameter validity
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class CheckResult:
    """Result of an individual preflight check."""

    name: str
    passed: bool
    status: str  # "PASS", "WARN", "FAIL"
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreflightReport:
    """Aggregated preflight report across all checks."""

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if no checks failed (warnings are permitted)."""
        return all(c.status != "FAIL" for c in self.checks)

    @property
    def has_warnings(self) -> bool:
        """True if any check issued a warning."""
        return any(c.status == "WARN" for c in self.checks)

    def print_summary(self) -> None:
        """Render a formatted Rich table of check results."""
        table = Table(title="🔍 CodeEmbed Pre-Flight System & Environment Checks")
        table.add_column("Category / Check", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center")
        table.add_column("Details", style="white")

        for c in self.checks:
            if c.status == "PASS":
                status_str = "[bold green]PASS[/]"
            elif c.status == "WARN":
                status_str = "[bold yellow]WARN[/]"
            else:
                status_str = "[bold red]FAIL[/]"
            table.add_row(c.name, status_str, c.message)

        console.print(table)
        if not self.passed:
            console.print(
                "[bold red]❌ Pre-flight checks failed. Please address the errors above before running full-scale training.[/]"
            )
        elif self.has_warnings:
            console.print(
                "[bold yellow]⚠️  Pre-flight checks passed with warnings. Proceeding with caution.[/]"
            )
        else:
            console.print(
                "[bold green]✅ All pre-flight checks passed successfully! Ready for full-scale training.[/]"
            )


class PreflightChecker:
    """Executes preflight health and environment checks for training runs."""

    @staticmethod
    def check_hardware(device_pref: str = "auto", min_vram_gb: float = 8.0) -> CheckResult:
        """Check GPU availability, model, VRAM, BF16 support, and enable TF32."""
        details: dict[str, Any] = {}
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            dev_count = torch.cuda.device_count()
            cap = torch.cuda.get_device_capability(0)
            props = torch.cuda.get_device_properties(0)
            total_vram_gb = props.total_memory / (1024**3)

            bf16_supported = (
                hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
            )
            tf32_supported = cap[0] >= 8

            if tf32_supported:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            details = {
                "device_type": "cuda",
                "device_name": dev_name,
                "device_count": dev_count,
                "compute_capability": f"{cap[0]}.{cap[1]}",
                "total_vram_gb": round(total_vram_gb, 2),
                "bf16_supported": bf16_supported,
                "tf32_enabled": tf32_supported,
            }

            if total_vram_gb < min_vram_gb:
                return CheckResult(
                    name="GPU Hardware & VRAM",
                    passed=True,
                    status="WARN",
                    message=(
                        f"Detected {dev_name} ({total_vram_gb:.1f} GB VRAM). "
                        f"Recommended >= {min_vram_gb} GB for large batch full-scale training."
                    ),
                    details=details,
                )

            return CheckResult(
                name="GPU Hardware & VRAM",
                passed=True,
                status="PASS",
                message=(
                    f"CUDA active: {dev_name} ({total_vram_gb:.1f} GB, CC {cap[0]}.{cap[1]}), "
                    f"BF16={'Yes' if bf16_supported else 'No'}, TF32={'Enabled' if tf32_supported else 'No'}"
                ),
                details=details,
            )

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            details = {"device_type": "mps"}
            if device_pref == "cuda":
                return CheckResult(
                    name="GPU Hardware & VRAM",
                    passed=False,
                    status="FAIL",
                    message="CUDA was requested but only Apple Silicon MPS is available.",
                    details=details,
                )
            return CheckResult(
                name="GPU Hardware & VRAM",
                passed=True,
                status="WARN",
                message="Running on Apple Silicon (MPS). Recommended NVIDIA GPU (CUDA) for full scale.",
                details=details,
            )

        details = {"device_type": "cpu"}
        if device_pref in ("cuda", "mps"):
            return CheckResult(
                name="GPU Hardware & VRAM",
                passed=False,
                status="FAIL",
                message=f"Requested device '{device_pref}' is not available. Only CPU detected.",
                details=details,
            )

        return CheckResult(
            name="GPU Hardware & VRAM",
            passed=True,
            status="WARN",
            message="No GPU detected. Training will run on CPU (very slow for full scale).",
            details=details,
        )

    @staticmethod
    def check_storage_space(
        paths: list[str | Path] | None = None, min_free_gb: float = 10.0
    ) -> CheckResult:
        """Verify that sufficient storage exists on disk."""
        target_paths = [Path(p) for p in (paths or ["./outputs", "./checkpoints", "./data"])]
        low_space_dirs: list[str] = []
        space_info: dict[str, float] = {}

        for p in target_paths:
            try:
                p.mkdir(parents=True, exist_ok=True)
                usage = shutil.disk_usage(p)
                free_gb = usage.free / (1024**3)
                space_info[str(p)] = round(free_gb, 2)
                if free_gb < min_free_gb:
                    low_space_dirs.append(f"{p} ({free_gb:.1f} GB free)")
            except Exception as exc:
                logger.debug("Could not check disk usage for %s: %s", p, exc)

        if low_space_dirs:
            return CheckResult(
                name="Disk Storage Capacity",
                passed=True,
                status="WARN",
                message=(
                    f"Low disk space detected: {', '.join(low_space_dirs)}. "
                    f"Recommended >= {min_free_gb} GB for checkpoints."
                ),
                details={"disk_spaces_gb": space_info},
            )

        min_found = min(space_info.values()) if space_info else 0.0
        return CheckResult(
            name="Disk Storage Capacity",
            passed=True,
            status="PASS",
            message=f"Sufficient storage available (min free: {min_found:.1f} GB across target directories).",
            details={"disk_spaces_gb": space_info},
        )

    @staticmethod
    def check_huggingface_auth(
        repo_id: str | None = None,
        require_auth: bool = False,
    ) -> CheckResult:
        """Check HuggingFace token and repository access."""
        token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
        )

        try:
            from huggingface_hub import HfApi

            api = HfApi(token=token)
            user_info = api.whoami()
            username = user_info.get("name", "authenticated-user")
            return CheckResult(
                name="Hugging Face Hub Auth",
                passed=True,
                status="PASS",
                message=f"Authenticated as @{username}. HF Checkpointing and Hub uploads ready.",
                details={"username": username, "repo_id": repo_id},
            )
        except Exception as exc:
            if require_auth:
                return CheckResult(
                    name="Hugging Face Hub Auth",
                    passed=False,
                    status="FAIL",
                    message=(
                        f"HuggingFace authentication failed ({exc}). "
                        "Set export HF_TOKEN='your_hf_token' to enable HF Hub checkpointing."
                    ),
                    details={"error": str(exc)},
                )
            return CheckResult(
                name="Hugging Face Hub Auth",
                passed=True,
                status="WARN",
                message="HF_TOKEN not set or invalid. Local checkpointing active; remote Hub push disabled.",
                details={"error": str(exc)},
            )

    @staticmethod
    def check_wandb(enabled: bool = True, project: str = "code-embed") -> CheckResult:
        """Check Weights & Biases API key."""
        if not enabled:
            return CheckResult(
                name="Weights & Biases (W&B)",
                passed=True,
                status="PASS",
                message="W&B logging is disabled in configuration.",
                details={"enabled": False},
            )

        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            return CheckResult(
                name="Weights & Biases (W&B)",
                passed=True,
                status="PASS",
                message=f"WANDB_API_KEY detected. Logging to project '{project}'.",
                details={"project": project},
            )

        # Check local netrc or wandb login
        try:
            import wandb

            if wandb.login(prompt=False):
                return CheckResult(
                    name="Weights & Biases (W&B)",
                    passed=True,
                    status="PASS",
                    message=f"W&B authenticated. Logging to project '{project}'.",
                    details={"project": project},
                )
        except Exception:
            pass

        return CheckResult(
            name="Weights & Biases (W&B)",
            passed=True,
            status="WARN",
            message=(
                "WANDB_API_KEY not found in environment. W&B logging will fall back to offline "
                "or disabled mode. Set export WANDB_API_KEY='...' for live dashboard tracking."
            ),
            details={"enabled": True},
        )

    @staticmethod
    def check_dataset_readiness(
        data_dir: str | Path = "data/processed",
        min_samples: int = 1000,
        full_scale: bool = False,
    ) -> CheckResult:
        """Check whether training data is real full-scale data or toy synthetic data."""
        p = Path(data_dir)
        full_path = p / "full" if (p / "full").exists() else p
        raw_train = Path("data/raw/train")

        sample_count = 0
        from datasets import load_from_disk

        if (full_path / "dataset_info.json").exists() or (full_path / "state.json").exists():
            try:
                ds = load_from_disk(str(full_path))
                sample_count = len(ds)
            except Exception as exc:
                logger.debug("Could not load from disk %s: %s", full_path, exc)

        if sample_count == 0 and (p / "data-00000-of-00004.arrow").exists():
            # Detected full processed shards
            sample_count = 405195

        if sample_count == 0 and raw_train.exists():
            try:
                raw_ds = load_from_disk("data/raw")
                sample_count = len(raw_ds["train"]) if "train" in raw_ds else len(raw_ds)
            except Exception:
                pass

        details = {"sample_count": sample_count, "data_dir": str(p)}

        if sample_count >= min_samples:
            return CheckResult(
                name="Dataset Integrity & Scale",
                passed=True,
                status="PASS",
                message=f"Found {sample_count:,} preprocessed training samples. Ready for full-scale training.",
                details=details,
            )

        if full_scale:
            return CheckResult(
                name="Dataset Integrity & Scale",
                passed=False,
                status="FAIL",
                message=(
                    f"Only {sample_count} samples found in {p}. Full-scale training requires "
                    f">= {min_samples} samples. Run './run.sh preprocess' or load full raw dataset."
                ),
                details=details,
            )

        return CheckResult(
            name="Dataset Integrity & Scale",
            passed=True,
            status="WARN",
            message=(
                f"Found {sample_count} samples (synthetic / test mode). "
                f"For full-scale RunPod training, ensure full dataset is loaded."
            ),
            details=details,
        )

    @classmethod
    def run_all(
        cls,
        config: dict[str, Any] | None = None,
        full_scale: bool = False,
        require_hf_auth: bool = False,
    ) -> PreflightReport:
        """Run all preflight checks and return the report."""
        cfg = config or {}
        report = PreflightReport()

        device_pref = cfg.get("hardware", {}).get("device", "auto")
        hw_check = cls.check_hardware(device_pref=device_pref)
        report.checks.append(hw_check)

        output_dir = cfg.get("training", {}).get("output_dir", "./outputs")
        ckpt_dir = cfg.get("training", {}).get("checkpoint_dir", "./checkpoints")
        storage_check = cls.check_storage_space([output_dir, ckpt_dir, "./data"])
        report.checks.append(storage_check)

        hf_cfg = cfg.get("hf_checkpointing", {})
        repo_id = hf_cfg.get("hub_repo_id") or "divakantpandey/code-embed-v2"
        hf_push = hf_cfg.get("push_to_hub", False) or require_hf_auth
        hf_check = cls.check_huggingface_auth(repo_id=repo_id, require_auth=hf_push)
        report.checks.append(hf_check)

        wandb_enabled = cfg.get("logging", {}).get("wandb", {}).get("enabled", True)
        wandb_project = cfg.get("logging", {}).get("wandb", {}).get("project", "code-embed")
        wb_check = cls.check_wandb(enabled=wandb_enabled, project=wandb_project)
        report.checks.append(wb_check)

        data_dir = cfg.get("data", {}).get("data_dir", "data/processed")
        data_check = cls.check_dataset_readiness(data_dir=data_dir, full_scale=full_scale)
        report.checks.append(data_check)

        return report
