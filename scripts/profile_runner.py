"""Scalene profiling runner and report generator for PyCodeEmbed.

Profiles repository runs (preprocess, train, evaluate, export, demo, etc.),
generates JSON profiles, standalone interactive HTML reports, and CLI summaries
in the 'profiling_reports/' directory, and enables viewing reports in a web browser.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger("profile_runner")
console = Console()

REPORTS_DIR = Path("profiling_reports")
CONFIG_FILE = Path("configs/scalene.yaml")

# Pre-defined run commands and their default execution targets
RUN_DEFINITIONS: dict[str, dict[str, Any]] = {
    "demo": {
        "description": "Interactive demo with a semantic search query",
        "script": "scripts/demo.py",
        "default_args": ["--query", "binary search"],
    },
    "download": {
        "description": "Download dataset samples",
        "script": "scripts/download_data.py",
        "default_args": [],
    },
    "preprocess": {
        "description": "Data preprocessing pipeline",
        "script": "scripts/preprocess.py",
        "default_args": ["--max-samples", "100"],
    },
    "train": {
        "description": "Full 3-stage training pipeline",
        "script": "scripts/train.py",
        "default_args": [
            "--config",
            "configs/train_config.yaml",
            "--max-samples",
            "32",
            "--epochs",
            "1",
        ],
    },
    "train-stage1": {
        "description": "Stage 1: SimCSE contrastive pre-training",
        "script": "scripts/train.py",
        "default_args": [
            "--config",
            "configs/train_config.yaml",
            "--stage",
            "1",
            "--max-samples",
            "32",
            "--epochs",
            "1",
        ],
    },
    "train-stage2": {
        "description": "Stage 2: Supervised MNRL fine-tuning",
        "script": "scripts/train.py",
        "default_args": [
            "--config",
            "configs/train_config.yaml",
            "--stage",
            "2",
            "--max-samples",
            "32",
            "--epochs",
            "1",
        ],
    },
    "train-stage3": {
        "description": "Stage 3: Matryoshka Representation Learning",
        "script": "scripts/train.py",
        "default_args": [
            "--config",
            "configs/train_config.yaml",
            "--stage",
            "3",
            "--max-samples",
            "32",
            "--epochs",
            "1",
        ],
    },
    "tune": {
        "description": "Hyperparameter optimization with Optuna",
        "script": "scripts/tune.py",
        "default_args": [
            "--config",
            "configs/train_config.yaml",
            "--n-trials",
            "2",
            "--epochs-per-trial",
            "1",
            "--val-samples",
            "20",
        ],
    },
    "tune-fast": {
        "description": "Fast hyperparameter sweep",
        "script": "scripts/tune.py",
        "default_args": [
            "--config",
            "configs/train_config.yaml",
            "--n-trials",
            "1",
            "--epochs-per-trial",
            "1",
            "--val-samples",
            "20",
        ],
    },
    "evaluate": {
        "description": "Full benchmark evaluation suite",
        "script": "scripts/evaluate.py",
        "default_args": ["--config", "configs/eval_config.yaml", "--max-samples", "15"],
    },
    "export": {
        "description": "Export model to ONNX and INT8 quantization",
        "script": "scripts/export.py",
        "default_args": [],
    },
    "test": {
        "description": "Pytest test suite (fast unit tests)",
        "script": "-m",
        "default_args": ["pytest", "tests/", "-m", "not slow and not gpu"],
    },
}


def ensure_reports_dir(directory: Path = REPORTS_DIR) -> Path:
    """Ensure report directory exists."""
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def build_scalene_cmd(
    target_script: str,
    target_args: list[str],
    json_output_path: Path,
    *,
    cpu_only: bool = True,
    no_stacks: bool = True,
    profile_only: str = "src,scripts",
) -> list[str]:
    """Build the Scalene CLI command line."""
    cmd = [sys.executable, "-m", "scalene", "run"]

    if CONFIG_FILE.exists():
        cmd.extend(["-c", str(CONFIG_FILE)])
    else:
        if cpu_only or sys.platform == "darwin":
            cmd.append("--cpu-only")
        if no_stacks or sys.platform == "darwin":
            cmd.append("--no-stacks")
        if profile_only:
            cmd.extend(["--profile-only", profile_only])

    cmd.extend(["-o", str(json_output_path)])

    if target_script == "-m":
        cmd.append("-m")
        if target_args:
            cmd.append(target_args[0])
            if len(target_args) > 1:
                cmd.append("---")
                cmd.extend(target_args[1:])
    else:
        cmd.append(target_script)
        if target_args:
            cmd.append("---")
            cmd.extend(target_args)

    return cmd


def generate_local_reports(
    json_path: Path,
    run_name: str,
    reports_dir: Path = REPORTS_DIR,
) -> tuple[Path, Path]:
    """Generate standalone HTML and text summary reports from JSON profile."""
    html_path = reports_dir / f"{run_name}_profile.html"
    summary_path = reports_dir / f"{run_name}_summary.txt"

    # 1. Generate standalone HTML report
    try:
        from scalene.scalene_parseargs import Filename, generate_html

        generate_html(
            profile_fname=Filename(str(json_path)),
            output_fname=Filename(str(html_path)),
            standalone=True,
        )
        logger.info("Generated standalone HTML report: %s", html_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to generate HTML report with generate_html: %s", exc)
        # Fallback to scalene view --standalone if needed
        try:
            subprocess.run(
                [sys.executable, "-m", "scalene", "view", str(json_path), "--standalone"],
                check=True,
                capture_output=True,
            )
            default_html = Path("scalene-profile.html")
            if default_html.exists():
                shutil.move(str(default_html), str(html_path))
        except Exception as inner_exc:  # noqa: BLE001
            logger.error("HTML generation fallback failed: %s", inner_exc)

    # 2. Generate CLI Text Summary
    try:
        from scalene.scalene_parseargs import ScaleneParseArgs

        with open(json_path, encoding="utf-8") as f:
            profile_data = json.load(f)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ScaleneParseArgs._display_profile_cli(
                profile_data, column_width=132, reduced_profile=False
            )

        summary_text = buf.getvalue()
        summary_path.write_text(summary_text, encoding="utf-8")
        logger.info("Generated text summary: %s", summary_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to generate text summary: %s", exc)

    # 3. Update dashboard and README catalog
    update_reports_index(reports_dir)

    return html_path, summary_path


def update_reports_index(reports_dir: Path = REPORTS_DIR) -> None:
    """Create or update index.html and README.md in the profiling reports directory."""
    json_files = sorted(reports_dir.glob("*_profile.json"))

    readme_content = [
        "# ⚡ PyCodeEmbed Profiling Reports",
        "",
        "This directory stores performance profiles (CPU, Memory, and execution breakdown) generated by [Scalene](https://github.com/plasma-umass/scalene).",
        "",
        "## Available Run Reports",
        "",
        "| Run Name | Date (UTC) | JSON Profile | Standalone HTML | Summary Text |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    dashboard_rows = []

    for jf in json_files:
        run_name = jf.stem.replace("_profile", "")
        html_file = reports_dir / f"{run_name}_profile.html"
        txt_file = reports_dir / f"{run_name}_summary.txt"

        mtime = datetime.fromtimestamp(jf.stat().st_mtime, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        html_link = f"[{html_file.name}]({html_file.name})" if html_file.exists() else "—"
        txt_link = f"[{txt_file.name}]({txt_file.name})" if txt_file.exists() else "—"

        readme_content.append(
            f"| `{run_name}` | {mtime} | [`{jf.name}`]({jf.name}) | {html_link} | {txt_link} |"
        )

        dashboard_rows.append(f"""
        <tr>
            <td><strong>{run_name}</strong></td>
            <td>{mtime}</td>
            <td><a href="{html_file.name}" class="btn" target="_blank">Open Web Report</a></td>
            <td><a href="{jf.name}" download>Download JSON</a></td>
            <td>{f'<a href="{txt_file.name}" target="_blank">View Text</a>' if txt_file.exists() else "—"}</td>
        </tr>""")

    readme_content.extend(
        [
            "",
            "## How to View Profiles in Web",
            "",
            "### 1. Interactive Web Server (Recommended)",
            "Launch Scalene's interactive browser interface for any run:",
            "```bash",
            "# Via profile runner script",
            "uv run python scripts/profile_runner.py view <run_name>",
            "",
            "# Via Makefile",
            "make profile-view RUN=<run_name>",
            "",
            "# Via run.sh",
            "./run.sh view-profile <run_name>",
            "",
            "# Or directly with scalene",
            "uv run scalene view profiling_reports/<run_name>_profile.json",
            "```",
            "This spins up a local server at `http://localhost:8088/` and opens your browser.",
            "",
            "### 2. Standalone HTML Reports (Zero-server)",
            "Each run produces a 100% self-contained HTML report with all styles and scripts embedded:",
            "```bash",
            "# On macOS",
            "open profiling_reports/<run_name>_profile.html",
            "",
            "# On Linux",
            "xdg-open profiling_reports/<run_name>_profile.html",
            "```",
            "",
            "### 3. Hosted Web UI (Drag & Drop)",
            "Open [https://plasma-umass.org/scalene-gui/](https://plasma-umass.org/scalene-gui/) and drag & drop any `profiling_reports/<run_name>_profile.json` file.",
            "",
            "### 4. Local Web Dashboard",
            "Start a local web server to browse all reports from one place:",
            "```bash",
            "python -m http.server 8000 --directory profiling_reports",
            "```",
            "Then visit `http://localhost:8000/`.",
        ]
    )

    (reports_dir / "README.md").write_text("\n".join(readme_content) + "\n", encoding="utf-8")

    # Write dashboard index.html
    html_dashboard = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>PyCodeEmbed Profiling Reports</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 40px; background: #0d1117; color: #c9d1d9; }}
        h1 {{ color: #58a6ff; }}
        p {{ color: #8b949e; font-size: 1.1em; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background: #161b22; border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #30363d; }}
        th {{ background: #21262d; color: #f0f6fc; }}
        a {{ color: #58a6ff; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .btn {{ display: inline-block; padding: 6px 12px; background: #238636; color: #fff; border-radius: 6px; font-weight: bold; text-decoration: none; }}
        .btn:hover {{ background: #2ea043; text-decoration: none; }}
        code {{ background: #21262d; padding: 2px 6px; border-radius: 4px; font-family: monospace; }}
    </style>
</head>
<body>
    <h1>⚡ PyCodeEmbed — Scalene Profiling Reports</h1>
    <p>Select any run below to open its interactive performance breakdown in your browser.</p>
    <table>
        <thead>
            <tr>
                <th>Run Name</th>
                <th>Generated At (UTC)</th>
                <th>Interactive Report</th>
                <th>JSON Profile</th>
                <th>Text Summary</th>
            </tr>
        </thead>
        <tbody>
            {"".join(dashboard_rows) if dashboard_rows else '<tr><td colspan="5" style="text-align: center; color: #8b949e;">No profiles generated yet. Run a profile using <code>make profile-train</code> or <code>./run.sh profile demo</code>.</td></tr>'}
        </tbody>
    </table>
</body>
</html>"""
    (reports_dir / "index.html").write_text(html_dashboard, encoding="utf-8")


def profile_run(
    run_name: str,
    extra_args: list[str] | None = None,
    reports_dir: Path = REPORTS_DIR,
    *,
    view_after: bool = False,
) -> int:
    """Execute a profiled run and generate all local reports."""
    ensure_reports_dir(reports_dir)

    # 1. Resolve run definition
    if run_name in RUN_DEFINITIONS:
        defn = RUN_DEFINITIONS[run_name]
        script = defn["script"]
        args = list(defn["default_args"])
        if extra_args:
            args.extend(extra_args)
    elif run_name == "all":
        console.print(
            "[bold cyan]🚀 Running and profiling full end-to-end pipeline (preprocess -> train -> evaluate -> export)...[/]"
        )
        stages = ["preprocess", "train", "evaluate", "export"]
        for st in stages:
            code = profile_run(st, reports_dir=reports_dir)
            if code != 0:
                console.print(
                    f"[bold red]❌ Pipeline stage '{st}' failed with code {code}. Stopping.[/]"
                )
                return code
        console.print("[bold green]✅ All pipeline stages profiled successfully![/]")
        if view_after:
            view_profile("train", reports_dir=reports_dir)
        return 0
    elif Path(run_name).exists():
        script = run_name
        args = extra_args or []
        run_name = Path(run_name).stem
    else:
        console.print(f"[bold red]❌ Unknown run name or script: '{run_name}'[/]")
        console.print(
            f"Supported predefined runs: {', '.join(RUN_DEFINITIONS.keys())}, 'all', or path to any .py file."
        )
        return 1

    json_output = reports_dir / f"{run_name}_profile.json"

    console.print(f"\n[bold green]⚡ Profiling '{run_name}' with Scalene...[/]")
    console.print(f"[dim]Script: {script} | Args: {' '.join(args)}[/]")

    cmd = build_scalene_cmd(
        target_script=script,
        target_args=args,
        json_output_path=json_output,
    )

    env = os.environ.copy()
    env["SCALENE_PROFILING"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["NUMBA_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    if sys.platform == "darwin":
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # Execute
    res = subprocess.run(cmd, env=env)

    if res.returncode != 0:
        console.print(f"[bold red]⚠️ Scalene run finished with return code {res.returncode}[/]")
    else:
        console.print(f"[bold green]✅ Scalene profile completed successfully for '{run_name}'[/]")

    # Generate standalone HTML and summary reports if profile was generated
    if json_output.exists() and json_output.stat().st_size > 0:
        html_path, summary_path = generate_local_reports(
            json_path=json_output,
            run_name=run_name,
            reports_dir=reports_dir,
        )
        console.print(f"[bold cyan]📁 Local reports created in '{reports_dir}':[/]")
        console.print(f"  • JSON Profile:     [dim]{json_output}[/]")
        console.print(f"  • Standalone HTML:  [green]{html_path}[/]")
        console.print(f"  • Text Summary:     [dim]{summary_path}[/]")
        console.print("\n[bold yellow]🌐 How to view in web:[/]")
        console.print(f"  1. Run:  [bold]uv run scalene view {json_output}[/]")
        console.print(f"     or:   [bold]make profile-view RUN={run_name}[/]")
        console.print(f"     or:   [bold]./run.sh view-profile {run_name}[/]")
        console.print(f"  2. Open standalone HTML:  [bold]open {html_path}[/]")
        console.print(
            "  3. Drag & drop JSON to:   [bold]https://plasma-umass.org/scalene-gui/[/]\n"
        )

        if view_after:
            view_profile(run_name, reports_dir=reports_dir)
    else:
        console.print(f"[bold red]⚠️ No profile file was generated at {json_output}[/]")

    return res.returncode


def view_profile(
    run_name: str | None = None,
    reports_dir: Path = REPORTS_DIR,
    *,
    mode: str = "web",
) -> int:
    """View an existing profile in web browser or terminal."""
    ensure_reports_dir(reports_dir)

    if not run_name:
        # Find the latest JSON file
        json_files = sorted(
            reports_dir.glob("*_profile.json"), key=lambda f: f.stat().st_mtime, reverse=True
        )
        if not json_files:
            console.print(
                f"[bold red]❌ No profiles found in '{reports_dir}'. Run a profiled command first.[/]"
            )
            return 1
        json_path = json_files[0]
        run_name = json_path.stem.replace("_profile", "")
    else:
        json_path = reports_dir / f"{run_name}_profile.json"
        if not json_path.exists():
            # Check if user passed full file name
            candidate = reports_dir / run_name
            if candidate.exists():
                json_path = candidate
            else:
                console.print(f"[bold red]❌ Profile file not found: {json_path}[/]")
                return 1

    html_path = reports_dir / f"{run_name}_profile.html"

    if mode == "html" or not shutil.which("scalene"):
        if html_path.exists():
            console.print(
                f"[bold cyan]Opening standalone HTML report in web browser: {html_path}[/]"
            )
            webbrowser.open(html_path.resolve().as_uri())
            return 0
        console.print("[dim]HTML report not found, launching scalene web server...[/]")

    if mode == "cli":
        txt_path = reports_dir / f"{run_name}_summary.txt"
        if txt_path.exists():
            print(txt_path.read_text(encoding="utf-8"))
            return 0
        cmd = [sys.executable, "-m", "scalene", "view", "--cli", str(json_path)]
        return subprocess.run(cmd).returncode

    console.print(f"[bold green]🌐 Launching Scalene interactive web viewer for '{run_name}'...[/]")
    console.print(f"[dim]Opening {json_path} (Press Ctrl+C to stop the viewer)[/]")
    cmd = [sys.executable, "-m", "scalene", "view", str(json_path)]
    try:
        return subprocess.run(cmd).returncode
    except KeyboardInterrupt:
        console.print("\n[dim]Viewer stopped.[/]")
        return 0


def list_profiles(reports_dir: Path = REPORTS_DIR) -> None:
    """List all available profile reports."""
    ensure_reports_dir(reports_dir)
    json_files = sorted(
        reports_dir.glob("*_profile.json"), key=lambda f: f.stat().st_mtime, reverse=True
    )

    if not json_files:
        console.print(f"[yellow]No profiling reports found in '{reports_dir}'.[/]")
        console.print(
            "Run a profile with: [bold]make profile-train[/] or [bold]./run.sh profile demo[/]"
        )
        return

    table = Table(title="⚡ PyCodeEmbed Available Profiling Reports")
    table.add_column("Run Name", style="bold cyan")
    table.add_column("Generated (UTC)", style="magenta")
    table.add_column("Size", style="green")
    table.add_column("HTML Report", style="blue")
    table.add_column("Summary", style="dim")

    for jf in json_files:
        run_name = jf.stem.replace("_profile", "")
        html_file = reports_dir / f"{run_name}_profile.html"
        txt_file = reports_dir / f"{run_name}_summary.txt"
        mtime = datetime.fromtimestamp(jf.stat().st_mtime, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        size_kb = f"{jf.stat().st_size / 1024:.1f} KB"

        html_status = "✅ Ready" if html_file.exists() else "—"
        txt_status = "✅ Ready" if txt_file.exists() else "—"

        table.add_row(run_name, mtime, size_kb, html_status, txt_status)

    console.print(table)


def main() -> None:
    """CLI Entry point."""
    parser = argparse.ArgumentParser(
        description="PyCodeEmbed Scalene Profiling Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")

    # Run subcommand
    run_parser = subparsers.add_parser("run", help="Profile a specific run")
    run_parser.add_argument(
        "run_name",
        type=str,
        help=f"Name of predefined run ({', '.join(RUN_DEFINITIONS.keys())}, 'all') or path to script",
    )
    run_parser.add_argument(
        "--view",
        "-v",
        action="store_true",
        help="Immediately launch web viewer after profiling",
    )
    run_parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Directory to save profiling reports",
    )
    run_parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments passed to the script (prefix with ---)",
    )

    # View subcommand
    view_parser = subparsers.add_parser("view", help="View an existing profile report")
    view_parser.add_argument(
        "run_name",
        type=str,
        nargs="?",
        default=None,
        help="Name of run to view (defaults to latest)",
    )
    view_parser.add_argument(
        "--mode",
        choices=["web", "html", "cli"],
        default="web",
        help="Viewing mode: 'web' (Scalene server), 'html' (standalone browser), 'cli' (terminal)",
    )
    view_parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Directory containing profiling reports",
    )

    # List subcommand
    list_parser = subparsers.add_parser("list", help="List all generated profile reports")
    list_parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Directory containing profiling reports",
    )

    args = parser.parse_args()

    if not args.command or args.command == "run":
        if not hasattr(args, "run_name") or not args.run_name:
            parser.print_help()
            sys.exit(1)
        # Strip leading '---' from extra_args if present
        extra = args.extra_args
        if extra and extra[0] == "---":
            extra = extra[1:]
        sys.exit(
            profile_run(
                args.run_name, extra_args=extra, reports_dir=args.reports_dir, view_after=args.view
            )
        )

    elif args.command == "view":
        sys.exit(view_profile(args.run_name, reports_dir=args.reports_dir, mode=args.mode))

    elif args.command == "list":
        list_profiles(reports_dir=args.reports_dir)
        sys.exit(0)


if __name__ == "__main__":
    main()
