#!/usr/bin/env bash
# ==============================================================================
# CodeEmbed: One-Click Setup & Run Script
# ==============================================================================
set -e

# Change to script's directory (repo root)
cd "$(dirname "$0")"

echo "=================================================="
echo "💻 CodeEmbed — One-Click Setup & Runner"
echo "=================================================="

# 1. Check & Install uv if missing
if ! command -v uv &> /dev/null; then
    echo "📦 uv not found. Installing uv (fast Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if ! command -v uv &> /dev/null; then
    echo "❌ Failed to install uv automatically. Please install uv manually: https://docs.astral.sh/uv/"
    exit 1
fi

echo "✅ uv is installed: $(uv --version)"

# 2. Environment Initialization (.env)
if [ ! -f .env ] && [ -f .env.example ]; then
    echo "⚙️  Creating default .env from .env.example..."
    cp .env.example .env
fi

# 3. Dependency Sync & Virtualenv Setup
echo "🔄 Installing dependencies with uv..."
uv sync --group dev

# 4. Command Router
COMMAND="${1:-demo}"

# Check if profiling is explicitly requested via subcommand or flag
if [ "$COMMAND" = "profile" ] || [ "$COMMAND" = "--profile" ]; then
    TARGET="${2:-demo}"
    echo "⚡ Profiling '$TARGET' with Scalene profiler..."
    exec uv run python scripts/profile_runner.py run "$TARGET" "${@:3}"
fi

if [ "$COMMAND" = "view-profile" ] || [ "$COMMAND" = "view" ]; then
    TARGET="${2}"
    echo "🌐 Opening profile viewer for '$TARGET'..."
    exec uv run python scripts/profile_runner.py view "$TARGET" "${@:3}"
fi

if [ "$COMMAND" = "list-profiles" ]; then
    exec uv run python scripts/profile_runner.py list
fi

# If PROFILE=1 or PROFILE=true is set in environment, route via profile_runner
if [ "$PROFILE" = "1" ] || [ "$PROFILE" = "true" ]; then
    echo "⚡ PROFILE=1 detected — routing '$COMMAND' via Scalene profile runner..."
    exec uv run python scripts/profile_runner.py run "$COMMAND" "${@:2}"
fi

case "$COMMAND" in
    demo)
        echo "🎮 Starting interactive demo..."
        uv run python scripts/demo.py "${@:2}"
        ;;
    train)
        echo "🚀 Starting 3-stage training pipeline..."
        uv run python scripts/train.py --config configs/train_config.yaml "${@:2}"
        ;;
    tune)
        echo "🔧 Starting hyperparameter tuning..."
        uv run python scripts/tune.py --config configs/train_config.yaml "${@:2}"
        ;;
    preprocess)
        echo "🔧 Running data preprocessing pipeline..."
        uv run python scripts/preprocess.py "${@:2}"
        ;;
    download)
        echo "📥 Downloading datasets..."
        uv run python scripts/download_data.py "${@:2}"
        ;;
    evaluate|eval)
        echo "📊 Running evaluation benchmarks..."
        uv run python scripts/evaluate.py --config configs/eval_config.yaml "${@:2}"
        ;;
    export)
        echo "📦 Exporting to ONNX and applying INT8 quantization..."
        uv run python scripts/export.py "${@:2}"
        ;;
    test)
        echo "🧪 Running unit tests..."
        uv run pytest tests/ -v "${@:2}"
        ;;
    lint)
        echo "🔍 Running code checks..."
        uv run ruff check src/ scripts/ tests/
        uv run ruff format --check src/ scripts/ tests/
        ;;
    push)
        echo "🚀 Pushing model to Hugging Face Hub..."
        uv run python scripts/push_to_hub.py "${@:2}"
        ;;
    all)
        echo "🚀 Running end-to-end pipeline: Preprocess -> Train -> Evaluate -> Export..."
        uv run python scripts/preprocess.py
        uv run python scripts/train.py --config configs/train_config.yaml
        uv run python scripts/evaluate.py --config configs/eval_config.yaml
        uv run python scripts/export.py
        ;;
    *)
        echo "Usage: ./run.sh [demo|train|tune|preprocess|download|evaluate|export|push|test|lint|all]"
        echo "       ./run.sh profile [demo|train|evaluate|preprocess|export|tune|all]"
        echo "       ./run.sh view-profile [run_name]"
        echo "       PROFILE=1 ./run.sh [command]"
        exit 1
        ;;
esac

