#!/bin/bash
mkdir -p /workspace && cd /workspace
exec >> /workspace/onstart.log 2>&1
echo "boot $(date -u)"
command -v git >/dev/null && command -v curl >/dev/null || { apt-get update -qq && apt-get install -y -qq git curl ca-certificates; }
[ -x "$HOME/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH" HF_HOME=/workspace/.cache/huggingface UV_LINK_MODE=copy
[ -d code ] || git clone -q https://github.com/divakantpandey/code-embed-v3 code
git -C code checkout -q d34068d
for f in run.sh train_v2zs.py eval_v2zs.py v2_zeroshot.yaml; do
  curl -fsSL -H "Authorization: Bearer $HF_TOKEN" -o /workspace/$f \
    https://huggingface.co/divakantpandey/code-embed-v3-src/resolve/main/v2-zeroshot/src/$f
done
(cd code && uv python install 3.12 && uv sync --frozen)
nvidia-smi
nohup bash /workspace/run.sh >> /workspace/run.log 2>&1 &
echo "started run.sh pid $!"
