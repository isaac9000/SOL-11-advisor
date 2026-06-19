#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

echo "=== RoPE advisor-worker optimization ==="
echo "Deploying evaluator (no-op if already deployed)..."
uv run modal deploy eval_modal_rope.py

echo ""
echo "Launching agent..."
uv run rope/agent.py \
    --baseline rope/starting_point.py \
    --iterations 25 \
    "$@"
