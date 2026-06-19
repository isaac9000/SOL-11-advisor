# RoPE Autoresearch

An advisor-worker agent pair that iteratively optimizes a CUDA kernel for Rotary Position Embeddings (RoPE) with Llama3 scaling on NVIDIA B200. Each iteration the **advisor** reviews experiment history and proposes a strategic direction; the **worker** implements exactly one change to `submission.py`, evaluates it on a B200 via Modal, logs the result, and stops. The outer loop drives the next iteration.

## Task

Implement the fastest possible **RoPE (Rotary Position Embedding)** kernel with Llama3 scaling (NVIDIA SOL-ExecBench):

```
inv_freq_exp = inv_freq[None, :, None].float().expand(bs, 64, 1)  # [bs, 64, 1]
pos_exp      = position_ids[:, None, :].float()                    # [bs, 1, seq_len]
freqs        = (inv_freq_exp @ pos_exp).transpose(1, 2)            # [bs, seq_len, 64]
emb          = cat((freqs, freqs), dim=-1)                         # [bs, seq_len, 128]
cos_sin      = stack([emb.cos(), emb.sin()], dim=-1)               # [bs, seq_len, 128, 2]
return cos_sin.to(bfloat16)
```

`run` receives keyword arguments and returns a single tensor:

| Argument | Shape | Dtype |
|---|---|---|
| `position_ids` | `[batch_size, seq_len]` | int64 |
| `inv_freq` | `[64]` | float32 |
| `attention_scaling` | scalar | float32 (always 1.0) |
| return `cos_sin` | `[batch_size, seq_len, 128, 2]` | bfloat16 |

Fixed architecture: head_dim=128, half_head_dim=64, attention_scaling=1.0.

**Benchmark cases (16 total):**

| # | bs | seq_len | Baseline (μs) | SOL (μs) |
|---|-----|---------|---------------|----------|
| 1 | 16  | 256 | 6.2 | 0.7 |
| 2 | 1   | 2048 | 4.3 | 0.5 |
| 3 | 1   | 613 | 2.7 | 0.4 |
| 4 | 1   | 1024 | 3.4 | 0.5 |
| 5 | 16  | 373 | 6.5 | 0.8 |
| 6 | 16  | 919 | 12.6 | 1.4 |
| 7 | 2   | 131 | 2.6 | 0.4 |
| 8 | 1   | 256 | 2.6 | 0.4 |
| 9 | 4   | 256 | 3.3 | 0.5 |
| 10 | 4  | 2048 | 9.8 | 0.9 |
| 11 | 4  | 1024 | 6.3 | 0.7 |
| 12 | 4  | 211 | 2.7 | 0.5 |
| 13 | 1  | 512 | 2.8 | 0.4 |
| 14 | 64 | 256 | 17.4 | 1.5 |
| 15 | 64 | 541 | 34.1 | 2.7 |
| 16 | 32 | 128 | 6.2 | 0.7 |

All 16 cases are used for both correctness testing and benchmarking. Correctness tolerance: `rtol=1e-2, atol=1e-2`. Score = `5.53 / geomean_μs` (≈1.0 at baseline, ≈8.1 at SOL).

## Setup

```bash
uv sync

# Configure Modal credentials
uv run modal token set --token-id <token-id> --token-secret <token-secret>

# Deploy the B200 evaluator (once, before any agent runs)
uv run modal deploy eval_modal_rope.py
```

Create a `.env` file in the repo root:

```
ANTHROPIC_API_KEY=...
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
AUTORESEARCH_MODEL=claude-sonnet-4-6   # optional, this is the default
```

## Running the agent

```bash
bash run_agent.sh
```

Or directly:

```bash
uv run rope/agent.py --baseline rope/starting_point.py --iterations 25
```

Quick correctness check without a full benchmark:

```bash
cd rope
uv run python run_eval.py submission.py -o results.json --mode test
```

## Structure

```
eval_modal_rope.py   — deployable Modal B200 evaluator
rope/
├── agent.py             — advisor-worker agentic loop (direct Anthropic SDK)
├── advisor_prompt.md    — advisor system prompt: strategy, comparison discipline
├── worker_prompt.md     — worker system prompt: task spec, mandatory sequence, rules
├── submission.py        — the kernel file the worker edits each iteration
├── starting_point.py    — baseline PyTorch kernel to seed each run
├── run_eval.py          — submits submission.py to the deployed Modal evaluator
├── tools.py             — logging, plotting, and get_experiment_history tool
└── runs/                — one directory per run: history, TSV log, plots, best submission
```

Each run directory contains:
- `experiment_history.md` — full log of every attempt with code and result
- `results.tsv` — tab-separated summary for plotting
- `progress.png` — latency scatter plot updated each experiment
- `iterations.png` — best latency per advisor iteration
- `best_submission.py` — snapshot of the fastest kernel found so far
- `proposals.md` — advisor proposals for every iteration
- `snapshot_iter{N}.py` — per-iteration snapshot of submission.py before the worker edits it
