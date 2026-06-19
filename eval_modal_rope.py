"""
Deployable Modal B200 evaluator for the RoPE (Rotary Position Embedding) task.

Computes cos/sin embeddings with Llama3 scaling from precomputed inv_freq
and position_ids, returning [batch_size, seq_len, head_dim, 2] bfloat16.

Deploy once:
    uv run modal deploy eval_modal_rope.py

Then the agent's run_eval.py calls evaluate_kernel.remote(kernel_code).
"""

import modal

# 16 benchmark workloads. All cases: head_dim=128, half_head_dim=64, attention_scaling=1.0
CASES = [
    {"batch_size": 16, "seq_len": 256},
    {"batch_size": 1,  "seq_len": 2048},
    {"batch_size": 1,  "seq_len": 613},
    {"batch_size": 1,  "seq_len": 1024},
    {"batch_size": 16, "seq_len": 373},
    {"batch_size": 16, "seq_len": 919},
    {"batch_size": 2,  "seq_len": 131},
    {"batch_size": 1,  "seq_len": 256},
    {"batch_size": 4,  "seq_len": 256},
    {"batch_size": 4,  "seq_len": 2048},
    {"batch_size": 4,  "seq_len": 1024},
    {"batch_size": 4,  "seq_len": 211},
    {"batch_size": 1,  "seq_len": 512},
    {"batch_size": 64, "seq_len": 256},
    {"batch_size": 64, "seq_len": 541},
    {"batch_size": 32, "seq_len": 128},
]

TEST_CASES = CASES
BENCHMARK_CASES = CASES

HEAD_DIM = 128
HALF_HEAD_DIM = 64
ATTENTION_SCALING = 1.0

# Scoring: score = SCORE_SCALE / geomean_us (≈1.0 at baseline ≈5.53 μs, ≈8.1 at SOL ≈0.68 μs)
SCORE_SCALE = 5.53

BENCH_USE_CUDA_EVENTS = True
BENCH_REL_ERROR       = 0.001       # stop when stderr/mean < 0.1%
BENCH_WALL_TIMEOUT_NS = 120e9
BENCH_NO_GRAD         = True
BENCH_MAX_REPEATS     = 100
BENCH_MAX_TIME_NS     = 10e9

# B200 (Blackwell sm_100) requires CUDA 12.8+ and PyTorch 2.6+.
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel",
        add_python="3.11",
    )
    .pip_install("triton")
)

app = modal.App("rope-kernel-eval")


@app.function(gpu="B200", image=image, timeout=600)
def evaluate_kernel(kernel_code: str, mode: str = "leaderboard") -> str:
    import contextlib
    import gc
    import importlib.util
    import json as _json
    import math
    import os as _os
    import tempfile
    import time
    import traceback

    import torch

    _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    gpu_name  = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
    torch_ver = torch.__version__

    # ── Reference input generation ────────────────────────────────────────────

    def ref_get_inputs(batch_size: int, seq_len: int) -> dict:
        rope_theta = 500000.0
        factor = 8.0
        low_freq_factor = 1.0
        high_freq_factor = 4.0
        original_max_position_embeddings = 8192

        dim_indices = torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device="cuda")
        inv_freq = 1.0 / (rope_theta ** (dim_indices / HEAD_DIM))

        wavelens = 2 * math.pi / inv_freq
        smooth_factor = (
            original_max_position_embeddings / wavelens - low_freq_factor
        ) / (high_freq_factor - low_freq_factor)
        smooth_factor = torch.clamp(smooth_factor, 0.0, 1.0)

        scaled_inv_freq = inv_freq / factor
        inv_freq = (1 - smooth_factor) * inv_freq + smooth_factor * scaled_inv_freq

        position_ids = (
            torch.arange(seq_len, dtype=torch.int64, device="cuda")
            .unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
        )

        return {
            "position_ids": position_ids,
            "inv_freq": inv_freq,
            "attention_scaling": ATTENTION_SCALING,
        }

    # ── Reference implementation ──────────────────────────────────────────────

    def ref_run(position_ids, inv_freq, attention_scaling):
        batch_size = position_ids.shape[0]
        inv_freq_expanded = inv_freq[None, :, None].float().expand(batch_size, -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * attention_scaling
        sin = emb.sin() * attention_scaling
        cos_sin = torch.stack([cos, sin], dim=-1)
        return cos_sin.to(dtype=torch.bfloat16)

    # ── Correctness check ─────────────────────────────────────────────────────

    def check_output(ref_out, sub_out, rtol=1e-2, atol=1e-2):
        if ref_out.shape != sub_out.shape:
            return False, f"shape mismatch: ref={ref_out.shape} sub={sub_out.shape}"
        ok = torch.allclose(ref_out.float(), sub_out.float(), rtol=rtol, atol=atol)
        if not ok:
            d = torch.abs(ref_out.float() - sub_out.float())
            return False, f"mismatch: max={d.max().item():.4e} mean={d.mean().item():.4e}"
        return True, "Match"

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _stats(durations):
        n = len(durations)
        avg = sum(durations) / n
        if n > 1:
            var = sum((x - avg) ** 2 for x in durations) / (n - 1)
            std = math.sqrt(var)
            err = std / math.sqrt(n)
        else:
            std, err = 0.0, 0.0
        return {"runs": n, "mean": avg, "std": std, "err": err}

    def clear_l2_cache():
        dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda")
        dummy.fill_(42)
        del dummy

    # ── Load submission ───────────────────────────────────────────────────────

    tmp_dir  = tempfile.mkdtemp(prefix="submission_")
    tmp_path = _os.path.join(tmp_dir, "submission.py")
    with open(tmp_path, "w") as f:
        f.write(kernel_code)

    try:
        spec = importlib.util.spec_from_file_location("submission", tmp_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sub_run = mod.run
    except Exception:
        return _json.dumps({
            "success": False,
            "error": f"Failed to load submission:\n{traceback.format_exc()}",
            "tests_passed": 0,
            "tests_total": len(TEST_CASES),
            "test_details": [],
            "gpu_name": gpu_name,
            "torch_version": torch_ver,
            "platform": "modal-b200",
            "failure_stage": "import",
        })

    # ── Correctness tests ─────────────────────────────────────────────────────

    test_details = []
    tests_passed = 0

    for tc in TEST_CASES:
        bs, sl = tc["batch_size"], tc["seq_len"]
        try:
            inputs  = ref_get_inputs(bs, sl)
            ref_out = ref_run(**inputs)
            torch.cuda.synchronize()

            with torch.no_grad():
                sub_out = sub_run(**inputs)
            torch.cuda.synchronize()

            passed, msg = check_output(ref_out, sub_out)
            del ref_out, sub_out
            gc.collect()
            torch.cuda.empty_cache()

            test_details.append({
                "batch_size": bs, "seq_len": sl,
                "passed": passed, "error": "" if passed else msg,
            })
            if passed:
                tests_passed += 1
        except Exception:
            test_details.append({
                "batch_size": bs, "seq_len": sl,
                "passed": False, "error": traceback.format_exc()[:600],
            })

    if tests_passed < len(TEST_CASES):
        return _json.dumps({
            "success": False,
            "tests_passed": tests_passed,
            "tests_total": len(TEST_CASES),
            "test_details": test_details,
            "error": "Correctness check failed — see test_details",
            "gpu_name": gpu_name,
            "torch_version": torch_ver,
            "platform": "modal-b200",
            "failure_stage": "correctness",
        })

    if mode == "test":
        return _json.dumps({
            "success": True,
            "tests_passed": tests_passed,
            "tests_total": len(TEST_CASES),
            "test_details": test_details,
            "gpu_name": gpu_name,
            "torch_version": torch_ver,
            "platform": "modal-b200",
        })

    # ── Benchmarks ────────────────────────────────────────────────────────────

    ctx = torch.no_grad() if BENCH_NO_GRAD else contextlib.nullcontext()
    benchmark_details = []
    bench_means_ns    = []

    for bench_args in BENCHMARK_CASES:
        bs, sl = bench_args["batch_size"], bench_args["seq_len"]
        inputs = ref_get_inputs(bs, sl)

        # Correctness re-check before timing
        with ctx:
            ref_out = ref_run(**inputs)
            sub_out = sub_run(**inputs)
            torch.cuda.synchronize()
            passed, msg = check_output(ref_out, sub_out)
            del ref_out, sub_out
            gc.collect()
            torch.cuda.empty_cache()

        if not passed:
            return _json.dumps({
                "success": False,
                "tests_passed": tests_passed,
                "tests_total": len(TEST_CASES),
                "test_details": test_details,
                "error": f"Benchmark correctness: {msg}",
                "gpu_name": gpu_name,
                "torch_version": torch_ver,
                "platform": "modal-b200",
                "failure_stage": "benchmark",
            })

        # Warmup
        for _ in range(3):
            sub_run(**inputs)
            torch.cuda.synchronize()

        durations_ns = []
        bm_start = time.perf_counter_ns()

        with ctx:
            for t in range(BENCH_MAX_REPEATS):
                clear_l2_cache()
                torch.cuda.synchronize()

                if BENCH_USE_CUDA_EVENTS:
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    output = sub_run(**inputs)
                    e.record()
                    torch.cuda.synchronize()
                    duration_ns = s.elapsed_time(e) * 1e6  # ms → ns
                else:
                    t0 = time.perf_counter_ns()
                    output = sub_run(**inputs)
                    torch.cuda.synchronize()
                    duration_ns = time.perf_counter_ns() - t0

                del output
                durations_ns.append(duration_ns)

                if t > 1:
                    st = _stats(durations_ns)
                    if st["mean"] > 0 and st["err"] / st["mean"] < BENCH_REL_ERROR:
                        break
                    if st["mean"] * st["runs"] > BENCH_MAX_TIME_NS:
                        break
                    if (time.perf_counter_ns() - bm_start) > BENCH_WALL_TIMEOUT_NS:
                        break

        st     = _stats(durations_ns)
        mean_us = st["mean"] / 1e3
        err_us  = st["err"] / 1e3
        benchmark_details.append({
            "batch_size": bs,
            "seq_len":    sl,
            "mean_us":    round(mean_us, 3),
            "err_us":     round(err_us, 3),
            "runs":       st["runs"],
        })
        bench_means_ns.append(st["mean"])

    means_s    = [ns / 1e9 for ns in bench_means_ns]
    geomean_s  = math.pow(math.prod(means_s), 1.0 / len(means_s))
    geomean_us = geomean_s * 1e6
    score      = SCORE_SCALE / geomean_us

    return _json.dumps({
        "success":          True,
        "tests_passed":     tests_passed,
        "tests_total":      len(TEST_CASES),
        "test_details":     test_details,
        "benchmark": {
            "geomean_us": round(geomean_us, 3),
            "score":      round(score, 3),
        },
        "benchmark_details": benchmark_details,
        "gpu_name":          gpu_name,
        "torch_version":     torch_ver,
        "platform":          "modal-b200",
    })
