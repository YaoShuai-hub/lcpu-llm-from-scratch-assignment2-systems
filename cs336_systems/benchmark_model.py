"""
benchmark_model.py
==================
Task 1 & 3: Single-GPU model benchmarking (forward + backward timing)
and mixed-precision (bfloat16 autocast) comparison.

Run:
    uv run python -m cs336_systems.benchmark_model
or with autocast:
    uv run python -m cs336_systems.benchmark_model --mixed-precision
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import statistics
import time

import torch

from cs336_basics.model import BasicsTransformerLM

# ---------------------------------------------------------------------------
# Model configurations (CS336 Assignment 2, Table in §2.1)
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "small": dict(
        d_model=768,
        num_heads=12,
        num_layers=12,
        d_ff=3072,
    ),
    "medium": dict(
        d_model=1024,
        num_heads=16,
        num_layers=24,
        d_ff=4096,
    ),
    "large": dict(
        d_model=1280,
        num_heads=20,
        num_layers=36,
        d_ff=5120,
    ),
    "xl": dict(
        d_model=1600,
        num_heads=25,
        num_layers=48,
        d_ff=6400,
    ),
    "2.7B": dict(
        d_model=2560,
        num_heads=32,
        num_layers=32,
        d_ff=10240,
    ),
}

# Common hyper-parameters
VOCAB_SIZE = 10_000
CONTEXT_LENGTH = 128
BATCH_SIZE = 4
ROPE_THETA = 10_000.0

NUM_WARMUP = 3
NUM_ITERS = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_model(cfg: dict, device: torch.device, dtype: torch.dtype = torch.float32) -> BasicsTransformerLM:
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=ROPE_THETA,
    )
    return model.to(device=device, dtype=dtype)


def make_inputs(device: torch.device) -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LENGTH), device=device)


def fmt(mean_ms: float, std_ms: float) -> str:
    return f"{mean_ms:8.2f} ± {std_ms:6.2f} ms"


# ---------------------------------------------------------------------------
# Core benchmarking function
# ---------------------------------------------------------------------------

def benchmark(
    name: str,
    model: BasicsTransformerLM,
    device: torch.device,
    use_autocast: bool = False,
    measure_no_warmup: bool = False,
) -> None:
    x = make_inputs(device)

    autocast_ctx = torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16) if use_autocast else contextlib.nullcontext()

    # ── Warm-up ────────────────────────────────────────────────────────────
    if not measure_no_warmup:
        for _ in range(NUM_WARMUP):
            with autocast_ctx:
                logits = model(x)
            loss = logits.sum()
            loss.backward()
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # ── Forward timing ─────────────────────────────────────────────────────
    fwd_times: list[float] = []
    for _ in range(NUM_ITERS):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            with autocast_ctx:
                logits = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        fwd_times.append((time.perf_counter() - t0) * 1e3)

    # ── Backward timing ────────────────────────────────────────────────────
    bwd_times: list[float] = []
    for _ in range(NUM_ITERS):
        model.zero_grad(set_to_none=True)
        with autocast_ctx:
            logits = model(x)
        loss = logits.sum()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        bwd_times.append((time.perf_counter() - t0) * 1e3)

    fwd_mean = statistics.mean(fwd_times)
    fwd_std = statistics.stdev(fwd_times) if len(fwd_times) > 1 else 0.0
    bwd_mean = statistics.mean(bwd_times)
    bwd_std = statistics.stdev(bwd_times) if len(bwd_times) > 1 else 0.0

    tag = " [mixed-precision]" if use_autocast else ""
    tag += " [NO warmup]" if measure_no_warmup else ""
    print(f"  {name:<8}{tag}")
    print(f"    Forward : {fmt(fwd_mean, fwd_std)}")
    print(f"    Backward: {fmt(bwd_mean, bwd_std)}")
    print()


# ---------------------------------------------------------------------------
# No-warm-up experiment (Task 1 extra question)
# ---------------------------------------------------------------------------

def benchmark_no_warmup(name: str, model: BasicsTransformerLM, device: torch.device) -> None:
    """Record the very first iteration without any warm-up."""
    x = make_inputs(device)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    logits = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    fwd_first = (time.perf_counter() - t0) * 1e3

    loss = logits.sum()
    t0 = time.perf_counter()
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    bwd_first = (time.perf_counter() - t0) * 1e3

    print(f"  {name:<8} [1st iter, NO warmup]")
    print(f"    Forward (1st) : {fwd_first:8.2f} ms")
    print(f"    Backward (1st): {bwd_first:8.2f} ms")
    print()


# ---------------------------------------------------------------------------
# Task 3: floating-point accumulation precision demo
# ---------------------------------------------------------------------------

def demo_fp_accumulation() -> None:
    print("=" * 60)
    print("Task 3 — Floating-point accumulation precision demo")
    print("=" * 60)
    N = 1_000
    delta = torch.tensor(0.01)

    # FP32
    acc_f32 = torch.tensor(0.0, dtype=torch.float32)
    for _ in range(N):
        acc_f32 = acc_f32 + delta.float()
    print(f"  FP32 sum of {N}×0.01 = {acc_f32.item():.6f}  (expected {N * 0.01:.6f})")

    # FP16 (implicit FP16 accumulation)
    acc_f16 = torch.tensor(0.0, dtype=torch.float16)
    for _ in range(N):
        acc_f16 = acc_f16 + delta.half()
    print(f"  FP16 sum of {N}×0.01 = {acc_f16.item():.6f}  (expected {N * 0.01:.6f})")

    # BF16
    acc_bf16 = torch.tensor(0.0, dtype=torch.bfloat16)
    for _ in range(N):
        acc_bf16 = acc_bf16 + delta.bfloat16()
    print(f"  BF16 sum of {N}×0.01 = {acc_bf16.item():.6f}  (expected {N * 0.01:.6f})")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Model benchmark")
    parser.add_argument("--mixed-precision", action="store_true",
                        help="Also benchmark with bfloat16 autocast (Task 3)")
    parser.add_argument("--no-warmup-demo", action="store_true",
                        help="Run the no-warmup experiment for each model")
    parser.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS.keys()),
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Which model sizes to benchmark")
    parser.add_argument("--fp-demo", action="store_true",
                        help="Run the floating-point accumulation precision demo")
    parser.add_argument("--model-dtype", choices=["fp32", "bf16"], default="fp32",
                        help="Parameter dtype for model weights (bf16 reduces memory for large models)")
    args = parser.parse_args()

    model_dtype = torch.float32 if args.model_dtype == "fp32" else torch.bfloat16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(device)}")
    print()

    if args.fp_demo:
        demo_fp_accumulation()

    print("=" * 60)
    print("Task 1 — Model benchmarking (forward + backward)")
    print(f"  batch={BATCH_SIZE}, context_length={CONTEXT_LENGTH}, "
            f"vocab={VOCAB_SIZE}, warmup={NUM_WARMUP}, iters={NUM_ITERS}, model_dtype={args.model_dtype}")
    print("=" * 60)

    for model_name in args.models:
        cfg = MODEL_CONFIGS[model_name]
        print(f"Building model '{model_name}' ...")
        model = None
        try:
            model = build_model(cfg, device, dtype=model_dtype)
            n_params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"  Parameters: {n_params:.1f}M")

            # Full-precision benchmark
            benchmark(model_name, model, device, use_autocast=False)

            # Mixed-precision benchmark (Task 3)
            if args.mixed_precision and device.type == "cuda":
                benchmark(model_name, model, device, use_autocast=True)

            # No-warmup demo
            if args.no_warmup_demo:
                benchmark_no_warmup(model_name, model, device)

        except torch.cuda.OutOfMemoryError:
            print(f"  !! OOM for model '{model_name}' — skipping\n")
        finally:
            if model is not None:
                del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print("Done.")


if __name__ == "__main__":
    main()
