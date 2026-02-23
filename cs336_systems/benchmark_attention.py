"""
benchmark_attention.py
======================
Task 5: Benchmark the CausalMultiHeadSelfAttention module (MHA) in isolation.
  - Sweep over sequence lengths to measure forward/backward time and memory.
  - Find the OOM threshold.
  - Compare native vs torch.compile.

Run:
    uv run python -m cs336_systems.benchmark_attention
    uv run python -m cs336_systems.benchmark_attention --compile
    uv run python -m cs336_systems.benchmark_attention --oom-search
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time

import torch
import torch.nn as nn

from cs336_basics.model import CausalMultiHeadSelfAttention, RotaryEmbedding

# ---------------------------------------------------------------------------
# Fixed hyper-parameters (matching a representative head of the 2.7B model)
# ---------------------------------------------------------------------------
D_MODEL = 1024      # model dimension
NUM_HEADS = 16      # number of heads  → d_head = 64
BATCH_SIZE = 4
ROPE_THETA = 10_000.0

SEQ_LENGTHS = [128, 256, 512, 1024, 2048, 4096]

NUM_WARMUP = 3
NUM_ITERS = 10


# ---------------------------------------------------------------------------
# Build MHA module
# ---------------------------------------------------------------------------

def build_mha(max_seq: int, device: torch.device) -> CausalMultiHeadSelfAttention:
    rope = RotaryEmbedding(
        context_length=max_seq,
        dim=D_MODEL // NUM_HEADS,
        theta=ROPE_THETA,
    )
    mha = CausalMultiHeadSelfAttention(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        positional_encoder=rope,
    )
    return mha.to(device)


def make_input(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.randn(BATCH_SIZE, seq_len, D_MODEL, device=device, requires_grad=True)


# ---------------------------------------------------------------------------
# Core benchmark loop
# ---------------------------------------------------------------------------

def benchmark_mha(
    mha: nn.Module,
    seq_len: int,
    device: torch.device,
    label: str = "",
) -> dict:
    """Run forward + backward timing and measure peak memory."""
    x = make_input(seq_len, device)

    # ── Warm-up ────────────────────────────────────────────────────────────
    for _ in range(NUM_WARMUP):
        out = mha(x)
        out.sum().backward()
        x.grad = None
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    # ── Forward timing ─────────────────────────────────────────────────────
    fwd_times: list[float] = []
    for _ in range(NUM_ITERS):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = mha(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        fwd_times.append((time.perf_counter() - t0) * 1e3)

    # ── Backward timing ────────────────────────────────────────────────────
    bwd_times: list[float] = []
    for _ in range(NUM_ITERS):
        x_iter = make_input(seq_len, device)
        out_iter = mha(x_iter)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out_iter.sum().backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        bwd_times.append((time.perf_counter() - t0) * 1e3)

    peak_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0

    return {
        "seq_len": seq_len,
        "fwd_mean": statistics.mean(fwd_times),
        "fwd_std": statistics.stdev(fwd_times) if len(fwd_times) > 1 else 0.0,
        "bwd_mean": statistics.mean(bwd_times),
        "bwd_std": statistics.stdev(bwd_times) if len(bwd_times) > 1 else 0.0,
        "peak_memory_gb": peak_bytes / 1024**3,
        "label": label,
    }


def print_results_table(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    print(f"  {'seq_len':>8} | {'fwd (ms)':>16} | {'bwd (ms)':>16} | {'peak mem (GB)':>14}")
    print("  " + "-" * 62)
    for r in rows:
        fwd = f"{r['fwd_mean']:7.2f} ± {r['fwd_std']:5.2f}"
        bwd = f"{r['bwd_mean']:7.2f} ± {r['bwd_std']:5.2f}"
        mem = f"{r['peak_memory_gb']:14.3f}"
        print(f"  {r['seq_len']:>8} | {fwd:>16} | {bwd:>16} | {mem}")


# ---------------------------------------------------------------------------
# OOM search
# ---------------------------------------------------------------------------

def find_oom_threshold(device: torch.device) -> None:
    """Double the sequence length until OOM, report the threshold."""
    print("\nOOM threshold search (doubling seq_len each step)...")
    seq_len = 1024
    last_ok = None
    while seq_len <= 1_000_000:
        try:
            torch.cuda.empty_cache()
            gc.collect()
            mha = build_mha(max_seq=seq_len, device=device)
            x = make_input(seq_len, device)
            out = mha(x)
            out.sum().backward()
            torch.cuda.synchronize()
            last_ok = seq_len
            print(f"  seq_len={seq_len:>8}  → OK   "
                  f"| peak mem = {torch.cuda.max_memory_allocated(device)/1024**3:.3f} GB")
            del mha, x, out
            seq_len *= 2
        except torch.cuda.OutOfMemoryError:
            print(f"  seq_len={seq_len:>8}  → OOM  ← threshold")
            torch.cuda.empty_cache()
            break

    if last_ok is not None:
        print(f"\nLargest successful seq_len: {last_ok}")
    else:
        print("OOM at the very first seq_len!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="MHA benchmark (Task 5)")
    parser.add_argument("--compile", action="store_true",
                        help="Benchmark with torch.compile")
    parser.add_argument("--oom-search", action="store_true",
                        help="Run the OOM threshold search")
    parser.add_argument("--seq-lengths", nargs="+", type=int,
                        default=SEQ_LENGTHS,
                        help="Sequence lengths to benchmark")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(device)}")
    print(f"Config : d_model={D_MODEL}, num_heads={NUM_HEADS}, batch={BATCH_SIZE}\n")

    # ── Standard benchmark ─────────────────────────────────────────────────
    native_rows: list[dict] = []
    max_seq = max(args.seq_lengths)
    mha_base = build_mha(max_seq=max_seq, device=device)

    for seq_len in args.seq_lengths:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        try:
            row = benchmark_mha(mha_base, seq_len, device, label="native")
            native_rows.append(row)
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at seq_len={seq_len} — stopping native sweep")
            torch.cuda.empty_cache()
            break

    print_results_table(native_rows, "Native PyTorch MHA")

    # ── torch.compile benchmark ────────────────────────────────────────────
    if args.compile:
        if device.type != "cuda":
            print("\ntorch.compile requires CUDA. Skipping.")
        else:
            print("\nCompiling MHA with torch.compile (first call triggers JIT)...")
            mha_compiled = torch.compile(build_mha(max_seq=max_seq, device=device))
            # Trigger compilation (not timed)
            _x = make_input(args.seq_lengths[0], device)
            _out = mha_compiled(_x)
            _out.sum().backward()
            torch.cuda.synchronize()
            del _x, _out
            print("Compilation done.\n")

            compiled_rows: list[dict] = []
            for seq_len in args.seq_lengths:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                try:
                    row = benchmark_mha(mha_compiled, seq_len, device, label="compiled")
                    compiled_rows.append(row)
                except torch.cuda.OutOfMemoryError:
                    print(f"  OOM at seq_len={seq_len} — stopping compiled sweep")
                    torch.cuda.empty_cache()
                    break

            print_results_table(compiled_rows, "torch.compile MHA")

            # Speed-up summary
            if native_rows and compiled_rows:
                print("\nSpeed-up (native fwd / compiled fwd):")
                n_map = {r["seq_len"]: r for r in native_rows}
                for r in compiled_rows:
                    n = n_map.get(r["seq_len"])
                    if n:
                        fwd_speedup = n["fwd_mean"] / r["fwd_mean"] if r["fwd_mean"] > 0 else float("nan")
                        bwd_speedup = n["bwd_mean"] / r["bwd_mean"] if r["bwd_mean"] > 0 else float("nan")
                        print(f"  seq_len={r['seq_len']:>6} | fwd ×{fwd_speedup:.2f} | bwd ×{bwd_speedup:.2f}")

    # ── OOM search ─────────────────────────────────────────────────────────
    if args.oom_search:
        if device.type != "cuda":
            print("\nOOM search requires CUDA. Skipping.")
        else:
            find_oom_threshold(device)

    print("\nDone.")


if __name__ == "__main__":
    main()
