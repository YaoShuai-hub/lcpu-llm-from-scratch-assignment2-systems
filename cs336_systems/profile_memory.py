"""
profile_memory.py
=================
Tasks 2 & 4: Memory snapshot profiling + nsys-friendly clean run.

Usage
-----
# Task 4 — Generate memory snapshot (visualize at pytorch.org/memory_viz):
    uv run python -m cs336_systems.profile_memory --mode memory

# Task 2 — Single clean forward/backward pass for nsys profiling:
    nsys profile -w true -t cuda,nvtx,osrt -s cpu -o profile_2_7b \\
        uv run python -m cs336_systems.profile_memory --mode nsys

# Task 4 table — Sweep context lengths and print peak memory:
    uv run python -m cs336_systems.profile_memory --mode sweep
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.cuda.nvtx as nvtx

from cs336_basics.model import BasicsTransformerLM

# ---------------------------------------------------------------------------
# 2.7B model config (primary target for nsys + memory profiling)
# ---------------------------------------------------------------------------
MODEL_2_7B = dict(
    vocab_size=10_000,
    context_length=128,
    d_model=2560,
    num_layers=32,
    num_heads=32,
    d_ff=10240,
    rope_theta=10_000.0,
)

BATCH_SIZE = 4
ROPE_THETA = 10_000.0
VOCAB_SIZE = 10_000
CONTEXT_LENGTH = 128


def build_model_2_7b(device: torch.device) -> BasicsTransformerLM:
    model = BasicsTransformerLM(**MODEL_2_7B)
    return model.to(device)


# ---------------------------------------------------------------------------
# Task 2: nsys-friendly script (one clean step with NVTX annotations)
# ---------------------------------------------------------------------------

def run_nsys(device: torch.device, include_backward: bool = True) -> None:
    """
    Runs a single forward (and optionally backward + optimizer) step
    of the 2.7B model with NVTX range annotations for nsys.
    """
    print("Building 2.7B model...")
    model = build_model_2_7b(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LENGTH), device=device)

    # Warm-up (not annotated, not counted)
    print("Warming up...")
    for _ in range(2):
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # ── Profiled step ──────────────────────────────────────────────────────
    print("Running profiled step (nsys will capture this)...")
    optimizer.zero_grad(set_to_none=True)

    with nvtx.range("forward_pass"):
        logits = model(x)
    torch.cuda.synchronize()

    if include_backward:
        loss = logits.sum()
        with nvtx.range("backward_pass"):
            loss.backward()
        torch.cuda.synchronize()

        with nvtx.range("optimizer_step"):
            optimizer.step()
        torch.cuda.synchronize()

    print("Done — nsys profile captured.")


# ---------------------------------------------------------------------------
# Task 4: Memory snapshot (forward-only & full training step)
# ---------------------------------------------------------------------------

def run_memory_snapshot(
    device: torch.device,
    include_backward: bool,
    context_length: int = CONTEXT_LENGTH,
    output_file: str | None = None,
) -> None:
    """
    Records a PyTorch memory snapshot during a single forward (or forward+backward)
    pass of the 2.7B model.

    Visualize the resulting .pickle file at: https://pytorch.org/memory_viz
    """
    cfg = {**MODEL_2_7B, "context_length": context_length}
    print(f"Building 2.7B model (context_length={context_length})...")
    model = BasicsTransformerLM(**cfg).to(device)

    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, context_length), device=device)

    # Warm-up (outside snapshot)
    for _ in range(2):
        logits = model(x)
        if include_backward:
            logits.sum().backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Start recording
    print("Starting memory recording...")
    torch.cuda.memory._record_memory_history(max_entries=100_000)

    # ── Profiled step ──────────────────────────────────────────────────────
    model.zero_grad(set_to_none=True)
    logits = model(x)
    if include_backward:
        loss = logits.sum()
        loss.backward()
    torch.cuda.synchronize()

    # Dump snapshot
    if output_file is None:
        suffix = "fwd_bwd" if include_backward else "fwd_only"
        output_file = f"memory_snapshot_{suffix}_ctx{context_length}.pickle"

    torch.cuda.memory._dump_snapshot(output_file)
    torch.cuda.memory._record_memory_history(enabled=None)

    peak_bytes = torch.cuda.max_memory_allocated(device)
    print(f"Memory snapshot saved → {output_file}")
    print(f"Peak memory allocated: {peak_bytes / 1024**3:.3f} GB")


# ---------------------------------------------------------------------------
# Task 4 table: Sweep over context lengths
# ---------------------------------------------------------------------------

def sweep_context_lengths(device: torch.device) -> None:
    """Print peak memory for different context lengths (Table in Task 4)."""
    context_lengths = [128, 256, 512]
    print(f"{'context_length':>16} | {'peak_memory (GB)':>18}")
    print("-" * 40)
    for ctx in context_lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            cfg = {**MODEL_2_7B, "context_length": ctx}
            model = BasicsTransformerLM(**cfg).to(device)
            x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, ctx), device=device)

            logits = model(x)
            loss = logits.sum()
            loss.backward()
            torch.cuda.synchronize()

            peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
            print(f"{ctx:>16} | {peak_gb:>18.3f}")
            del model, x, logits, loss
        except torch.cuda.OutOfMemoryError:
            print(f"{ctx:>16} | {'OOM':>18}")
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Memory / nsys profiling for 2.7B model")
    parser.add_argument(
        "--mode",
        choices=["nsys", "memory", "sweep"],
        default="memory",
        help=(
            "nsys   – run a single annotated step (call via nsys profile ...)\n"
            "memory – dump PyTorch memory snapshots (fwd-only and fwd+bwd)\n"
            "sweep  – print peak memory for different context lengths (Task 4 table)"
        ),
    )
    parser.add_argument("--no-backward", action="store_true",
                        help="Only run forward pass (relevant for 'memory' mode)")
    parser.add_argument("--context-length", type=int, default=CONTEXT_LENGTH,
                        help="Context length to use (memory mode)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available — this script is designed for GPU profiling.")
        return

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(device)}\n")

    if args.mode == "nsys":
        run_nsys(device, include_backward=not args.no_backward)

    elif args.mode == "memory":
        # Task 4: generate both forward-only and full-step snapshots
        for include_bwd in ([False, True] if not args.no_backward else [False]):
            torch.cuda.reset_peak_memory_stats(device)
            run_memory_snapshot(
                device,
                include_backward=include_bwd,
                context_length=args.context_length,
            )
            torch.cuda.empty_cache()
            print()

    elif args.mode == "sweep":
        sweep_context_lengths(device)

    print("Done.")


if __name__ == "__main__":
    main()
