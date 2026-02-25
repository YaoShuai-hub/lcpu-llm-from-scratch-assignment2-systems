from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from cs336_systems.ddp import DDPBucketed, DDPIndividualParameters
from cs336_systems.optimizer import ShardedOptimizer


def _set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _init_pg(rank: int, world_size: int, backend: str, master_port: int) -> torch.device:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return device


def _cleanup_pg() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


@dataclass
class CommRow:
    backend: str
    world_size: int
    size_mb: int
    avg_ms: float


def _worker_comm(rank: int, world_size: int, backend: str, sizes_mb: list[int], iters: int, warmup: int, master_port: int, out_file: str) -> None:
    device = _init_pg(rank, world_size, backend, master_port)
    rows: list[CommRow] = []
    for size_mb in sizes_mb:
        numel = size_mb * 1024 * 1024 // 4
        tensor = torch.ones(numel, device=device, dtype=torch.float32)

        for _ in range(warmup):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        dist.barrier()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        dist.barrier()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        avg_ms = (t1 - t0) * 1000.0 / iters
        if rank == 0:
            rows.append(CommRow(backend=backend, world_size=world_size, size_mb=size_mb, avg_ms=avg_ms))

    if rank == 0:
        with open(out_file, "w") as f:
            json.dump([asdict(r) for r in rows], f)
    _cleanup_pg()


def run_comm_single_node(gpu_world_sizes: list[int], cpu_world_sizes: list[int], sizes_mb: list[int], iters: int, warmup: int) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    port = 29501

    for ws in cpu_world_sizes:
        out_file = f"/tmp/comm_gloo_{ws}.json"
        mp.spawn(_worker_comm, args=(ws, "gloo", sizes_mb, iters, warmup, port, out_file), nprocs=ws, join=True)
        with open(out_file, "r") as f:
            all_rows.extend(json.load(f))
        port += 1

    for ws in gpu_world_sizes:
        out_file = f"/tmp/comm_nccl_{ws}.json"
        mp.spawn(_worker_comm, args=(ws, "nccl", sizes_mb, iters, warmup, port, out_file), nprocs=ws, join=True)
        with open(out_file, "r") as f:
            all_rows.extend(json.load(f))
        port += 1

    return all_rows


class TinyStack(nn.Module):
    def __init__(self, width: int = 2048, depth: int = 12):
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.append(nn.Linear(width, width, bias=False))
            layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(width, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.net(x))


def _allreduce_grads_naive(params: list[torch.nn.Parameter]) -> None:
    ws = dist.get_world_size()
    for p in params:
        if p.grad is None:
            continue
        p.grad.div_(ws)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)


def _allreduce_grads_flat(params: list[torch.nn.Parameter]) -> None:
    ws = dist.get_world_size()
    grads = [p.grad.view(-1) for p in params if p.grad is not None]
    flat = torch.cat(grads)
    flat.div_(ws)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    off = 0
    for p in params:
        if p.grad is None:
            continue
        n = p.grad.numel()
        p.grad.copy_(flat[off:off + n].view_as(p.grad))
        off += n


def _worker_naive_vs_flat(rank: int, world_size: int, steps: int, master_port: int, out_file: str) -> None:
    device = _init_pg(rank, world_size, "nccl", master_port)
    _set_seed(42)

    # Fixed total communication volume (~64 MB), but split into many tiny tensors
    # to highlight kernel launch / collective-call overhead in the naive strategy.
    n_tensors = 4096
    elems_per_tensor = 4096  # 4096 * 4096 * 4 bytes ~= 64 MB total
    grads = [torch.randn(elems_per_tensor, device=device, dtype=torch.float32) for _ in range(n_tensors)]
    flat_grad = torch.cat(grads)

    # Warmup
    for _ in range(3):
        for g in grads:
            dist.all_reduce(g, op=dist.ReduceOp.SUM)
        dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)
        dist.barrier()
        torch.cuda.synchronize(device)

    def run(mode: str) -> float:
        t0 = time.perf_counter()
        for _ in range(steps):
            if mode == "naive":
                for g in grads:
                    dist.all_reduce(g, op=dist.ReduceOp.SUM)
            else:
                dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)
            dist.barrier()
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0 / steps

    naive_ms = run("naive")
    flat_ms = run("flat")

    if rank == 0:
        with open(out_file, "w") as f:
            json.dump({"world_size": world_size, "naive_ms": naive_ms, "flat_ms": flat_ms}, f)
    _cleanup_pg()


def run_naive_vs_flat(world_size: int, steps: int) -> dict[str, Any]:
    out_file = "/tmp/naive_vs_flat.json"
    mp.spawn(_worker_naive_vs_flat, args=(world_size, steps, 29551, out_file), nprocs=world_size, join=True)
    with open(out_file, "r") as f:
        return json.load(f)


def _worker_overlap(rank: int, world_size: int, steps: int, master_port: int, out_file: str) -> None:
    device = _init_pg(rank, world_size, "nccl", master_port)
    _set_seed(123)

    base_model = TinyStack(width=1024, depth=20).to(device)
    ref_model = TinyStack(width=1024, depth=20).to(device)
    ref_model.load_state_dict(base_model.state_dict())

    ddp_overlap = DDPIndividualParameters(base_model)
    params_ref = [p for p in ref_model.parameters() if p.requires_grad]

    x = torch.randn(16, 1024, device=device)
    y = torch.randn(16, 1024, device=device)

    # warmup
    for _ in range(2):
        for p in params_ref:
            p.grad = None
        loss = ((ref_model(x) - y) ** 2).mean()
        loss.backward()
        _allreduce_grads_naive(params_ref)

        ref_model.zero_grad(set_to_none=True)
        loss2 = ((ddp_overlap(x) - y) ** 2).mean()
        loss2.backward()
        ddp_overlap.finish_gradient_synchronization()

    def run_serial() -> float:
        t0 = time.perf_counter()
        for _ in range(steps):
            for p in params_ref:
                p.grad = None
            loss = ((ref_model(x) - y) ** 2).mean()
            loss.backward()
            _allreduce_grads_naive(params_ref)
            dist.barrier()
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0 / steps

    def run_overlap() -> float:
        t0 = time.perf_counter()
        for _ in range(steps):
            ddp_overlap.zero_grad(set_to_none=True)
            loss = ((ddp_overlap(x) - y) ** 2).mean()
            loss.backward()
            ddp_overlap.finish_gradient_synchronization()
            dist.barrier()
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0 / steps

    serial_ms = run_serial()
    overlap_ms = run_overlap()

    if rank == 0:
        with open(out_file, "w") as f:
            json.dump({"world_size": world_size, "serial_ms": serial_ms, "overlap_ms": overlap_ms}, f)
    _cleanup_pg()


def run_overlap_benchmark(world_size: int, steps: int) -> dict[str, Any]:
    out_file = "/tmp/overlap.json"
    mp.spawn(_worker_overlap, args=(world_size, steps, 29561, out_file), nprocs=world_size, join=True)
    with open(out_file, "r") as f:
        return json.load(f)


def _worker_bucket(rank: int, world_size: int, bucket_sizes_mb: list[float], steps: int, master_port: int, out_file: str) -> None:
    device = _init_pg(rank, world_size, "nccl", master_port)
    _set_seed(456)

    results = []
    x = torch.randn(16, 1024, device=device)
    y = torch.randn(16, 1024, device=device)

    for bsz in bucket_sizes_mb:
        model = TinyStack(width=1024, depth=20).to(device)
        ddp = DDPBucketed(model, bucket_size_mb=bsz)

        for _ in range(2):
            ddp.reset_grad_counts()
            ddp.zero_grad(set_to_none=True)
            loss = ((ddp(x) - y) ** 2).mean()
            loss.backward()
            ddp.finish_gradient_synchronization()

        t0 = time.perf_counter()
        for _ in range(steps):
            ddp.reset_grad_counts()
            ddp.zero_grad(set_to_none=True)
            loss = ((ddp(x) - y) ** 2).mean()
            loss.backward()
            ddp.finish_gradient_synchronization()
            dist.barrier()
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        avg_ms = (t1 - t0) * 1000.0 / steps
        if rank == 0:
            results.append({"bucket_size_mb": bsz, "avg_ms": avg_ms})

    if rank == 0:
        with open(out_file, "w") as f:
            json.dump(results, f)
    _cleanup_pg()


def run_bucket_benchmark(world_size: int, bucket_sizes_mb: list[float], steps: int) -> list[dict[str, Any]]:
    out_file = "/tmp/bucket.json"
    mp.spawn(_worker_bucket, args=(world_size, bucket_sizes_mb, steps, 29571, out_file), nprocs=world_size, join=True)
    with open(out_file, "r") as f:
        return json.load(f)


def _optimizer_state_bytes_adamw(opt: torch.optim.Optimizer) -> int:
    total = 0
    for state in opt.state.values():
        for v in state.values():
            if torch.is_tensor(v):
                total += v.numel() * v.element_size()
    return total


def _sharded_optimizer_state_bytes(opt: ShardedOptimizer) -> int:
    if getattr(opt, "_local_optimizer", None) is None:
        return 0
    return _optimizer_state_bytes_adamw(opt._local_optimizer)


def _worker_sharding(rank: int, world_size: int, steps: int, master_port: int, out_file: str) -> None:
    device = _init_pg(rank, world_size, "nccl", master_port)
    _set_seed(789)

    model_a = TinyStack(width=1536, depth=16).to(device)
    model_b = TinyStack(width=1536, depth=16).to(device)
    model_b.load_state_dict(model_a.state_dict())

    opt_full = torch.optim.AdamW(model_a.parameters(), lr=1e-3)
    opt_shard = ShardedOptimizer(model_b.parameters(), torch.optim.AdamW, lr=1e-3)

    x = torch.randn(8, 1536, device=device)
    y = torch.randn(8, 1536, device=device)

    # Warmup
    for _ in range(2):
        opt_full.zero_grad(set_to_none=True)
        loss = ((model_a(x) - y) ** 2).mean()
        loss.backward()
        opt_full.step()

        opt_shard.zero_grad(set_to_none=True)
        loss2 = ((model_b(x) - y) ** 2).mean()
        loss2.backward()
        opt_shard.step()

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        opt_full.zero_grad(set_to_none=True)
        loss = ((model_a(x) - y) ** 2).mean()
        loss.backward()
        opt_full.step()
        dist.barrier()
        torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    full_ms = (t1 - t0) * 1000.0 / steps
    full_peak = torch.cuda.max_memory_allocated(device)
    full_state = _optimizer_state_bytes_adamw(opt_full)

    torch.cuda.reset_peak_memory_stats(device)
    t2 = time.perf_counter()
    for _ in range(steps):
        opt_shard.zero_grad(set_to_none=True)
        loss = ((model_b(x) - y) ** 2).mean()
        loss.backward()
        opt_shard.step()
        dist.barrier()
        torch.cuda.synchronize(device)
    t3 = time.perf_counter()
    shard_ms = (t3 - t2) * 1000.0 / steps
    shard_peak = torch.cuda.max_memory_allocated(device)
    shard_state_local = _sharded_optimizer_state_bytes(opt_shard)

    stats = torch.tensor([
        float(full_ms),
        float(shard_ms),
        float(full_peak),
        float(shard_peak),
        float(full_state),
        float(shard_state_local),
    ], device=device)
    gathered = [torch.zeros_like(stats) for _ in range(world_size)]
    dist.all_gather(gathered, stats)

    if rank == 0:
        rows = [
            {
                "rank": i,
                "full_step_ms": float(t[0].item()),
                "shard_step_ms": float(t[1].item()),
                "full_peak_gb": float(t[2].item() / 1024**3),
                "shard_peak_gb": float(t[3].item() / 1024**3),
                "full_optimizer_state_gb": float(t[4].item() / 1024**3),
                "shard_optimizer_state_local_gb": float(t[5].item() / 1024**3),
            }
            for i, t in enumerate(gathered)
        ]
        with open(out_file, "w") as f:
            json.dump(rows, f)

    _cleanup_pg()


def run_sharding_benchmark(world_size: int, steps: int) -> list[dict[str, Any]]:
    out_file = "/tmp/sharding.json"
    mp.spawn(_worker_sharding, args=(world_size, steps, 29581, out_file), nprocs=world_size, join=True)
    with open(out_file, "r") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run missing DDP/sharding benchmarks for report")
    parser.add_argument("--output", default="benchmark_results_missing.json")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    n_gpu = torch.cuda.device_count()
    if n_gpu < 2:
        raise RuntimeError("Need at least 2 GPUs for NCCL benchmarks")

    gpu_world_sizes = [2, 4] if n_gpu >= 4 else [2]
    cpu_world_sizes = [2, 4]

    results: dict[str, Any] = {}

    results["distributed_communication_single_node"] = run_comm_single_node(
        gpu_world_sizes=gpu_world_sizes,
        cpu_world_sizes=cpu_world_sizes,
        sizes_mb=[1, 100, 1024],
        iters=args.iters,
        warmup=args.warmup,
    )

    ws_for_training = 4 if n_gpu >= 4 else 2
    results["naive_ddp_benchmarking_vs_flat"] = run_naive_vs_flat(world_size=ws_for_training, steps=args.steps)
    results["ddp_overlap_individual_parameters_benchmarking"] = run_overlap_benchmark(world_size=ws_for_training, steps=args.steps)
    results["ddp_bucketed_benchmarking"] = run_bucket_benchmark(
        world_size=ws_for_training,
        bucket_sizes_mb=[1, 5, 10, 25, 50, 100, 500],
        steps=args.steps,
    )
    results["optimizer_state_sharding_accounting"] = run_sharding_benchmark(world_size=ws_for_training, steps=args.steps)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
