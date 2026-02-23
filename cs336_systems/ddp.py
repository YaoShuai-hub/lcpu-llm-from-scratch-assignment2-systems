"""
Distributed Data Parallel (DDP) implementations.

Two variants:
  1. DDPIndividualParameters  - all-reduce each gradient tensor separately
                                 (communication overlaps with backprop via hooks)
  2. DDPBucketed               - accumulate gradients into buckets, then all-reduce
                                 per bucket (overlaps with backprop via hooks)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn


# ───────────────────────────────────────────────────────────────────────────────
# DDP: Individual Parameters
# ───────────────────────────────────────────────────────────────────────────────

class DDPIndividualParameters(nn.Module):
    """
    DDP wrapper that all-reduces each parameter's gradient individually,
    overlapping communication with backward computation via autograd hooks.

    On creation, parameters are broadcast from rank 0 to all other ranks.
    During backward, a non-blocking all_reduce is launched for each parameter
    as soon as its gradient becomes available.  After backward, the caller
    should invoke ``finish_gradient_synchronization()`` (or the
    ``ddp_individual_parameters_on_after_backward`` adapter) to wait for all
    pending all-reduces to complete before the optimizer step.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

        # Broadcast parameters from rank-0 so all ranks start identically.
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # Storage for pending async work handles, one per parameter.
        self._pending_works: list[dist.Work] = []

        # Register a backward hook on every parameter that has requires_grad.
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        for param in self.module.parameters():
            if param.requires_grad:
                hook = param.register_post_accumulate_grad_hook(
                    self._make_hook()
                )
                self._hooks.append(hook)

    def _make_hook(self):
        """Return a hook that launches an all-reduce for the parameter's grad."""
        def hook(param):
            if param.grad is None:
                return
            # Divide by world size so that the average gradient is used
            # (equivalent to computing the sum over all ranks and dividing).
            param.grad.div_(dist.get_world_size())
            work = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self._pending_works.append(work)
        return hook

    def finish_gradient_synchronization(self):
        """Wait for all pending gradient all-reduces to finish."""
        for work in self._pending_works:
            work.wait()
        self._pending_works.clear()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


# ───────────────────────────────────────────────────────────────────────────────
# DDP: Bucketed
# ───────────────────────────────────────────────────────────────────────────────

class DDPBucketed(nn.Module):
    """
    DDP wrapper that accumulates gradients into fixed-size buckets and
    launches an all-reduce per bucket, overlapping communication with backward.

    Parameters (in reverse registration order, which mirrors typical backward
    order) are assigned to buckets of ``bucket_size_mb`` megabytes.  When a
    bucket's last expected gradient arrives, an async all-reduce is launched for
    the entire bucket.

    After backward the caller must invoke ``finish_gradient_synchronization()``
    to wait for all buckets to be reduced.
    """

    def __init__(self, module: nn.Module, bucket_size_mb: Optional[float] = None):
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb

        # Broadcast parameters from rank-0.
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # Build bucket structure.
        # We process parameters in *reverse* order of registration (which is
        # typically reverse topological / backward order).
        trainable_params = [p for p in self.module.parameters() if p.requires_grad]
        # Reverse so that the first bucket to fill corresponds to later layers
        # (which have their gradients ready first during backward).
        reversed_params = list(reversed(trainable_params))

        self._buckets: list[list[nn.Parameter]] = []  # list of param groups
        self._param_to_bucket: dict[int, int] = {}    # param id → bucket index

        if bucket_size_mb is None:
            # Single bucket containing all parameters.
            self._buckets = [reversed_params]
        else:
            bucket_size_bytes = bucket_size_mb * 1024 * 1024
            current_bucket: list[nn.Parameter] = []
            current_bytes = 0.0
            for param in reversed_params:
                param_bytes = param.numel() * param.element_size()
                if current_bucket and current_bytes + param_bytes > bucket_size_bytes:
                    self._buckets.append(current_bucket)
                    current_bucket = []
                    current_bytes = 0.0
                current_bucket.append(param)
                current_bytes += param_bytes
            if current_bucket:
                self._buckets.append(current_bucket)

        for bucket_idx, bucket in enumerate(self._buckets):
            for param in bucket:
                self._param_to_bucket[id(param)] = bucket_idx

        # Per-bucket state tracking.
        n_buckets = len(self._buckets)
        self._bucket_grad_count = [0] * n_buckets     # how many grads arrived
        self._bucket_pending_works: list[Optional[dist.Work]] = [None] * n_buckets

        # Register backward hooks.
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        for param in trainable_params:
            hook = param.register_post_accumulate_grad_hook(self._make_hook(param))
            self._hooks.append(hook)

    def _make_hook(self, param: nn.Parameter):
        """Return a hook that accumulates grad counts and fires all-reduce when a bucket is full."""
        def hook(p):
            if p.grad is None:
                return
            bucket_idx = self._param_to_bucket[id(p)]
            self._bucket_grad_count[bucket_idx] += 1
            # When all grads in the bucket have arrived, launch all-reduce.
            if self._bucket_grad_count[bucket_idx] == len(self._buckets[bucket_idx]):
                self._launch_bucket_allreduce(bucket_idx)
        return hook

    def _launch_bucket_allreduce(self, bucket_idx: int):
        """Flatten, all-reduce, then scatter back the params in this bucket."""
        world_size = dist.get_world_size()
        bucket_params = self._buckets[bucket_idx]

        # Flatten all grads in this bucket into a single contiguous tensor.
        grads = [p.grad.view(-1) for p in bucket_params]
        flat = torch.cat(grads)
        flat.div_(world_size)
        work = dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=True)

        # We store the work handle together with enough info to scatter back.
        # We'll do the scatter-back in finish_gradient_synchronization().
        self._bucket_pending_works[bucket_idx] = (work, bucket_params, flat)

    def finish_gradient_synchronization(self):
        """
        Wait for all pending bucket all-reduces and write the reduced gradients
        back into the individual parameter .grad tensors.
        """
        for bucket_idx, work_info in enumerate(self._bucket_pending_works):
            if work_info is None:
                continue
            work, bucket_params, flat = work_info
            work.wait()
            # Scatter reduced values back into individual .grad tensors.
            offset = 0
            for param in bucket_params:
                numel = param.grad.numel()
                param.grad.copy_(flat[offset: offset + numel].view_as(param.grad))
                offset += numel
            self._bucket_pending_works[bucket_idx] = None

    def reset_grad_counts(self):
        """Reset per-bucket gradient counters at the start of each training step."""
        self._bucket_grad_count = [0] * len(self._buckets)
        self._bucket_pending_works = [None] * len(self._buckets)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
