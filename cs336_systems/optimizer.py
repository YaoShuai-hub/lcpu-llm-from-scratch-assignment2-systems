"""
Sharded Optimizer: Each rank maintains optimizer state for only a subset of
the model's parameters, reducing optimizer-state memory by a factor of world_size.

Algorithm
---------
1. All model parameters are partitioned round-robin across ranks.
2. Each rank creates a **local** optimizer instance that only covers its shard.
3. During ``step()``, each rank performs a local optimizer update on its shard,
   then the updated parameter tensors are broadcast to all other ranks so
   every rank ends up with the same (fully updated) model weights.
4. ``zero_grad()`` zeros gradients for *all* parameters, regardless of ownership.
"""

from __future__ import annotations

from typing import Type, Iterable

import torch
import torch.distributed as dist
import torch.optim as optim


class ShardedOptimizer:
    """
    A drop-in wrapper around a standard ``torch.optim.Optimizer`` that shards
    optimizer state across distributed ranks.

    Args:
        params: Iterable of ``torch.Tensor`` parameters or param groups (dicts).
        optimizer_cls: The optimizer class to instantiate locally.
        **kwargs: Keyword arguments forwarded to the optimizer constructor.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        optimizer_cls: Type[optim.Optimizer],
        **kwargs,
    ):
        # Flatten to a list of tensors (handle both tensors and param-group dicts).
        raw = list(params)
        all_params: list[torch.Tensor] = []
        for item in raw:
            if isinstance(item, dict):
                all_params.extend(item["params"])
            else:
                all_params.append(item)

        self._all_params: list[torch.Tensor] = all_params

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Round-robin partition: rank r owns parameters at indices r, r+world_size, …
        self._my_params: list[torch.Tensor] = [
            p for i, p in enumerate(all_params) if i % world_size == rank
        ]

        # Build the local optimizer.
        if self._my_params:
            self._local_optimizer: optim.Optimizer = optimizer_cls(self._my_params, **kwargs)
        else:
            self._local_optimizer = None  # type: ignore[assignment]

        # Keep track of the rank-to-param assignment for broadcasting.
        self._param_owner: list[int] = [i % world_size for i in range(len(all_params))]

    # ------------------------------------------------------------------
    # Public interface (mirrors torch.optim.Optimizer)
    # ------------------------------------------------------------------

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero (or None-out) gradients for all parameters across all ranks."""
        for param in self._all_params:
            if param.grad is not None:
                if set_to_none:
                    param.grad = None
                else:
                    param.grad.zero_()

    def step(self, closure=None) -> None:
        """
        1. Each rank updates its local parameter shard via the local optimizer.
        2. Every updated parameter tensor is then broadcast from its owner rank
           to all other ranks, ensuring a consistent model state.
        """
        loss = None
        if self._local_optimizer is not None:
            loss = self._local_optimizer.step(closure)

        world_size = dist.get_world_size()

        # Broadcast each parameter from its designated owner.
        for i, param in enumerate(self._all_params):
            src_rank = self._param_owner[i]
            dist.broadcast(param.data, src=src_rank)

        return loss

    # ------------------------------------------------------------------
    # Passthrough helpers (useful for logging / checkpointing)
    # ------------------------------------------------------------------

    @property
    def param_groups(self):
        if self._local_optimizer is not None:
            return self._local_optimizer.param_groups
        return []

    def state_dict(self):
        if self._local_optimizer is not None:
            return self._local_optimizer.state_dict()
        return {}

    def load_state_dict(self, state_dict):
        if self._local_optimizer is not None:
            self._local_optimizer.load_state_dict(state_dict)
