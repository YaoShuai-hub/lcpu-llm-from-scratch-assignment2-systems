"""
FlashAttention2 implementations using PyTorch and Triton.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class FlashAttentionPyTorch(torch.autograd.Function):
    """
    FlashAttention2 implemented using standard PyTorch operations (no Triton).

    Forward pass uses tiling to avoid materializing the full (n_queries x n_keys)
    attention matrix, saving only O (output) and L (log-sum-exp of attention scores).

    The interface is:
        output = FlashAttentionPyTorch.apply(q, k, v, is_causal)
    where q, k, v have shape (batch, seq, d) and is_causal is a bool.
    """

    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        """
        FlashAttention2 tiled forward pass.

        Args:
            q: (batch, n_q, d)
            k: (batch, n_k, d)
            v: (batch, n_k, d)
            is_causal: bool

        Returns:
            O: (batch, n_q, d) - attention output
        """
        batch, n_q, d = q.shape
        n_k = k.shape[1]
        scale = d ** -0.5

        # Block sizes for tiling
        Bc = min(64, n_k)
        Br = min(64, n_q)

        # Initialize outputs
        O = torch.zeros_like(q)
        # L = m + log(l) — log-sum-exp of attention scores per query
        L = torch.full((batch, n_q), float("-inf"), device=q.device, dtype=torch.float32)

        # Upcast to float32 for numerical stability
        q_f = q.float()
        k_f = k.float()
        v_f = v.float()

        # Tiled computation over query blocks
        for i in range(0, n_q, Br):
            qi = q_f[:, i : i + Br, :]  # (batch, br, d)
            br_i = qi.shape[1]

            oi = torch.zeros(batch, br_i, d, device=q.device, dtype=torch.float32)
            mi = torch.full((batch, br_i), float("-inf"), device=q.device, dtype=torch.float32)
            li = torch.zeros(batch, br_i, device=q.device, dtype=torch.float32)

            # Inner loop over key blocks
            for j in range(0, n_k, Bc):
                kj = k_f[:, j : j + Bc, :]  # (batch, bc, d)
                vj = v_f[:, j : j + Bc, :]  # (batch, bc, d)

                # Compute attention scores: S_ij = q_i @ k_j.T / sqrt(d)
                sij = torch.einsum("bid,bjd->bij", qi, kj) * scale  # (batch, br_i, bc)

                # Causal masking
                if is_causal:
                    qi_idx = torch.arange(i, i + br_i, device=q.device).unsqueeze(1)  # (br_i, 1)
                    kj_idx = torch.arange(j, j + kj.shape[1], device=q.device).unsqueeze(0)  # (1, bc)
                    sij = sij.masked_fill(qi_idx < kj_idx, float("-inf"))

                # Row max of current block
                mij = sij.amax(dim=-1)  # (batch, br_i)

                # Exponentiate after shifting by row max
                pij = torch.exp(sij - mij.unsqueeze(-1))  # (batch, br_i, bc)
                lij = pij.sum(dim=-1)  # (batch, br_i)

                # Update running stats using online softmax trick
                mi_new = torch.maximum(mi, mij)

                exp_mi_diff = torch.exp(mi - mi_new)  # correction factor for old running stats
                exp_mij_diff = torch.exp(mij - mi_new)  # correction factor for new block

                li_new = exp_mi_diff * li + exp_mij_diff * lij
                oi_new = (
                    exp_mi_diff.unsqueeze(-1) * oi
                    + exp_mij_diff.unsqueeze(-1) * torch.einsum("bij,bjd->bid", pij, vj)
                )

                mi = mi_new
                li = li_new
                oi = oi_new

            # Normalize by the row-wise sum
            oi = oi / li.unsqueeze(-1)

            # Write output and log-sum-exp
            O[:, i : i + br_i, :] = oi.to(q.dtype)
            L[:, i : i + br_i] = mi + torch.log(li)

        ctx.save_for_backward(q, k, v, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        """
        FlashAttention2 backward pass.

        Recomputes the attention probabilities P from saved Q, K, L
        to avoid storing the full attention matrix.
        """
        q, k, v, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        batch, n_q, d = q.shape
        n_k = k.shape[1]
        scale = d ** -0.5

        dO_f = dO.float()
        q_f = q.float()
        k_f = k.float()
        v_f = v.float()
        O_f = O.float()
        L_f = L.float()  # (batch, n_q)

        # Recompute full S from Q, K
        S = torch.einsum("bid,bjd->bij", q_f, k_f) * scale  # (batch, n_q, n_k)

        if is_causal:
            q_idx = torch.arange(n_q, device=q.device).unsqueeze(1)  # (n_q, 1)
            k_idx = torch.arange(n_k, device=q.device).unsqueeze(0)  # (1, n_k)
            S = S.masked_fill(q_idx < k_idx, float("-inf"))

        # P = softmax(S) = exp(S - L)
        P = torch.exp(S - L_f.unsqueeze(-1))  # (batch, n_q, n_k)

        # dV = P^T @ dO
        dV = torch.einsum("bij,bid->bjd", P, dO_f)  # (batch, n_k, d)

        # dP = dO @ V^T
        dP = torch.einsum("bid,bjd->bij", dO_f, v_f)  # (batch, n_q, n_k)

        # Compute D_i = rowsum(dO * O)  (scalar per query)
        Di = (dO_f * O_f).sum(dim=-1, keepdim=True)  # (batch, n_q, 1)

        # dS = P * (dP - D_i) * scale
        dS = P * (dP - Di) * scale  # (batch, n_q, n_k)

        # dQ = dS @ K
        dQ = torch.einsum("bij,bjd->bid", dS, k_f)  # (batch, n_q, d)

        # dK = dS^T @ Q
        dK = torch.einsum("bij,bid->bjd", dS, q_f)  # (batch, n_k, d)

        return dQ.to(q.dtype), dK.to(k.dtype), dV.to(v.dtype), None


# ───────────────────────────────────────────────────────────────────────────────
# Triton-based FlashAttention2
# ───────────────────────────────────────────────────────────────────────────────

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True

    @triton.jit
    def _flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
        stride_qb, stride_qn, stride_qd,
        stride_kb, stride_kn, stride_kd,
        stride_vb, stride_vn, stride_vd,
        stride_ob, stride_on, stride_od,
        stride_lb, stride_ln,
        n_q, n_k, d, scale,
        IS_CAUSAL: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Triton kernel for FlashAttention2 forward pass."""
        # Program ids
        pid_b = tl.program_id(0)
        pid_q = tl.program_id(1)

        # Query block start
        q_start = pid_q * BLOCK_Q
        q_offs = q_start + tl.arange(0, BLOCK_Q)  # (BLOCK_Q,)
        d_offs = tl.arange(0, BLOCK_K)  # reuse BLOCK_K for d dim if d <= BLOCK_K

        # Actually use a separate constant for d — handled via masking
        d_range = tl.arange(0, BLOCK_K)

        # Pointers to Q block: shape BLOCK_Q x d
        Q_block_ptr = Q_ptr + pid_b * stride_qb + q_offs[:, None] * stride_qn + d_range[None, :] * stride_qd

        # Load Q block, mask out-of-bounds
        q_mask = (q_offs[:, None] < n_q) & (d_range[None, :] < d)
        qi = tl.load(Q_block_ptr, mask=q_mask, other=0.0).to(tl.float32)  # (BLOCK_Q, BLOCK_K)
        qi = qi * scale

        # Running stats
        mi = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
        li = tl.zeros([BLOCK_Q], dtype=tl.float32)
        oi = tl.zeros([BLOCK_Q, BLOCK_K], dtype=tl.float32)

        # Iterate over key blocks
        for j in tl.range(0, tl.cdiv(n_k, BLOCK_K)):
            k_start = j * BLOCK_K
            k_offs = k_start + tl.arange(0, BLOCK_K)

            K_block_ptr = K_ptr + pid_b * stride_kb + k_offs[:, None] * stride_kn + d_range[None, :] * stride_kd
            V_block_ptr = V_ptr + pid_b * stride_vb + k_offs[:, None] * stride_vn + d_range[None, :] * stride_vd

            kv_mask = (k_offs[:, None] < n_k) & (d_range[None, :] < d)
            kj = tl.load(K_block_ptr, mask=kv_mask, other=0.0).to(tl.float32)
            vj = tl.load(V_block_ptr, mask=kv_mask, other=0.0).to(tl.float32)

            # S_ij = qi @ kj.T  [BLOCK_Q, BLOCK_K]
            sij = tl.dot(qi, tl.trans(kj))  # (BLOCK_Q, BLOCK_K)

            # Mask out invalid k positions
            k_valid_mask = k_offs[None, :] < n_k
            sij = tl.where(k_valid_mask, sij, float("-inf"))

            # Causal mask
            if IS_CAUSAL:
                causal_mask = q_offs[:, None] >= k_offs[None, :]
                sij = tl.where(causal_mask, sij, float("-inf"))

            # Row max
            mij = tl.max(sij, axis=1)  # (BLOCK_Q,)
            pij = tl.exp(sij - mij[:, None])  # (BLOCK_Q, BLOCK_K)
            lij = tl.sum(pij, axis=1)  # (BLOCK_Q,)

            mi_new = tl.maximum(mi, mij)
            exp_mi = tl.exp(mi - mi_new)
            exp_mij = tl.exp(mij - mi_new)

            li = exp_mi * li + exp_mij * lij
            oi = exp_mi[:, None] * oi + exp_mij[:, None] * tl.dot(pij, vj)
            mi = mi_new

        # Normalize
        oi = oi / li[:, None]

        # Write output
        O_block_ptr = O_ptr + pid_b * stride_ob + q_offs[:, None] * stride_on + d_range[None, :] * stride_od
        o_mask = (q_offs[:, None] < n_q) & (d_range[None, :] < d)
        tl.store(O_block_ptr, oi, mask=o_mask)

        # Write L = mi + log(li)
        L_val = mi + tl.log(li)
        L_block_ptr = L_ptr + pid_b * stride_lb + q_offs * stride_ln
        tl.store(L_block_ptr, L_val, mask=q_offs < n_q)

    @triton.jit
    def _flash_bwd_kernel(
        Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, dO_ptr,
        dQ_ptr, dK_ptr, dV_ptr,
        stride_qb, stride_qn, stride_qd,
        stride_kb, stride_kn, stride_kd,
        stride_vb, stride_vn, stride_vd,
        stride_ob, stride_on, stride_od,
        stride_lb, stride_ln,
        stride_dob, stride_don, stride_dod,
        stride_dqb, stride_dqn, stride_dqd,
        stride_dkb, stride_dkn, stride_dkd,
        stride_dvb, stride_dvn, stride_dvd,
        n_q, n_k, d, scale,
        IS_CAUSAL: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Triton kernel for FlashAttention2 backward pass - computes dK and dV for a K-block."""
        pid_b = tl.program_id(0)
        pid_k = tl.program_id(1)

        k_start = pid_k * BLOCK_K
        k_offs = k_start + tl.arange(0, BLOCK_K)
        d_range = tl.arange(0, BLOCK_K)

        # Load K and V blocks (cast to fp32 for computation)
        KV_mask = (k_offs[:, None] < n_k) & (d_range[None, :] < d)
        kj = tl.load(K_ptr + pid_b * stride_kb + k_offs[:, None] * stride_kn + d_range[None, :] * stride_kd, mask=KV_mask, other=0.0).to(tl.float32)
        vj = tl.load(V_ptr + pid_b * stride_vb + k_offs[:, None] * stride_vn + d_range[None, :] * stride_vd, mask=KV_mask, other=0.0).to(tl.float32)

        dkj = tl.zeros([BLOCK_K, BLOCK_K], dtype=tl.float32)
        dvj = tl.zeros([BLOCK_K, BLOCK_K], dtype=tl.float32)

        for i in tl.range(0, tl.cdiv(n_q, BLOCK_Q)):
            q_start = i * BLOCK_Q
            q_offs = q_start + tl.arange(0, BLOCK_Q)
            Q_mask = (q_offs[:, None] < n_q) & (d_range[None, :] < d)

            qi  = tl.load(Q_ptr  + pid_b * stride_qb  + q_offs[:, None] * stride_qn  + d_range[None, :] * stride_qd,  mask=Q_mask, other=0.0).to(tl.float32)
            oi  = tl.load(O_ptr  + pid_b * stride_ob  + q_offs[:, None] * stride_on  + d_range[None, :] * stride_od,  mask=Q_mask, other=0.0).to(tl.float32)
            doi = tl.load(dO_ptr + pid_b * stride_dob + q_offs[:, None] * stride_don + d_range[None, :] * stride_dod, mask=Q_mask, other=0.0).to(tl.float32)
            li  = tl.load(L_ptr  + pid_b * stride_lb  + q_offs * stride_ln, mask=q_offs < n_q, other=0.0)  # (BLOCK_Q,)

            # S = qi @ kj.T * scale
            sij = tl.dot(qi, tl.trans(kj)) * scale  # (BLOCK_Q, BLOCK_K)
            k_valid = k_offs[None, :] < n_k
            sij = tl.where(k_valid, sij, float("-inf"))
            if IS_CAUSAL:
                causal_mask = q_offs[:, None] >= k_offs[None, :]
                sij = tl.where(causal_mask, sij, float("-inf"))

            pij = tl.exp(sij - li[:, None])  # (BLOCK_Q, BLOCK_K)

            # dV += P^T @ dO
            dvj += tl.dot(tl.trans(pij), doi)

            # dP = dO @ V^T
            dpij = tl.dot(doi, tl.trans(vj))  # (BLOCK_Q, BLOCK_K)

            # Di = rowsum(dO * O)
            Di = tl.sum(doi * oi, axis=1)  # (BLOCK_Q,)

            # dS = P * (dP - Di)
            dsij = pij * (dpij - Di[:, None]) * scale  # (BLOCK_Q, BLOCK_K)

            # dK += dS^T @ Q
            dkj += tl.dot(tl.trans(dsij), qi)

        # Write back dK, dV
        dK_block_ptr = dK_ptr + pid_b * stride_dkb + k_offs[:, None] * stride_dkn + d_range[None, :] * stride_dkd
        dV_block_ptr = dV_ptr + pid_b * stride_dvb + k_offs[:, None] * stride_dvn + d_range[None, :] * stride_dvd
        tl.store(dK_block_ptr, dkj, mask=KV_mask)
        tl.store(dV_block_ptr, dvj, mask=KV_mask)

    @triton.jit
    def _flash_bwd_dq_kernel(
        Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, dO_ptr, dQ_ptr,
        stride_qb, stride_qn, stride_qd,
        stride_kb, stride_kn, stride_kd,
        stride_vb, stride_vn, stride_vd,
        stride_ob, stride_on, stride_od,
        stride_lb, stride_ln,
        stride_dob, stride_don, stride_dod,
        stride_dqb, stride_dqn, stride_dqd,
        n_q, n_k, d, scale,
        IS_CAUSAL: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Triton kernel to compute dQ for a Q-block."""
        pid_b = tl.program_id(0)
        pid_q = tl.program_id(1)

        q_start = pid_q * BLOCK_Q
        q_offs = q_start + tl.arange(0, BLOCK_Q)
        d_range = tl.arange(0, BLOCK_K)

        Q_mask = (q_offs[:, None] < n_q) & (d_range[None, :] < d)
        qi  = tl.load(Q_ptr  + pid_b * stride_qb  + q_offs[:, None] * stride_qn  + d_range[None, :] * stride_qd,  mask=Q_mask, other=0.0).to(tl.float32)
        oi  = tl.load(O_ptr  + pid_b * stride_ob  + q_offs[:, None] * stride_on  + d_range[None, :] * stride_od,  mask=Q_mask, other=0.0).to(tl.float32)
        doi = tl.load(dO_ptr + pid_b * stride_dob + q_offs[:, None] * stride_don + d_range[None, :] * stride_dod, mask=Q_mask, other=0.0).to(tl.float32)
        li  = tl.load(L_ptr  + pid_b * stride_lb  + q_offs * stride_ln, mask=q_offs < n_q, other=0.0)

        dqi = tl.zeros([BLOCK_Q, BLOCK_K], dtype=tl.float32)
        Di = tl.sum(doi * oi, axis=1)  # (BLOCK_Q,)

        for j in tl.range(0, tl.cdiv(n_k, BLOCK_K)):
            k_start = j * BLOCK_K
            k_offs = k_start + tl.arange(0, BLOCK_K)
            KV_mask = (k_offs[:, None] < n_k) & (d_range[None, :] < d)
            kj = tl.load(K_ptr + pid_b * stride_kb + k_offs[:, None] * stride_kn + d_range[None, :] * stride_kd, mask=KV_mask, other=0.0).to(tl.float32)
            vj = tl.load(V_ptr + pid_b * stride_vb + k_offs[:, None] * stride_vn + d_range[None, :] * stride_vd, mask=KV_mask, other=0.0).to(tl.float32)

            sij = tl.dot(qi, tl.trans(kj)) * scale
            k_valid = k_offs[None, :] < n_k
            sij = tl.where(k_valid, sij, float("-inf"))
            if IS_CAUSAL:
                causal_mask = q_offs[:, None] >= k_offs[None, :]
                sij = tl.where(causal_mask, sij, float("-inf"))

            pij = tl.exp(sij - li[:, None])

            dpij = tl.dot(doi, tl.trans(vj))
            dsij = pij * (dpij - Di[:, None]) * scale

            dqi += tl.dot(dsij, kj)

        dQ_block_ptr = dQ_ptr + pid_b * stride_dqb + q_offs[:, None] * stride_dqn + d_range[None, :] * stride_dqd
        tl.store(dQ_block_ptr, dqi, mask=Q_mask)

    def _flash_fwd_triton(q, k, v, is_causal):
        """Launch the Triton forward kernel."""
        batch, n_q, d = q.shape
        n_k = k.shape[1]
        scale = float(d ** -0.5)

        # Determine block sizes (must be power of 2)
        BLOCK_K = max(16, min(64, triton.next_power_of_2(d)))
        BLOCK_Q = BLOCK_K  # square blocks

        # Allocate output tensors (fp32, same as inputs)
        O = torch.empty_like(q)
        L = torch.empty((batch, n_q), device=q.device, dtype=torch.float32)

        # Keep fp32 — kernels cast internally
        qf = q.contiguous()
        kf = k.contiguous()
        vf = v.contiguous()

        grid = (batch, triton.cdiv(n_q, BLOCK_Q))
        _flash_fwd_kernel[grid](
            qf, kf, vf, O, L,
            qf.stride(0), qf.stride(1), qf.stride(2),
            kf.stride(0), kf.stride(1), kf.stride(2),
            vf.stride(0), vf.stride(1), vf.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            n_q, n_k, d, scale,
            IS_CAUSAL=is_causal,
            BLOCK_Q=BLOCK_Q,
            BLOCK_K=BLOCK_K,
        )
        return O, L

    def _flash_bwd_triton(q, k, v, O, L, dO, is_causal):
        """Launch Triton backward kernels."""
        batch, n_q, d = q.shape
        n_k = k.shape[1]
        scale = float(d ** -0.5)

        BLOCK_K = max(16, min(64, triton.next_power_of_2(d)))
        BLOCK_Q = BLOCK_K

        dQ = torch.zeros_like(q, dtype=torch.float32)
        dK = torch.zeros_like(k, dtype=torch.float32)
        dV = torch.zeros_like(v, dtype=torch.float32)

        # Keep fp32 — kernels cast internally
        qf  = q.contiguous()
        kf  = k.contiguous()
        vf  = v.contiguous()
        Of  = O.contiguous()
        dOf = dO.contiguous()

        # Compute dK, dV
        grid_k = (batch, triton.cdiv(n_k, BLOCK_K))
        _flash_bwd_kernel[grid_k](
            qf, kf, vf, Of, L, dOf,
            dQ, dK, dV,
            qf.stride(0),  qf.stride(1),  qf.stride(2),
            kf.stride(0),  kf.stride(1),  kf.stride(2),
            vf.stride(0),  vf.stride(1),  vf.stride(2),
            Of.stride(0),  Of.stride(1),  Of.stride(2),
            L.stride(0),   L.stride(1),
            dOf.stride(0), dOf.stride(1), dOf.stride(2),
            dQ.stride(0),  dQ.stride(1),  dQ.stride(2),
            dK.stride(0),  dK.stride(1),  dK.stride(2),
            dV.stride(0),  dV.stride(1),  dV.stride(2),
            n_q, n_k, d, scale,
            IS_CAUSAL=is_causal,
            BLOCK_Q=BLOCK_Q,
            BLOCK_K=BLOCK_K,
        )

        # Compute dQ
        grid_q = (batch, triton.cdiv(n_q, BLOCK_Q))
        _flash_bwd_dq_kernel[grid_q](
            qf, kf, vf, Of, L, dOf, dQ,
            qf.stride(0),  qf.stride(1),  qf.stride(2),
            kf.stride(0),  kf.stride(1),  kf.stride(2),
            vf.stride(0),  vf.stride(1),  vf.stride(2),
            Of.stride(0),  Of.stride(1),  Of.stride(2),
            L.stride(0),   L.stride(1),
            dOf.stride(0), dOf.stride(1), dOf.stride(2),
            dQ.stride(0),  dQ.stride(1),  dQ.stride(2),
            n_q, n_k, d, scale,
            IS_CAUSAL=is_causal,
            BLOCK_Q=BLOCK_Q,
            BLOCK_K=BLOCK_K,
        )

        return dQ.to(q.dtype), dK.to(k.dtype), dV.to(v.dtype)

    class FlashAttentionTriton(torch.autograd.Function):
        """
        FlashAttention2 using Triton kernels for the forward and backward passes.
        """

        @staticmethod
        def forward(ctx, q, k, v, is_causal=False):
            O, L = _flash_fwd_triton(q, k, v, is_causal)
            ctx.save_for_backward(q, k, v, O, L)
            ctx.is_causal = is_causal
            return O

        @staticmethod
        def backward(ctx, dO):
            q, k, v, O, L = ctx.saved_tensors
            is_causal = ctx.is_causal
            dQ, dK, dV = _flash_bwd_triton(q, k, v, O, L, dO, is_causal)
            return dQ, dK, dV, None

except ImportError:
    _TRITON_AVAILABLE = False

    class FlashAttentionTriton(torch.autograd.Function):  # type: ignore[no-redef]
        """Stub when Triton is not available."""

        @staticmethod
        def forward(ctx, q, k, v, is_causal=False):
            raise RuntimeError("Triton is not installed. Cannot run FlashAttentionTriton.")

        @staticmethod
        def backward(ctx, dO):
            raise RuntimeError("Triton is not installed. Cannot run FlashAttentionTriton.")
