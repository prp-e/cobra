# model/mamba_scan.py
"""Selective-scan implementations, pure PyTorch, no custom kernels.

Discretization (ZOH, per spec):
    A_bar = exp(delta * A)                (elementwise over d_inner,N)
    B_bar ~= delta * B                    (Euler approx, standard in ref impl)
Recurrence:
    h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
    y_t = sum_N(C_t * h_t) + D * x_t

Shapes used throughout:
    x, delta        : (Bsz, L, d_inner)
    A                : (d_inner, N)            -- input-independent, negative real
    B_sel, C_sel     : (Bsz, L, N)              -- shared across d_inner channels
    D                : (d_inner,)
    h (state)        : (Bsz, d_inner, N)
"""
import torch


def selective_scan_naive(x, delta, A, B_sel, C_sel, D, h0=None):
    """Reference sequential scan (Python loop over time). O(L) python
    iterations; used for correctness testing and single-token recurrent
    inference (called once per generated token with L=1).
    """
    Bsz, L, d_inner = x.shape
    N = A.shape[-1]
    orig_dtype = x.dtype
    # numerically sensitive recurrence: always compute in fp32
    x_f = x.float()
    delta_f = delta.float()
    A_f = A.float()
    B_f = B_sel.float()
    C_f = C_sel.float()
    D_f = D.float()

    h = torch.zeros(Bsz, d_inner, N, device=x.device, dtype=torch.float32) if h0 is None else h0.float()
    ys = []
    for t in range(L):
        x_t = x_f[:, t, :]        # (Bsz, d_inner)
        delta_t = delta_f[:, t, :]  # (Bsz, d_inner)
        B_t = B_f[:, t, :]        # (Bsz, N)
        C_t = C_f[:, t, :]        # (Bsz, N)

        deltaA = delta_t.unsqueeze(-1) * A_f.unsqueeze(0)            # (Bsz, d_inner, N)
        A_bar = torch.exp(deltaA)                                    # (Bsz, d_inner, N)
        B_bar = delta_t.unsqueeze(-1) * B_t.unsqueeze(1)              # (Bsz, d_inner, N)
        h = A_bar * h + B_bar * x_t.unsqueeze(-1)                     # (Bsz, d_inner, N)
        y_t = torch.einsum("bdn,bn->bd", h, C_t) + D_f.unsqueeze(0) * x_t  # (Bsz, d_inner)
        ys.append(y_t)

    y = torch.stack(ys, dim=1)  # (Bsz, L, d_inner)
    return y.to(orig_dtype), h.to(orig_dtype)


def selective_scan_chunked(x, delta, A, B_sel, C_sel, D, chunk_size=64, h0=None):
    """Chunked parallel scan: within a chunk, use a log-space cumsum trick to
    compute all intra-chunk states in one shot via batched matmul-free
    elementwise ops (no python loop over time within a chunk); the state is
    carried sequentially only *across* chunks (L/chunk_size python
    iterations instead of L).

    Derivation (per chunk, local time t=1..Lc, carry-in h0):
        h_t = Abar_{1:t} * h0 + sum_{s<=t} Abar_{s+1:t} * Bbar_s * x_s
        let S_t = cumsum_{k<=t}(delta_k * A)          (log of running product)
        Abar_{1:t} = exp(S_t)
        Abar_{s+1:t} = exp(S_t - S_s)
        => h_t = exp(S_t) * ( h0 + cumsum_{s<=t}[ exp(-S_s) * Bbar_s * x_s ] )
    """
    Bsz, L, d_inner = x.shape
    N = A.shape[-1]
    orig_dtype = x.dtype
    x_f = x.float()
    delta_f = delta.float()
    A_f = A.float()
    B_f = B_sel.float()
    C_f = C_sel.float()
    D_f = D.float()

    h = torch.zeros(Bsz, d_inner, N, device=x.device, dtype=torch.float32) if h0 is None else h0.float()
    ys = []

    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        xc = x_f[:, start:end, :]        # (Bsz, Lc, d_inner)
        deltac = delta_f[:, start:end, :]  # (Bsz, Lc, d_inner)
        Bc = B_f[:, start:end, :]        # (Bsz, Lc, N)
        Cc = C_f[:, start:end, :]        # (Bsz, Lc, N)

        deltaA = deltac.unsqueeze(-1) * A_f.view(1, 1, d_inner, N)   # (Bsz, Lc, d_inner, N)
        S = torch.cumsum(deltaA, dim=1)                               # (Bsz, Lc, d_inner, N) running log-product
        Abar_cum = torch.exp(S)                                       # (Bsz, Lc, d_inner, N)

        h0_term = Abar_cum * h.unsqueeze(1)                           # (Bsz, Lc, d_inner, N)

        deltaB_x = deltac.unsqueeze(-1) * Bc.unsqueeze(2) * xc.unsqueeze(-1)  # (Bsz, Lc, d_inner, N)
        u = deltaB_x * torch.exp(-S)                                  # (Bsz, Lc, d_inner, N) rescaled increment
        u_cumsum = torch.cumsum(u, dim=1)                              # (Bsz, Lc, d_inner, N)
        in_chunk_term = Abar_cum * u_cumsum                            # (Bsz, Lc, d_inner, N)

        h_all = h0_term + in_chunk_term                                # (Bsz, Lc, d_inner, N) state at every t in chunk
        y_chunk = torch.einsum("bldn,bln->bld", h_all, Cc) + D_f.view(1, 1, d_inner) * xc  # (Bsz, Lc, d_inner)
        ys.append(y_chunk)

        h = h_all[:, -1, :, :]  # (Bsz, d_inner, N) carry to next chunk

    y = torch.cat(ys, dim=1)  # (Bsz, L, d_inner)
    return y.to(orig_dtype), h.to(orig_dtype)


def selective_scan(x, delta, A, B_sel, C_sel, D, mode="chunked", chunk_size=64, h0=None):
    """Dispatch. mode in {"naive", "chunked"}. Returns (y, h_final)."""
    if mode == "naive":
        return selective_scan_naive(x, delta, A, B_sel, C_sel, D, h0=h0)
    elif mode == "chunked":
        return selective_scan_chunked(x, delta, A, B_sel, C_sel, D, chunk_size=chunk_size, h0=h0)
    else:
        raise ValueError(f"unknown scan mode {mode!r}")