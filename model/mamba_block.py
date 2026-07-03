# model/mamba_block.py
"""Core Mamba mixer block (selection + causal conv + selective scan)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.mamba_scan import selective_scan


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))  # (d_model,) -- excluded from weight decay (ndim=1)

    def forward(self, x):
        # x: (Bsz, L, d_model)
        var = x.pow(2).mean(dim=-1, keepdim=True)                # (Bsz, L, 1)
        x_norm = x * torch.rsqrt(var + self.eps)                  # (Bsz, L, d_model)
        return x_norm * self.weight                                # (Bsz, L, d_model)


def causal_depthwise_conv1d(x, weight, bias):
    """Self-implemented causal depthwise conv1d: left-pad only (kernel-1
    zeros), then a grouped conv1d with groups == channels (depthwise).
    No causal-conv1d package used -- just F.pad + F.conv1d primitives.

    x:      (Bsz, d_inner, L)
    weight: (d_inner, 1, K)
    bias:   (d_inner,)
    returns:(Bsz, d_inner, L)
    """
    K = weight.shape[-1]
    x_pad = F.pad(x, (K - 1, 0))               # (Bsz, d_inner, L+K-1) -- left pad only, causal
    y = F.conv1d(x_pad, weight, bias=bias, groups=x.shape[1])  # (Bsz, d_inner, L)
    return y


class MambaBlock(nn.Module):
    """One Mamba mixer (no residual/norm -- those live in model.py's Layer).

    x_in, z = in_proj(x).chunk(2)
    x_in = SiLU(causal_depthwise_conv1d(x_in))
    x_dbl = x_proj(x_in) -> split (delta_raw, B_sel, C_sel)
    delta = softplus(dt_proj(delta_raw))     # dt_proj.bias == "dt_bias" in the spec
    A = -exp(A_log)                           # (d_inner, N), S4D-real init
    y, _ = selective_scan(x_in, delta, A, B_sel, C_sel, D)
    y = y * SiLU(z)
    out = out_proj(y)
    """

    def __init__(self, cfg, n_layer_total: int, scan_mode: str = "chunked"):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.d_inner = cfg.d_inner
        self.d_state = cfg.d_state          # N
        self.d_conv = cfg.d_conv            # K
        self.dt_rank = cfg.dt_rank
        self.scan_mode = scan_mode
        self.scan_chunk_size = cfg.scan_chunk_size

        d_inner, d_state, d_conv, dt_rank, d_model = (
            self.d_inner, self.d_state, self.d_conv, self.dt_rank, self.d_model
        )

        # --- input projection: splits into (x_in, z), each d_inner wide ---
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        nn.init.normal_(self.in_proj.weight, mean=0.0, std=0.02)

        # --- causal depthwise conv over the x_in branch ---
        self.conv1d_weight = nn.Parameter(torch.empty(d_inner, 1, d_conv))  # (d_inner,1,K)
        nn.init.normal_(self.conv1d_weight, mean=0.0, std=0.02)
        self.conv1d_bias = nn.Parameter(torch.zeros(d_inner))              # (d_inner,) -- excluded from wd

        # --- selection projection: x_in -> (delta_raw, B_sel, C_sel) ---
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        nn.init.normal_(self.x_proj.weight, mean=0.0, std=0.02)

        # --- delta_raw -> delta (dt_proj.bias plays the role of "dt_bias") ---
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        nn.init.normal_(self.dt_proj.weight, mean=0.0, std=dt_rank ** -0.5)
        dt_min, dt_max = cfg.dt_min, cfg.dt_max
        # sample target deltas log-uniformly in [dt_min, dt_max], invert softplus for the bias
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # numerically stable softplus^{-1}
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)  # (d_inner,) -- excluded from weight decay (ndim=1)

        # --- S4D-real init: A_log = log(arange(1,N+1)) broadcast over d_inner; A = -exp(A_log) ---
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_inner, 1)  # (d_inner,N)
        self.A_log = nn.Parameter(torch.log(A_init))  # (d_inner, N) -- excluded from weight decay by name
        self.D = nn.Parameter(torch.ones(d_inner))     # (d_inner,) -- excluded from weight decay

        # --- output projection, scaled by 1/sqrt(n_layer) at init ---
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.out_proj.weight.div_(math.sqrt(n_layer_total))

    def _selection(self, x_in):
        """x_in: (Bsz, L, d_inner) -> delta, B_sel, C_sel"""
        x_dbl = self.x_proj(x_in)  # (Bsz, L, dt_rank + 2N)
        delta_raw, B_sel, C_sel = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )  # (Bsz,L,dt_rank), (Bsz,L,N), (Bsz,L,N)
        delta = F.softplus(self.dt_proj(delta_raw))  # (Bsz, L, d_inner); dt_proj bias == dt_bias
        return delta, B_sel, C_sel

    def forward(self, x):
        """x: (Bsz, L, d_model) -> (Bsz, L, d_model). Uses the parallel
        chunked scan (fast path for training / prefill over full sequences).
        """
        Bsz, L, _ = x.shape
        xz = self.in_proj(x)                                   # (Bsz, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)                            # (Bsz,L,d_inner) each

        x_in_t = x_in.transpose(1, 2)                            # (Bsz, d_inner, L)
        x_in_t = causal_depthwise_conv1d(x_in_t, self.conv1d_weight, self.conv1d_bias)  # (Bsz,d_inner,L)
        x_in = F.silu(x_in_t.transpose(1, 2))                    # (Bsz, L, d_inner)

        delta, B_sel, C_sel = self._selection(x_in)              # (Bsz,L,d_inner),(Bsz,L,N),(Bsz,L,N)
        A = -torch.exp(self.A_log)                                # (d_inner, N)

        y, _ = selective_scan(
            x_in, delta, A, B_sel, C_sel, self.D,
            mode=self.scan_mode, chunk_size=self.scan_chunk_size,
        )  # (Bsz, L, d_inner)

        y = y * F.silu(z)                                         # (Bsz, L, d_inner)
        out = self.out_proj(y)                                    # (Bsz, L, d_model)
        return out

    # ---------------- recurrent single-step inference path ----------------

    def allocate_state(self, batch_size, device, dtype=torch.float32):
        conv_state = torch.zeros(batch_size, self.d_inner, self.d_conv - 1, device=device, dtype=dtype)
        ssm_state = torch.zeros(batch_size, self.d_inner, self.d_state, device=device, dtype=dtype)
        return {"conv_state": conv_state, "ssm_state": ssm_state}

    def step(self, x, state):
        """x: (Bsz, 1, d_model) single time step. state: {"conv_state":(Bsz,d_inner,K-1),
        "ssm_state":(Bsz,d_inner,N)}. Returns (out, new_state), out: (Bsz,1,d_model).
        """
        Bsz = x.shape[0]
        xz = self.in_proj(x.squeeze(1))                # (Bsz, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)                    # (Bsz, d_inner) each

        conv_state = state["conv_state"]                 # (Bsz, d_inner, K-1)
        conv_in = torch.cat([conv_state, x_in.unsqueeze(-1)], dim=-1)  # (Bsz, d_inner, K)
        w = self.conv1d_weight.squeeze(1)                 # (d_inner, K)
        conv_out = (conv_in * w.unsqueeze(0)).sum(dim=-1) + self.conv1d_bias.unsqueeze(0)  # (Bsz,d_inner)
        new_conv_state = conv_in[:, :, 1:]                 # (Bsz, d_inner, K-1) drop oldest
        x_in = F.silu(conv_out)                             # (Bsz, d_inner)

        delta, B_sel, C_sel = self._selection(x_in.unsqueeze(1))  # (Bsz,1,d_inner),(Bsz,1,N),(Bsz,1,N)
        delta_t = delta.squeeze(1)   # (Bsz, d_inner)
        B_t = B_sel.squeeze(1)       # (Bsz, N)
        C_t = C_sel.squeeze(1)       # (Bsz, N)

        A = -torch.exp(self.A_log)                          # (d_inner, N)
        h = state["ssm_state"]                                # (Bsz, d_inner, N)
        deltaA = delta_t.unsqueeze(-1) * A.unsqueeze(0)        # (Bsz, d_inner, N)
        A_bar = torch.exp(deltaA)                              # (Bsz, d_inner, N)
        B_bar = delta_t.unsqueeze(-1) * B_t.unsqueeze(1)        # (Bsz, d_inner, N)
        new_h = A_bar * h + B_bar * x_in.unsqueeze(-1)          # (Bsz, d_inner, N)
        y_t = torch.einsum("bdn,bn->bd", new_h, C_t) + self.D.unsqueeze(0) * x_in  # (Bsz, d_inner)

        y_t = y_t * F.silu(z)                                    # (Bsz, d_inner)
        out = self.out_proj(y_t).unsqueeze(1)                     # (Bsz, 1, d_model)

        new_state = {"conv_state": new_conv_state, "ssm_state": new_h}
        return out, new_state