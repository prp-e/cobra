# model/model.py
"""Full Mamba LM: embedding -> N pre-norm residual Mamba blocks -> norm ->
tied lm_head. No attention, no positional embeddings (SSM handles position
implicitly via the recurrence)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.mamba_block import MambaBlock, RMSNorm


class Layer(nn.Module):
    """Pre-norm residual wrapper around one MambaBlock: x = x + mixer(norm(x))."""

    def __init__(self, cfg, n_layer_total, scan_mode="chunked"):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.mixer = MambaBlock(cfg, n_layer_total, scan_mode=scan_mode)

    def forward(self, x):
        # x: (Bsz, L, d_model)
        return x + self.mixer(self.norm(x))  # (Bsz, L, d_model)

    def step(self, x, state):
        # x: (Bsz, 1, d_model)
        y, new_state = self.mixer.step(self.norm(x), state)  # (Bsz,1,d_model)
        return x + y, new_state


class MambaLM(nn.Module):
    def __init__(self, cfg, scan_mode: str = "chunked"):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)   # (vocab, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        self.layers = nn.ModuleList(
            [Layer(cfg, cfg.n_layer, scan_mode=scan_mode) for _ in range(cfg.n_layer)]
        )
        self.norm_f = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # weight tying

    def forward(self, idx, targets=None):
        """idx: (Bsz, L) int64 token ids. targets: (Bsz, L) int64 or None.
        returns logits (Bsz, L, vocab_size), loss (scalar or None).
        """
        x = self.embedding(idx)              # (Bsz, L, d_model)
        for layer in self.layers:
            x = layer(x)                      # (Bsz, L, d_model)
        x = self.norm_f(x)                    # (Bsz, L, d_model)
        logits = self.lm_head(x)              # (Bsz, L, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),  # (Bsz*L, vocab_size) fp32 for stable loss
                targets.reshape(-1),                            # (Bsz*L,)
            )
        return logits, loss

    # ---------------- recurrent single-step inference API ----------------

    def allocate_inference_cache(self, batch_size, device, dtype=torch.float32):
        return [layer.mixer.allocate_state(batch_size, device, dtype=dtype) for layer in self.layers]

    def step(self, idx_last, cache):
        """idx_last: (Bsz, 1) int64. cache: list of per-layer state dicts.
        Returns logits (Bsz, 1, vocab_size), updated cache.
        """
        x = self.embedding(idx_last)          # (Bsz, 1, d_model)
        for i, layer in enumerate(self.layers):
            x, cache[i] = layer.step(x, cache[i])  # (Bsz, 1, d_model)
        x = self.norm_f(x)                     # (Bsz, 1, d_model)
        logits = self.lm_head(x)               # (Bsz, 1, vocab_size)
        return logits, cache

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embedding.weight.numel()
        return n