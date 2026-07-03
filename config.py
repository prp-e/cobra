# config.py
"""Single source of truth for model/training hyperparameters.

Import as:
    from config import get_model_config, get_train_config
"""
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size_raw: int = 50257          # GPT-2 BPE vocab (incl. <|endoftext|>=50256)
    vocab_pad_multiple: int = 64         # pad vocab up to a multiple of this for tensor-core friendliness
    d_model: int = 2048
    n_layer: int = 48
    expand: int = 2                       # E: d_inner = expand * d_model
    d_state: int = 16                     # N
    d_conv: int = 4                       # causal depthwise conv kernel size
    dt_rank: Optional[int] = None         # if None, set to ceil(d_model/16)
    seq_len: int = 2048
    rms_norm_eps: float = 1e-5
    dt_min: float = 1e-3
    dt_max: float = 1e-1
    scan_chunk_size: int = 64             # chunked parallel scan chunk length (64-128)

    def __post_init__(self):
        if self.dt_rank is None:
            self.dt_rank = math.ceil(self.d_model / 16)
        self.d_inner = self.expand * self.d_model
        pad = self.vocab_pad_multiple
        self.vocab_size = ((self.vocab_size_raw + pad - 1) // pad) * pad  # 50257 -> 50304


def get_model_config(size: str = "1.4B") -> ModelConfig:
    """Config switch. size in {"1.4B", "2.8B"}."""
    if size == "1.4B":
        return ModelConfig(d_model=2048, n_layer=48)
    elif size == "2.8B":
        return ModelConfig(d_model=2560, n_layer=64)
    else:
        raise ValueError(f"unknown size {size!r}, expected '1.4B' or '2.8B'")


@dataclass
class TrainConfig:
    model_size: str = "1.4B"

    # optimizer
    peak_lr: float = 3e-4               # 3e-4 for 1.4B, use 2.5e-4 for 2.8B (set in get_train_config)
    min_lr_ratio: float = 0.10          # cosine decays down to 10% of peak
    warmup_ratio: float = 0.015         # 1.5% of total steps, within the 1-2% spec range
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip_norm: float = 1.0

    # batch / schedule
    seq_len: int = 2048
    global_batch_tokens: int = 524288   # 2**19 ~ 0.5M tokens/step (within 0.5M-1M target)
    micro_batch_size: int = 8           # sequences per forward/backward; tune per-GPU (see train.py arithmetic)
    max_tokens: int = 100_000_000_000   # ~100B tokens, one epoch over sample-100BT

    # checkpoint / eval / logging cadence, expressed in *tokens* (converted to steps in train.py)
    checkpoint_every_tokens: int = 3_000_000_000   # every 2-5B tokens
    eval_every_tokens: int = 500_000_000
    sample_every_tokens: int = 500_000_000
    log_every_steps: int = 10

    # io
    data_meta_path: str = "data/cache/meta.json"
    out_dir: str = "checkpoints"
    log_path: str = "checkpoints/train_log.jsonl"

    # grad-accum arithmetic (computed, not stored as free params):
    #   grad_accum_steps = global_batch_tokens // (micro_batch_size * seq_len)
    #   e.g. B200 80-180GB : micro_batch_size=32 -> 32*2048=65536 tok/microbatch
    #                        grad_accum_steps = 524288 / 65536 = 8
    #        A100 80GB     : micro_batch_size=8  -> 8*2048=16384 tok/microbatch
    #                        grad_accum_steps = 524288 / 16384 = 32
    #        4090 24GB     : micro_batch_size=2  -> 2*2048=4096 tok/microbatch
    #                        grad_accum_steps = 524288 / 4096 = 128
    #   Same *architecture* and same *effective* global batch everywhere;
    #   only micro_batch_size + grad_accum_steps change per GPU.


def get_train_config(model_size: str = "1.4B", **overrides) -> TrainConfig:
    if model_size == "1.4B":
        cfg = TrainConfig(model_size="1.4B", peak_lr=3e-4)
    elif model_size == "2.8B":
        cfg = TrainConfig(model_size="2.8B", peak_lr=2.5e-4)
    else:
        raise ValueError(f"unknown size {model_size!r}")
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def grad_accum_steps(global_batch_tokens: int, micro_batch_size: int, seq_len: int) -> int:
    tokens_per_microbatch = micro_batch_size * seq_len
    assert global_batch_tokens % tokens_per_microbatch == 0, (
        f"global_batch_tokens={global_batch_tokens} must be divisible by "
        f"micro_batch_size*seq_len={tokens_per_microbatch}"
    )
    steps = global_batch_tokens // tokens_per_microbatch
    return max(steps, 1)