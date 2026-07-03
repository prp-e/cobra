# train.py
"""Single-GPU training loop (B200/H200/A100/4090 via micro-batch + grad
accum, never via architecture changes). bf16 autocast forward/backward,
fp32 master weights + optimizer state (no GradScaler needed since we never
cast params themselves to fp16/bf16). Per-layer activation checkpointing
is ON by default -- without it, the pure-PyTorch chunked scan's
intermediates (O(Bsz*L*d_inner*N) per layer) all stay resident
simultaneously across every layer until backward starts, which OOMs even
at batch size 1 on a single GPU for a 48-layer model. Fully resumable
after preemption."""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch

from config import get_model_config, get_train_config, grad_accum_steps
from data.dataset import build_loader
from model.model import MambaLM
from evaluate import evaluate_perplexity
from sample import generate, FIXED_PROMPTS


def get_param_groups(model, weight_decay):
    """Decay all matrix params (embeddings, in/out/x/dt_proj weights); no
    decay for biases, RMSNorm weights, A_log, D (ndim<2 catches biases/norms/
    D; explicit name check catches A_log, which is 2D: (d_inner, N))."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "A_log" in name or name.endswith(".D"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def lr_at_step(step, total_steps, warmup_steps, peak_lr, min_lr_ratio):
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = peak_lr * min_lr_ratio
    return min_lr + (peak_lr - min_lr) * cosine


def save_checkpoint(path, model, optimizer, step, tokens_seen, data_gen, tcfg, mcfg):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "data_generator_state": data_gen.get_state(),
        "train_config": tcfg.__dict__,
        "model_config": mcfg.__dict__,
    }
    tmp_path = path + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)  # atomic-ish on same filesystem
    print(f"[checkpoint] saved {path} (step={step}, tokens_seen={tokens_seen})")


def load_checkpoint(path, model, optimizer, data_gen, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    torch.set_rng_state(ckpt["torch_rng_state"].cpu())
    if ckpt.get("cuda_rng_state_all") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state_all"])
    random.setstate(ckpt["python_rng_state"])
    np.random.set_state(ckpt["numpy_rng_state"])
    data_gen.set_state(ckpt["data_generator_state"])
    return ckpt["step"], ckpt["tokens_seen"]


def latest_checkpoint(out_dir):
    if not os.path.isdir(out_dir):
        return None
    ckpts = [f for f in os.listdir(out_dir) if f.startswith("ckpt_step") and f.endswith(".pt")]
    if not ckpts:
        return None
    ckpts.sort(key=lambda s: int(s[len("ckpt_step"):-len(".pt")]))
    return os.path.join(out_dir, ckpts[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_size", type=str, default="1.4B", choices=["1.4B", "2.8B"])
    ap.add_argument("--micro_batch_size", type=int, default=8, help="sequences per microbatch; tune per GPU")
    ap.add_argument("--global_batch_tokens", type=int, default=524288)
    ap.add_argument("--max_tokens", type=int, default=None, help="override TrainConfig.max_tokens (e.g. for smoke tests)")
    ap.add_argument("--data_meta_path", type=str, default="data/cache/meta.json")
    ap.add_argument("--out_dir", type=str, default="checkpoints")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--scan_mode", type=str, default="chunked", choices=["chunked", "naive"])
    ap.add_argument("--grad_checkpointing", type=lambda s: s.lower() != "false", default=True,
                     help="per-layer activation checkpointing (default True); disable only if you "
                          "have verified you have enough memory without it -- required at typical "
                          "batch sizes since the pure-PyTorch scan's intermediates are O(n_layer) "
                          "resident simultaneously without it")
    args = ap.parse_args()

    mcfg = get_model_config(args.model_size)
    tcfg = get_train_config(
        args.model_size,
        micro_batch_size=args.micro_batch_size,
        global_batch_tokens=args.global_batch_tokens,
        data_meta_path=args.data_meta_path,
        out_dir=args.out_dir,
    )
    if args.max_tokens is not None:
        tcfg.max_tokens = args.max_tokens

    os.makedirs(tcfg.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    accum_steps = grad_accum_steps(tcfg.global_batch_tokens, tcfg.micro_batch_size, tcfg.seq_len)
    tokens_per_step = tcfg.global_batch_tokens
    total_steps = tcfg.max_tokens // tokens_per_step
    warmup_steps = max(1, int(total_steps * tcfg.warmup_ratio))
    print(f"[train] accum_steps={accum_steps} tokens_per_step={tokens_per_step} "
          f"total_steps={total_steps} warmup_steps={warmup_steps} "
          f"grad_checkpointing={args.grad_checkpointing}")

    model = MambaLM(
        mcfg, scan_mode=args.scan_mode, use_grad_checkpointing=args.grad_checkpointing
    ).to(device)
    print(f"[train] model params: {model.num_params()/1e9:.3f}B "
          f"(non-embedding: {model.num_params(non_embedding=True)/1e9:.3f}B)")

    param_groups = get_param_groups(model, tcfg.weight_decay)
    optimizer = torch.optim.AdamW(
        param_groups, lr=tcfg.peak_lr, betas=(tcfg.beta1, tcfg.beta2), eps=1e-8
    )

    data_gen = torch.Generator()
    data_gen.manual_seed(args.seed)
    train_ds, train_loader = build_loader(
        tcfg.data_meta_path, "train", tcfg.seq_len, tcfg.micro_batch_size,
        generator=data_gen, num_workers=args.num_workers,
    )
    val_ds, _ = build_loader(
        tcfg.data_meta_path, "val", tcfg.seq_len, tcfg.micro_batch_size,
        generator=torch.Generator().manual_seed(999), num_workers=0,
    )

    step, tokens_seen = 0, 0
    if args.resume:
        ckpt_path = latest_checkpoint(tcfg.out_dir)
        if ckpt_path is not None:
            step, tokens_seen = load_checkpoint(ckpt_path, model, optimizer, data_gen, device)
            print(f"[resume] loaded {ckpt_path}: step={step}, tokens_seen={tokens_seen}")
        else:
            print("[resume] no checkpoint found, starting fresh")

    log_f = open(tcfg.log_path, "a")

    def log_jsonl(record):
        record["ts"] = time.time()
        log_f.write(json.dumps(record) + "\n")
        log_f.flush()

    next_ckpt_at = tokens_seen + tcfg.checkpoint_every_tokens
    next_eval_at = tokens_seen + tcfg.eval_every_tokens
    next_sample_at = tokens_seen + tcfg.sample_every_tokens

    train_iter = iter(train_loader)
    model.train()
    t0 = time.time()

    while tokens_seen < tcfg.max_tokens and step < total_steps:
        lr = lr_at_step(step, total_steps, warmup_steps, tcfg.peak_lr, tcfg.min_lr_ratio)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro in range(accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x = x.to(device, non_blocking=True)  # (micro_bs, seq_len)
            y = y.to(device, non_blocking=True)  # (micro_bs, seq_len)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                logits, loss = model(x, targets=y)  # logits:(micro_bs,seq_len,vocab), loss: scalar
            loss = loss / accum_steps
            loss.backward()
            loss_accum += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip_norm)
        optimizer.step()

        step += 1
        tokens_seen += tokens_per_step

        if step % tcfg.log_every_steps == 0:
            dt = time.time() - t0
            toks_per_sec = tokens_per_step * tcfg.log_every_steps / max(dt, 1e-6)
            print(f"step={step} tokens={tokens_seen/1e9:.3f}B loss={loss_accum:.4f} "
                  f"lr={lr:.2e} grad_norm={float(grad_norm):.2f} tok/s={toks_per_sec:.0f}")
            log_jsonl({"event": "train_step", "step": step, "tokens_seen": tokens_seen,
                       "loss": loss_accum, "lr": lr, "grad_norm": float(grad_norm),
                       "tokens_per_sec": toks_per_sec})
            t0 = time.time()

        if tokens_seen >= next_eval_at:
            model.eval()
            ppl = evaluate_perplexity(model, val_ds, device, num_batches=50, batch_size=tcfg.micro_batch_size,
                                       seq_len=tcfg.seq_len)
            model.train()
            print(f"[eval] step={step} tokens={tokens_seen/1e9:.3f}B val_ppl={ppl:.3f}")
            log_jsonl({"event": "eval", "step": step, "tokens_seen": tokens_seen, "val_ppl": ppl})
            next_eval_at += tcfg.eval_every_tokens

        if tokens_seen >= next_sample_at:
            model.eval()
            samples = {}
            with torch.no_grad():
                for name, prompt in FIXED_PROMPTS.items():
                    samples[name] = generate(model, prompt, max_new_tokens=40, device=device)
            model.train()
            print(f"[sample] step={step}\n" + "\n".join(f"  {k}: {v!r}" for k, v in samples.items()))
            log_jsonl({"event": "sample", "step": step, "tokens_seen": tokens_seen, "samples": samples})
            next_sample_at += tcfg.sample_every_tokens

        if tokens_seen >= next_ckpt_at:
            ckpt_path = os.path.join(tcfg.out_dir, f"ckpt_step{step}.pt")
            save_checkpoint(ckpt_path, model, optimizer, step, tokens_seen, data_gen, tcfg, mcfg)
            next_ckpt_at += tcfg.checkpoint_every_tokens

    # final checkpoint
    final_path = os.path.join(tcfg.out_dir, f"ckpt_step{step}.pt")
    save_checkpoint(final_path, model, optimizer, step, tokens_seen, data_gen, tcfg, mcfg)
    log_f.close()
    print("[train] done.")


if __name__ == "__main__":
    main()