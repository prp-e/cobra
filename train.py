# train.py
"""Single-GPU training loop (B200/H200/A100/4090 via micro-batch + grad
accum, never via architecture changes). bf16 autocast forward/backward,
fp32 master weights + optimizer state (no GradScaler needed since we never
cast params themselves to fp16/bf16). Per-layer activation checkpointing
is ON by default -- without it, the pure-PyTorch chunked scan's
intermediates (O(Bsz*L*d_inner*N) per layer) all stay resident
simultaneously across every layer until backward starts, which OOMs even
at batch size 1 on a single GPU for a 48-layer model.

This version adds: (a) forced `spawn` multiprocessing start method +
default num_workers=0, to eliminate a classic Linux hang where DataLoader
workers are fork()'d from a process that already holds a CUDA context;
(b) a synthetic-data preflight timing test so you know, before any real
data loads, how long one forward/backward actually takes (the
scan-with-checkpointing combo is genuinely slow -- many seconds per
microbatch is expected, not a bug -- this makes that visible instead of
looking like a hang); (c) early per-microbatch heartbeat prints so the
first couple of steps aren't silent. Fully resumable after preemption.
"""
import argparse
import json
import math
import os
import random
import sys
import time
import warnings

import numpy as np
import torch

# Silence a benign FutureWarning emitted by torch.utils.checkpoint's internal
# state-preservation context manager; unrelated to correctness or speed.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\.utils\.checkpoint")

# Reduce CUDA allocator fragmentation, which matters more once activation
# checkpointing creates a churn of many small alloc/free cycles per layer.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from config import get_model_config, get_train_config, grad_accum_steps  # noqa: E402
from data.dataset import PackedTokenDataset  # noqa: E402
from model.model import MambaLM  # noqa: E402
from evaluate import evaluate_perplexity  # noqa: E402
from sample import generate, FIXED_PROMPTS  # noqa: E402

MODEL_SIZES = ["150M", "1.4B", "2.8B"]  # keep in sync with config.py's get_model_config/get_train_config


def log(msg):
    print(msg, flush=True)


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
    log(f"[checkpoint] saved {path} (step={step}, tokens_seen={tokens_seen})")


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


def build_train_val_loaders(meta_path, seq_len, micro_batch_size, data_gen, num_workers):
    train_ds = PackedTokenDataset(meta_path, "train", seq_len)
    val_ds = PackedTokenDataset(meta_path, "val", seq_len)

    train_sampler = torch.utils.data.RandomSampler(
        train_ds, replacement=True, num_samples=len(train_ds), generator=data_gen
    )
    # timeout only meaningful (and only legal) when num_workers > 0: it turns
    # a genuinely stuck worker into a loud RuntimeError instead of a silent
    # hang, which is exactly the failure mode we want visibility into.
    loader_kwargs = dict(
        batch_size=micro_batch_size, sampler=train_sampler, drop_last=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, timeout=180)
    train_loader = torch.utils.data.DataLoader(train_ds, **loader_kwargs)
    return train_ds, train_loader, val_ds


def preflight_timing_check(model, device, micro_batch_size, seq_len, vocab_size):
    """Run one forward+backward on synthetic random tokens, with explicit
    CUDA syncs, and print exact timings. This isolates 'is the model slow'
    from 'is the data pipeline slow' before we touch any real data, and
    gives you a concrete per-microbatch time to multiply by accum_steps."""
    log("[preflight] running one synthetic forward+backward to measure timing "
        "(this is expected to take a while -- pure-PyTorch scan + grad "
        "checkpointing is not fast; this just makes that visible)...")
    x = torch.randint(0, vocab_size, (micro_batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (micro_batch_size, seq_len), device=device)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
        logits, loss = model(x, targets=y)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    loss.backward()
    if device == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()

    model.zero_grad(set_to_none=True)  # discard synthetic grads, don't pollute real training
    fwd_s, bwd_s = t1 - t0, t2 - t1
    log(f"[preflight] forward={fwd_s:.2f}s backward={bwd_s:.2f}s "
        f"total_microbatch~={fwd_s + bwd_s:.2f}s")
    return fwd_s + bwd_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_size", type=str, default="1.4B", choices=MODEL_SIZES)
    ap.add_argument("--micro_batch_size", type=int, default=8, help="sequences per microbatch; tune per GPU")
    ap.add_argument("--global_batch_tokens", type=int, default=524288)
    ap.add_argument("--max_tokens", type=int, default=None, help="override TrainConfig.max_tokens (e.g. for smoke tests)")
    ap.add_argument("--data_meta_path", type=str, default="data/cache/meta.json")
    ap.add_argument("--out_dir", type=str, default="checkpoints")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--num_workers", type=int, default=0,
                     help="0 (default) avoids fork+CUDA hang risk entirely; memmap reads are "
                          "cheap so 0 workers is usually fast enough. Raise only if the "
                          "preflight/heartbeat timing shows data loading (not the model) is "
                          "the bottleneck.")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--scan_mode", type=str, default="chunked", choices=["chunked", "naive"])
    ap.add_argument("--scan_chunk_size", type=int, default=None,
                     help="override ModelConfig.scan_chunk_size (e.g. try 128/256 if the "
                          "python-loop-over-chunks overhead dominates; this is a compute "
                          "tiling granularity, not an architecture change)")
    ap.add_argument("--grad_checkpointing", type=lambda s: s.lower() != "false", default=True,
                     help="per-layer activation checkpointing (default True); disable only if "
                          "you have verified you have enough memory without it")
    ap.add_argument("--skip_preflight", action="store_true",
                     help="skip the synthetic timing smoke test (not recommended on a new setup)")
    args = ap.parse_args()

    mcfg = get_model_config(args.model_size)
    if args.scan_chunk_size is not None:
        mcfg.scan_chunk_size = args.scan_chunk_size

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
    log(f"[train] model_size={args.model_size} accum_steps={accum_steps} tokens_per_step={tokens_per_step} "
        f"total_steps={total_steps} warmup_steps={warmup_steps} "
        f"grad_checkpointing={args.grad_checkpointing} scan_chunk_size={mcfg.scan_chunk_size} "
        f"num_workers={args.num_workers}")

    model = MambaLM(
        mcfg, scan_mode=args.scan_mode, use_grad_checkpointing=args.grad_checkpointing
    ).to(device)
    log(f"[train] model params: {model.num_params()/1e9:.3f}B "
        f"(non-embedding: {model.num_params(non_embedding=True)/1e9:.3f}B)")

    param_groups = get_param_groups(model, tcfg.weight_decay)
    optimizer = torch.optim.AdamW(
        param_groups, lr=tcfg.peak_lr, betas=(tcfg.beta1, tcfg.beta2), eps=1e-8
    )

    if not args.skip_preflight and device == "cuda":
        model.train()
        est_microbatch_s = preflight_timing_check(
            model, device, tcfg.micro_batch_size, tcfg.seq_len, mcfg.vocab_size
        )
        est_step_s = est_microbatch_s * accum_steps
        log(f"[preflight] estimated time per optimizer step (accum_steps={accum_steps}): "
            f"~{est_step_s:.1f}s (~{est_step_s/60:.1f} min). If this looks too slow, try "
            f"raising --micro_batch_size, raising --scan_chunk_size, or (if memory allows) "
            f"--grad_checkpointing False.")

    log("[data] building train/val datasets from local shards (no network)...")
    data_gen = torch.Generator()
    data_gen.manual_seed(args.seed)
    t_data0 = time.perf_counter()
    train_ds, train_loader, val_ds = build_train_val_loaders(
        tcfg.data_meta_path, tcfg.seq_len, tcfg.micro_batch_size, data_gen, args.num_workers
    )
    log(f"[data] ready in {time.perf_counter() - t_data0:.1f}s "
        f"(train windows={len(train_ds):,}, val windows={len(val_ds):,})")

    step, tokens_seen = 0, 0
    if args.resume:
        ckpt_path = latest_checkpoint(tcfg.out_dir)
        if ckpt_path is not None:
            step, tokens_seen = load_checkpoint(ckpt_path, model, optimizer, data_gen, device)
            log(f"[resume] loaded {ckpt_path}: step={step}, tokens_seen={tokens_seen}")
        else:
            log("[resume] no checkpoint found, starting fresh")

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
    HEARTBEAT_STEPS = 2  # print every microbatch for the first N optimizer steps, then quiet down

    log("[train] entering main loop...")
    while tokens_seen < tcfg.max_tokens and step < total_steps:
        lr = lr_at_step(step, total_steps, warmup_steps, tcfg.peak_lr, tcfg.min_lr_ratio)
        for g in optimizer.param_groups:
            g["lr"] = lr

        heartbeat = step < HEARTBEAT_STEPS
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        step_t0 = time.perf_counter()
        for micro in range(accum_steps):
            micro_t0 = time.perf_counter()
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

            if heartbeat:
                log(f"  [heartbeat] step={step} microbatch={micro+1}/{accum_steps} "
                    f"loss={loss.item()*accum_steps:.4f} elapsed={time.perf_counter()-micro_t0:.2f}s")

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip_norm)
        optimizer.step()

        step += 1
        tokens_seen += tokens_per_step

        if heartbeat or step % tcfg.log_every_steps == 0:
            dt = time.time() - t0
            n_steps_since_log = tcfg.log_every_steps if not heartbeat else 1
            toks_per_sec = tokens_per_step * n_steps_since_log / max(dt, 1e-6)
            log(f"step={step} tokens={tokens_seen/1e9:.3f}B loss={loss_accum:.4f} "
                f"lr={lr:.2e} grad_norm={float(grad_norm):.2f} tok/s={toks_per_sec:.0f} "
                f"step_time={time.perf_counter()-step_t0:.1f}s")
            log_jsonl({"event": "train_step", "step": step, "tokens_seen": tokens_seen,
                       "loss": loss_accum, "lr": lr, "grad_norm": float(grad_norm),
                       "tokens_per_sec": toks_per_sec})
            t0 = time.time()

        if tokens_seen >= next_eval_at:
            model.eval()
            ppl = evaluate_perplexity(model, val_ds, device, num_batches=50, batch_size=tcfg.micro_batch_size,
                                       seq_len=tcfg.seq_len)
            model.train()
            log(f"[eval] step={step} tokens={tokens_seen/1e9:.3f}B val_ppl={ppl:.3f}")
            log_jsonl({"event": "eval", "step": step, "tokens_seen": tokens_seen, "val_ppl": ppl})
            next_eval_at += tcfg.eval_every_tokens

        if tokens_seen >= next_sample_at:
            model.eval()
            samples = {}
            with torch.no_grad():
                for name, prompt in FIXED_PROMPTS.items():
                    samples[name] = generate(model, prompt, max_new_tokens=40, device=device)
            model.train()
            log(f"[sample] step={step}\n" + "\n".join(f"  {k}: {v!r}" for k, v in samples.items()))
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
    log("[train] done.")


if __name__ == "__main__":
    # Must set the multiprocessing start method before any DataLoader with
    # num_workers>0 is created. Default on Linux is 'fork', which forking a
    # process that already holds a CUDA context is a known source of silent
    # hangs; 'spawn' re-imports the module fresh in each worker instead and
    # sidesteps that entirely. Harmless even when num_workers=0.
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set (e.g. re-entrant call in some environments)
    main()