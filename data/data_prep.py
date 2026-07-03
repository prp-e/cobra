# data/data_prep.py
"""OFFLINE, ONE-TIME script. Requires internet. Run once per machine/cluster
before any training. See offline_setup_notes.md.

Uses the `datasets` library to fully DOWNLOAD HuggingFaceFW/fineweb-edu
(config "sample-100BT") to the local HF cache (~299GB of parquet, ~140
shards) -- not streamed -- then iterates it as a regular, memory-mapped
Arrow `Dataset` object. Tokenizes with the GPT-2 BPE tokenizer (tiktoken
"gpt2"), inserts <|endoftext|> between documents, concatenates into one
long token stream, and writes fixed-size uint16 shards to disk. Every
`val_every_n_shards`-th shard (by index) is routed to a held-out validation
directory instead of train -- this split is fixed forever at packing time.
"""
import argparse
import json
import os
import time

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset_config", type=str, default="sample-100BT")
    p.add_argument("--out_dir", type=str, default="data/shards")
    p.add_argument("--meta_path", type=str, default="data/cache/meta.json")
    p.add_argument("--shard_tokens", type=int, default=100_000_000, help="tokens per shard file")
    p.add_argument("--max_tokens", type=int, default=100_000_000_000, help="stop after this many tokens")
    p.add_argument("--val_every_n_shards", type=int, default=200, help="1/N shards -> held-out val")
    p.add_argument("--num_proc", type=int, default=8, help="parallel workers for the dataset download/extract")
    p.add_argument("--log_every_docs", type=int, default=50_000)
    return p.parse_args()


def main():
    args = parse_args()

    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    eot_id = enc.eot_token  # 50256
    vocab_size_raw = enc.n_vocab  # 50257

    train_dir = os.path.join(args.out_dir, "train")
    val_dir = os.path.join(args.out_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.meta_path), exist_ok=True)

    print(f"Downloading {args.dataset_name} ({args.dataset_config}) via `datasets` "
          f"(non-streaming; this pulls the full ~299GB config to the local HF cache "
          f"the first time, num_proc={args.num_proc}) ...")
    # No `streaming=True`: this actually downloads+caches the parquet shards
    # locally (resumable by `datasets` itself across re-runs) and returns a
    # regular, memory-mapped Arrow-backed Dataset -- fast random/sequential
    # local reads afterwards, no network needed once this call returns.
    ds = load_dataset(
        args.dataset_name, name=args.dataset_config, split="train", num_proc=args.num_proc
    )
    print(f"Downloaded/loaded dataset with {len(ds)} rows. Beginning tokenization...")

    buffer = []  # python list of python ints; flushed to np arrays per shard
    shard_idx = 0
    total_tokens = 0
    n_docs = 0
    train_shards, val_shards = [], []
    t0 = time.time()

    def flush_shard(tok_list):
        nonlocal shard_idx, total_tokens
        arr = np.asarray(tok_list, dtype=np.uint16)
        is_val = (shard_idx % args.val_every_n_shards == 0)
        outdir = val_dir if is_val else train_dir
        fname = f"shard_{shard_idx:05d}.bin"
        path = os.path.join(outdir, fname)
        arr.tofile(path)
        rel = os.path.join("val" if is_val else "train", fname)
        (val_shards if is_val else train_shards).append({"path": rel, "n_tokens": int(arr.shape[0])})
        total_tokens += arr.shape[0]
        shard_idx += 1

    stop = False
    for doc in ds:
        if stop:
            break
        text = doc.get("text", "")
        if not text:
            continue
        ids = enc.encode_ordinary(text)
        ids.append(eot_id)  # separator between documents
        buffer.extend(ids)
        n_docs += 1

        while len(buffer) >= args.shard_tokens:
            flush_shard(buffer[: args.shard_tokens])
            buffer = buffer[args.shard_tokens :]
            if total_tokens >= args.max_tokens:
                stop = True
                break

        if n_docs % args.log_every_docs == 0:
            dt = time.time() - t0
            print(f"docs={n_docs} tokens={total_tokens} shards={shard_idx} elapsed={dt:.0f}s")

    # flush any remainder (< shard_tokens) as a final (smaller) train shard
    if len(buffer) > 0 and total_tokens < args.max_tokens:
        flush_shard(buffer)

    meta = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "tokenizer": "tiktoken:gpt2",
        "eot_id": eot_id,
        "vocab_size_raw": vocab_size_raw,
        "dtype": "uint16",
        "shard_tokens_target": args.shard_tokens,
        "val_every_n_shards": args.val_every_n_shards,
        "total_tokens": total_tokens,
        "n_docs": n_docs,
        "train_shards": train_shards,
        "val_shards": val_shards,
    }
    with open(args.meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Done. total_tokens={total_tokens} train_shards={len(train_shards)} "
          f"val_shards={len(val_shards)}. Wrote {args.meta_path}")


if __name__ == "__main__":
    main()