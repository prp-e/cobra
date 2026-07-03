# data/dataset.py
"""Training-time data loading. Reads only local memmap shards produced by
data/data_prep.py -- no network, no re-tokenization.
"""
import bisect
import json
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedTokenDataset(Dataset):
    """Memmap-backed dataset over a flat, concatenated uint16 token stream
    split across multiple shard files. __getitem__ returns a random-access
    contiguous window of length seq_len+1 (x = window[:-1], y = window[1:]).

    Indexing is by a global "valid start position" index over the whole
    virtual concatenation of shards; we binary-search a small (num_shards
    length) cumulative-length table to map a global index to (shard, local
    offset), so we never build an O(total_tokens) index array.
    """

    def __init__(self, meta_path: str, split: str, seq_len: int):
        assert split in ("train", "val")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        shard_root = os.path.dirname(os.path.dirname(meta_path))  # data/cache/.. -> data/
        shard_root = os.path.join(shard_root, "shards") if os.path.basename(shard_root) != "shards" else shard_root
        shard_list = meta["train_shards"] if split == "train" else meta["val_shards"]
        if len(shard_list) == 0:
            raise RuntimeError(f"No shards found for split={split!r} in {meta_path}")

        self.seq_len = seq_len
        self.mmaps: List[np.memmap] = []
        self.lengths: List[int] = []
        data_dir = os.path.dirname(os.path.dirname(meta_path))  # e.g. data/cache -> data
        base_dir = os.path.join(os.path.dirname(meta_path), "..") if False else None
        # Resolve shard paths relative to the "data/" directory that contains
        # both "cache/" (meta.json) and "shards/" (the .bin files).
        data_root = os.path.abspath(os.path.join(os.path.dirname(meta_path), ".."))
        for entry in shard_list:
            path = os.path.join(data_root, "shards", os.path.basename(os.path.dirname(entry["path"])),
                                 os.path.basename(entry["path"]))
            n = entry["n_tokens"]
            mm = np.memmap(path, dtype=np.uint16, mode="r", shape=(n,))
            self.mmaps.append(mm)
            self.lengths.append(n)

        # number of valid start offsets per shard (need seq_len+1 tokens)
        self.valid_per_shard = [max(0, n - (seq_len + 1)) for n in self.lengths]
        self.cum_valid = np.cumsum(self.valid_per_shard).tolist()  # small array, len = num_shards
        self._total = self.cum_valid[-1] if self.cum_valid else 0
        if self._total <= 0:
            raise RuntimeError(f"Not enough tokens per shard for seq_len={seq_len} in split={split!r}")

    def __len__(self) -> int:
        return self._total

    def _locate(self, global_idx: int) -> Tuple[int, int]:
        shard_idx = bisect.bisect_right(self.cum_valid, global_idx)
        prev_cum = self.cum_valid[shard_idx - 1] if shard_idx > 0 else 0
        local_offset = global_idx - prev_cum
        return shard_idx, local_offset

    def __getitem__(self, global_idx: int):
        shard_idx, local_offset = self._locate(int(global_idx))
        mm = self.mmaps[shard_idx]
        window = mm[local_offset : local_offset + self.seq_len + 1]
        window = np.asarray(window, dtype=np.int64)  # (seq_len+1,) upcast for embedding lookup
        x = torch.from_numpy(window[:-1].copy())  # (seq_len,)
        y = torch.from_numpy(window[1:].copy())   # (seq_len,)
        return x, y


def build_loader(meta_path: str, split: str, seq_len: int, batch_size: int,
                  generator: torch.Generator, num_workers: int = 2, pin_memory: bool = True):
    """RandomSampler(replacement=True) draws i.i.d. random indices lazily via
    `generator` (chunks of 32 internally) -- safe even when len(dataset) is
    ~100B, unlike a no-replacement permutation sampler. `generator`'s state
    is owned by train.py and saved/restored in checkpoints for exact resume.
    """
    ds = PackedTokenDataset(meta_path, split, seq_len)
    sampler = torch.utils.data.RandomSampler(
        ds, replacement=True, num_samples=len(ds), generator=generator
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
        pin_memory=pin_memory, drop_last=True, persistent_workers=(num_workers > 0),
    )
    return ds, loader