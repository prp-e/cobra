# evaluate.py
"""Held-out perplexity evaluation. Reads only local val shards (data/dataset.py);
no network. Importable by train.py, and runnable standalone against a checkpoint."""
import argparse
import math

import torch

from config import get_model_config
from data.dataset import PackedTokenDataset
from model.model import MambaLM


@torch.no_grad()
def evaluate_perplexity(model, val_dataset: PackedTokenDataset, device, num_batches=50,
                         batch_size=8, seq_len=2048):
    model.eval()
    gen = torch.Generator().manual_seed(12345)  # fixed seed -> deterministic eval subset each call
    sampler = torch.utils.data.RandomSampler(
        val_dataset, replacement=True, num_samples=num_batches * batch_size, generator=gen
    )
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler=sampler)

    total_loss, total_tokens = 0.0, 0
    for x, y in loader:
        x = x.to(device)  # (batch_size, seq_len)
        y = y.to(device)  # (batch_size, seq_len)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            logits, loss = model(x, targets=y)  # loss: scalar mean cross-entropy over (batch*seq_len)
        n_tok = x.numel()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok

    mean_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(mean_loss, 20.0))  # clamp exponent for safety against blowup early in training
    return ppl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--data_meta_path", type=str, default="data/cache/meta.json")
    ap.add_argument("--model_size", type=str, default="1.4B")
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_batches", type=int, default=100)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mcfg = get_model_config(args.model_size)
    model = MambaLM(mcfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    val_ds = PackedTokenDataset(args.data_meta_path, "val", args.seq_len)
    ppl = evaluate_perplexity(model, val_ds, device, num_batches=args.num_batches,
                               batch_size=args.batch_size, seq_len=args.seq_len)
    print(f"val perplexity: {ppl:.4f}")


if __name__ == "__main__":
    main()