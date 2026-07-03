# sample.py
"""Autoregressive sampling using Mamba's O(1)-per-token recurrent inference
path (MambaBlock.step), not the parallel training scan. Prompt is fed
token-by-token to build the (conv_state, ssm_state) cache, then generation
continues with the same cache -- no re-running the full prefix each step."""
import argparse

import tiktoken
import torch
import torch.nn.functional as F

from config import get_model_config
from model.model import MambaLM

# Fixed prompts used for periodic qualitative sanity checks during training.
FIXED_PROMPTS = {
    "science": "The mitochondria is the powerhouse of the cell because",
    "history": "The French Revolution began in 1789 when",
    "howto": "To bake a simple loaf of bread, first you need to",
}

_enc = None


def _get_encoder():
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("gpt2")
    return _enc


@torch.no_grad()
def generate(model, prompt: str, max_new_tokens: int = 40, device: str = "cuda",
             temperature: float = 0.8, top_k: int = 50):
    model.eval()
    enc = _get_encoder()
    prompt_ids = enc.encode_ordinary(prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)  # (1, prompt_len)

    cache = model.allocate_inference_cache(batch_size=1, device=device, dtype=torch.float32)

    # prefill: step through the prompt tokens to build up cache state
    logits = None
    for t in range(idx.shape[1]):
        logits, cache = model.step(idx[:, t : t + 1], cache)  # (1,1,vocab)

    generated = list(prompt_ids)
    next_logits = logits  # (1,1,vocab) logits after last prompt token
    for _ in range(max_new_tokens):
        probs_logits = next_logits[:, -1, :].float() / max(temperature, 1e-5)  # (1, vocab)
        if top_k is not None:
            v, _ = torch.topk(probs_logits, min(top_k, probs_logits.size(-1)))
            probs_logits[probs_logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(probs_logits, dim=-1)  # (1, vocab)
        next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)
        generated.append(int(next_id.item()))
        next_logits, cache = model.step(next_id, cache)  # (1,1,vocab)

    text = enc.decode(generated)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--model_size", type=str, default="1.4B")
    ap.add_argument("--prompt", type=str, default=None, help="if omitted, uses FIXED_PROMPTS")
    ap.add_argument("--max_new_tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mcfg = get_model_config(args.model_size)
    model = MambaLM(mcfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    prompts = {"prompt": args.prompt} if args.prompt is not None else FIXED_PROMPTS
    for name, prompt in prompts.items():
        text = generate(model, prompt, max_new_tokens=args.max_new_tokens, device=device,
                         temperature=args.temperature, top_k=args.top_k)
        print(f"=== {name} ===\n{text}\n")


if __name__ == "__main__":
    main()