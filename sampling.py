"""Helpers shared by sky_completions.py and corpus_completions.py."""
import torch

from model import Config, TinyGPT


def load_checkpoint(path="results/model.pt"):
    """Load the model saved by train.py. Returns the model and the list that maps
    compact token IDs back to the original o200k_base IDs."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = TinyGPT(Config(**ck["config"]))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck["used_token_ids"]


def complete(model, used, enc, prompt_ids, seed, new_tokens=30, temperature=0.8, top_k=40):
    """Continue the prompt (original token IDs). Returns (full text, continuation only)."""
    to_small = {t: i for i, t in enumerate(used)}
    g = torch.Generator().manual_seed(seed)
    idx = torch.tensor([[to_small[t] for t in prompt_ids]])
    new_ids = [used[c] for c in model.stream(idx, new_tokens, temperature, top_k, g)]
    full = enc.decode_bytes(list(prompt_ids) + new_ids).decode("utf-8", errors="replace")
    cont = enc.decode_bytes(new_ids).decode("utf-8", errors="replace")
    return full, cont
