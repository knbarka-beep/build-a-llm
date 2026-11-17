"""Step 3: train a tiny randomly initialised transformer on fineweb_corpus.txt for about 60 seconds on CPU.

Usage:  py train.py
Everything printed and saved comes from your own local run.
"""
import argparse
import codecs
import math
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window needed
import matplotlib.pyplot as plt
import tiktoken
import torch

from model import Config, TinyGPT

SEED = 1337     # same seed for model initialisation, batches and every generated sample
BATCH = 32      # sequences per update
BLOCK = 64      # tokens per sequence
LR = 2e-3


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles choke on some characters

    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="fineweb_corpus.txt")
    p.add_argument("--tokens", default="fineweb_tokens.txt")
    p.add_argument("--seconds", type=float, default=60.0, help="training time budget")
    p.add_argument("--prompt", default=" Don't care about", help="sentence beginning used for every sample")
    p.add_argument("--new-tokens", type=int, default=30)
    p.add_argument("--out", default="results")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    log_file = open(out / "train_log.txt", "w", encoding="utf-8")

    def log(line):
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    # 1. Tokenizer and token check ------------------------------------------------------
    enc = tiktoken.get_encoding("o200k_base")  # GPT-4o's tokenizer (downloaded on first run)
    text = Path(a.corpus).read_text(encoding="utf-8")
    ids = enc.encode(text, disallowed_special=())

    raw = Path(a.tokens).read_text(encoding="utf-8-sig", errors="replace")
    file_ids = [int(x) for x in re.findall(r"\d+", raw)]
    if file_ids == ids:
        log(f"Token check OK: all {len(ids)} IDs in {a.tokens} match the corpus.")
    else:
        log(f"Token check FAILED: {a.tokens} has {len(file_ids)} IDs, the corpus gives {len(ids)}. "
            f"Using IDs regenerated from the corpus.")
        (out / "fineweb_tokens_regenerated.txt").write_text(", ".join(map(str, ids)), encoding="utf-8")

    # 2. Compact vocabulary: only the token IDs that actually occur ---------------------
    used = sorted(set(ids))                         # compact ID -> original o200k_base ID
    to_small = {t: i for i, t in enumerate(used)}   # original ID -> compact ID
    data = torch.tensor([to_small[t] for t in ids], dtype=torch.long)
    log(f"Corpus: {len(ids)} tokens, {len(used)} distinct (of {enc.n_vocab} in the full tokenizer).")

    prompt_ids = enc.encode(a.prompt, disallowed_special=())
    missing = [t for t in prompt_ids if t not in to_small]
    if missing:
        sys.exit(f"The prompt {a.prompt!r} uses tokens that never occur in the corpus: {missing}. "
                 f"Pick a prompt that appears in fineweb_corpus.txt.")

    # 3. Model ---------------------------------------------------------------------------
    torch.manual_seed(SEED)
    cfg = Config(vocab_size=len(used), block_size=BLOCK)
    model = TinyGPT(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model: {n_params:,} parameters, random start. Random guessing would give loss "
        f"ln({len(used)}) = {math.log(len(used)):.2f}.")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    def get_batch():
        ix = torch.randint(0, len(data) - BLOCK - 1, (BATCH,))
        x = torch.stack([data[i:i + BLOCK] for i in ix])
        y = torch.stack([data[i + 1:i + BLOCK + 1] for i in ix])
        return x, y

    samples = []

    def sample(label):
        g = torch.Generator().manual_seed(SEED)  # same seed every time, so differences come from the model
        idx = torch.tensor([[to_small[t] for t in prompt_ids]])
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")  # tokens can split a character
        pieces = [dec.decode(enc.decode_bytes(prompt_ids))]
        print(f"      [{label}] {pieces[0]}", end="", flush=True)
        for c in model.stream(idx, a.new_tokens, 0.8, 40, g):
            piece = dec.decode(enc.decode_single_token_bytes(used[c]))
            pieces.append(piece)
            print(piece, end="", flush=True)  # streamed as it appears
        pieces.append(dec.decode(b"", final=True))
        print(flush=True)
        model.train()
        samples.append(f"=== {label} ===\n{''.join(pieces)}\n")

    # 4. Training ------------------------------------------------------------------------
    log("")
    log(" step |   loss | elapsed | tokens/s")
    sample("step 0, before training")
    model.train()

    losses = []
    train_time = 0.0       # only counts the update steps, not the time spent generating samples
    win_t, win_n = 0.0, 0  # for tokens/second over the last few steps
    step = 0
    while train_time < a.seconds:
        t0 = time.perf_counter()
        x, y = get_batch()
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        dt = time.perf_counter() - t0

        step += 1
        train_time += dt
        win_t += dt
        win_n += BATCH * BLOCK
        losses.append(loss.item())

        if step == 1 or step % 5 == 0:
            log(f"{step:5d} | {losses[-1]:6.4f} | {train_time:6.1f}s | {win_n / win_t:8.0f}")
            win_t, win_n = 0.0, 0
        if step % 20 == 0:
            sample(f"step {step}")

    sample(f"step {step}, final")

    # 5. Save everything -----------------------------------------------------------------
    log(f"\nDone: {step} updates in {train_time:.1f}s of training time, final loss {losses[-1]:.4f}.")

    (out / "samples.txt").write_text("\n".join(samples), encoding="utf-8")

    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, len(losses) + 1), losses, linewidth=1)
    plt.axhline(math.log(len(used)), linestyle="--", color="gray",
                label=f"random guessing, ln({len(used)}) = {math.log(len(used)):.2f}")
    plt.xlabel("update step")
    plt.ylabel("training loss")
    plt.title("Tiny transformer on 10 FineWeb documents")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "loss_plot.png", dpi=150)

    torch.save({
        "state_dict": model.state_dict(),
        "config": asdict(cfg),
        "used_token_ids": used,   # compact ID -> original o200k_base ID
        "prompt": a.prompt,
        "seed": SEED,
        "steps": step,
        "final_loss": losses[-1],
    }, out / "model.pt")

    log(f"Saved to {out.resolve()}: train_log.txt, samples.txt, loss_plot.png, model.pt")
    log_file.close()


if __name__ == "__main__":
    main()
