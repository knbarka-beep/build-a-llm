"""Complete "The sky is" 100 times with the trained model and save the results.

Usage:  py sky_completions.py
If the model has never seen one of the tokens, it cannot read the prompt. The script then says so and stops,
instead of making something up.
"""
import sys
from pathlib import Path

import tiktoken

from sampling import complete, load_checkpoint

PROMPT = "The sky is"
N = 100


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    enc = tiktoken.get_encoding("o200k_base")
    model, used = load_checkpoint()

    ids = enc.encode(PROMPT)
    known = set(used)
    missing = [t for t in ids if t not in known]
    print(f"{PROMPT!r} is {len(ids)} tokens: {[enc.decode([t]) for t in ids]}")
    if missing:
        print(f"FAILED: the model's vocabulary does not contain {len(missing)} of them: "
              f"{[enc.decode([t]) for t in missing]}")
        print("It only knows the 1 814-ish tokens that occur in fineweb_corpus.txt, so it cannot read this prompt.")
        sys.exit(1)

    lines = []
    for i in range(N):
        full, _ = complete(model, used, enc, ids, seed=i)
        lines.append(f"{i + 1:3d}: {full}")
    Path("sky_completions.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {N} completions to sky_completions.txt")


if __name__ == "__main__":
    main()
