"""Pick a short sentence that appears exactly in fineweb_corpus.txt, use its first words as a prompt,
and let the trained model complete it 100 times.

Usage:  py corpus_completions.py
"""
import re
import sys
from collections import Counter
from pathlib import Path

import tiktoken

from sampling import complete, load_checkpoint

N = 100
BEGIN_WORDS = 3


def opening(sentence):
    """First two words, lowercased, used to find a common sentence opening."""
    return " ".join(re.sub(r"[^\w']", "", w).lower() for w in sentence.split()[:2])


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    enc = tiktoken.get_encoding("o200k_base")
    model, used = load_checkpoint()
    known = set(used)

    text = Path("fineweb_corpus.txt").read_text(encoding="utf-8")
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    counts = Counter(opening(s) for s in sentences)

    # short sentences, most common opening first, then shortest first
    candidates = [s for s in sentences if 6 <= len(s.split()) <= 25 and s in text]
    candidates.sort(key=lambda s: (-counts[opening(s)], len(s)))

    chosen = None
    for s in candidates:
        beginning = re.match(r"\S+(?:\s+\S+){%d}" % (BEGIN_WORDS - 1), s).group(0)
        # inside the corpus a sentence is normally preceded by a space, which belongs to its first token
        prefix = "" if text.startswith(s) else " "
        ids = enc.encode(prefix + beginning)
        if all(t in known for t in ids):
            chosen = (s, beginning, prefix, ids)
            break
        print(f"Skipped (token not in the model's vocabulary): {s[:60]!r}")
    if chosen is None:
        sys.exit("No sentence found whose beginning only uses tokens the model knows.")

    sentence, beginning, prefix, ids = chosen
    print(f"Chosen sentence: {sentence}")
    print(f"Beginning:       {beginning!r} (opening {opening(sentence)!r} occurs {counts[opening(sentence)]} times)")

    outputs = [complete(model, used, enc, ids, seed=i)[0] for i in range(N)]
    exact = sum(sentence in o for o in outputs)

    lines = [f"Chosen sentence: {sentence}", f"Beginning: {prefix}{beginning}", ""]
    lines += [f"{i + 1:3d}: {o}" for i, o in enumerate(outputs)]
    lines += ["", f"{exact} of {N} completions contain the whole original sentence."]
    Path("corpus_completions.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {N} completions to corpus_completions.txt ({exact} contain the whole original sentence)")


if __name__ == "__main__":
    main()
