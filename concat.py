import json, sys
src, out = sys.argv[1], sys.argv[2]
data = json.load(open(src, encoding="utf-8"))
texts = [r["row"]["text"] for r in sorted(data["rows"], key=lambda r: r["row_idx"])]
clean = [" ".join(t.replace("\r\n", "\n").replace("\r", "\n").split("\n")) for t in texts]
corpus = " ".join(clean)
open(out, "w", encoding="utf-8", newline="").write(corpus)
print("documents:", len(texts))
print("characters:", len(corpus))
print("bytes (UTF-8):", len(corpus.encode("utf-8")))
print("line breaks left:", corpus.count("\n"))
print("first 200 chars:", corpus[:200])
