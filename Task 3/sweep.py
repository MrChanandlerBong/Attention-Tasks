import torch, matplotlib.pyplot as plt
from needle import ppl_under_cap, make_haystack, needle_probe, tok, wiki

n_toks = tok(wiki, return_tensors="pt", truncation=True, max_length=1024).input_ids.shape[1]
print("doc tokens:", n_toks)

caps = [32, 64, 128, 256, 512]
results = {"slide": [], "sink": [], "h2o": []}

for cap in caps:
    print("cap =", cap)
    for pol in ["slide", "sink", "h2o"]:
        p = ppl_under_cap(wiki, cap=cap, policy=pol, sinks=1)
        results[pol].append(p)
        print(" ", pol, "ppl", round(p, 3))

baseline = ppl_under_cap(wiki, cap=None, policy="slide")
print("no-cap baseline ppl:", round(baseline, 3))

plt.figure(figsize=(9, 5))
for pol, vals in results.items():
    plt.plot(caps, vals, marker="o", label=pol)
plt.axhline(baseline, ls="--", color="gray", label="no eviction")
plt.xlabel("cache budget (tokens)")
plt.ylabel("perplexity (lower is better)")
plt.title("quality vs memory")
plt.legend()
plt.tight_layout()
plt.savefig("sweep_ppl.png", dpi=140)
plt.close()

hay = make_haystack("The hidden cursed technique is called SHADOW-77.", wiki)
q = "What is the hidden cursed technique called?"
needle_hits = {"slide": [], "sink": [], "h2o": []}
for cap in caps:
    for pol in ["slide", "sink", "h2o"]:
        a = needle_probe(hay, q, cap=cap, policy=pol, sinks=1)
        hit = "SHADOW-77" in a
        needle_hits[pol].append(int(hit))
        print(f"needle cap={cap} {pol}: hit={hit}")

plt.figure(figsize=(9, 4))
for pol, hits in needle_hits.items():
    plt.plot(caps, hits, marker="s", label=pol)
plt.xlabel("cache budget")
plt.ylabel("needle recovered (1=yes)")
plt.title("needle-in-haystack vs budget")
plt.legend()
plt.tight_layout()
plt.savefig("sweep_needle.png", dpi=140)
plt.close()
print("saved sweep_ppl.png and sweep_needle.png")
