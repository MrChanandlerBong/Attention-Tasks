import torch, math, urllib.request, re
from transformers import AutoModelForCausalLM, AutoTokenizer
from yeet import slide, sink_slide, h2o, bump, _n

NAME = "Qwen/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(NAME)
mdl = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, attn_implementation="eager")
mdl.eval()


def grab_wiki(title="Jujutsu_Kaisen"):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    # summary is too short — hit the plain extract endpoint instead
    url = f"https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&explaintext=1&titles={title}"
    req = urllib.request.Request(url, headers={"User-Agent": "postman-task3-eval/0.1 (student project)"})
    with urllib.request.urlopen(req) as r:
        import json
        data = json.loads(r.read())
    pages = data["query"]["pages"]
    txt = next(iter(pages.values()))["extract"]
    txt = re.sub(r"\n+", "\n", txt).strip()
    return txt


try:
    wiki = grab_wiki()
    print("pulled wiki article,", len(wiki), "chars")
except Exception as e:
    print("wiki fetch failed:", e, "— falling back to filler")
    filler = "The history of computing spans many decades and includes many surprising turns. "
    wiki = filler * 40


@torch.no_grad()
def ppl_under_cap(text, cap, policy, sinks=1):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=1024).input_ids
    if ids.shape[1] < 2:
        return float("nan")

    total_nll = 0.0
    tokens = 0
    kv = None
    scores = None

    for t in range(ids.shape[1] - 1):
        cur = ids[:, t:t+1]
        need_attn = (policy == "h2o")

        if kv is None:
            out = mdl(cur, use_cache=True, output_attentions=need_attn)
        else:
            pos = torch.tensor([_n(kv)], dtype=torch.long)
            out = mdl(cur, past_key_values=kv, use_cache=True,
                      output_attentions=need_attn, cache_position=pos)
        kv = out.past_key_values

        if policy == "h2o":
            step = torch.stack([a[0, :, -1, :] for a in out.attentions]).mean(0)
            scores = bump(scores, step)

        n = _n(kv)
        if cap is not None and n > cap:
            if policy == "slide":
                kv = slide(kv, cap)
            elif policy == "sink":
                kv = sink_slide(kv, cap, sinks=sinks)
            elif policy == "h2o":
                kv, scores = h2o(kv, cap, scores)

        logp = torch.log_softmax(out.logits[0, -1], dim=-1)
        tgt = ids[0, t+1].item()
        total_nll += -logp[tgt].item()
        tokens += 1

    return math.exp(total_nll / tokens)


def make_haystack(needle, article, max_words=200):
    words = article.split()[:max_words]
    mid = len(words) // 2
    return " ".join(words[:mid]) + " " + needle + " " + " ".join(words[mid:])


@torch.no_grad()
def needle_probe(haystack, question, cap, policy, sinks=1, n_answer=30):
    prompt = haystack + "\n\nQuestion: " + question + "\nAnswer:"
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids

    out = mdl(ids, use_cache=True, output_attentions=(policy == "h2o"))
    kv = out.past_key_values
    scores = None
    if policy == "h2o":
        avg = torch.stack([a[0].mean(0).mean(0) for a in out.attentions]).mean(0)
        scores = avg.clone()

    if cap is not None and _n(kv) > cap:
        if policy == "slide":
            kv = slide(kv, cap)
        elif policy == "sink":
            kv = sink_slide(kv, cap, sinks=sinks)
        elif policy == "h2o":
            kv, scores = h2o(kv, cap, scores)

    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    gen = [nxt.item()]

    for _ in range(n_answer - 1):
        need_attn = (policy == "h2o")
        pos = torch.tensor([_n(kv)], dtype=torch.long)
        out = mdl(nxt, past_key_values=kv, use_cache=True,
                  output_attentions=need_attn, cache_position=pos)
        kv = out.past_key_values
        if policy == "h2o":
            step = torch.stack([a[0, :, -1, :] for a in out.attentions]).mean(0)
            scores = bump(scores, step)
        n = _n(kv)
        if cap is not None and n > cap:
            if policy == "slide":
                kv = slide(kv, cap)
            elif policy == "sink":
                kv = sink_slide(kv, cap, sinks=sinks)
            elif policy == "h2o":
                kv, scores = h2o(kv, cap, scores)
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        gen.append(nxt.item())

    return tok.decode(gen, skip_special_tokens=True)


needle_fact = "The hidden cursed technique is called SHADOW-77."
question = "What is the hidden cursed technique called?"

hay = make_haystack(needle_fact, wiki)
print("haystack length in tokens:", tok(hay, return_tensors="pt").input_ids.shape[1])

print("\nperplexity @ cap=64:")
for pol in ["slide", "sink", "h2o"]:
    p = ppl_under_cap(wiki, cap=64, policy=pol, sinks=1)
    print(f"  {pol}: {p:.3f}")
print(f"  no eviction (baseline): {ppl_under_cap(wiki, cap=None, policy='slide'):.3f}")

print("\nneedle test @ cap=64:")
for pol in ["slide", "sink", "h2o"]:
    ans = needle_probe(hay, question, cap=64, policy=pol, sinks=1)
    hit = "SHADOW-77" in ans
    print(f"  {pol}: hit={hit} | ans={ans[:80]!r}")
ans_full = needle_probe(hay, question, cap=None, policy="slide")
print(f"  no eviction: hit={'SHADOW-77' in ans_full} | ans={ans_full[:80]!r}")
