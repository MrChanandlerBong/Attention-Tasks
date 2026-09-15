import torch
from yeet import slide, sink_slide, h2o, bump, _n


@torch.no_grad()
def run(mdl, tok, prompt, cap, policy, max_new=100, sinks=1):
    ids = tok(prompt, return_tensors="pt").input_ids
    out = mdl(ids, use_cache=True, output_attentions=(policy == "h2o"))
    kv = out.past_key_values
    scores = None
    if policy == "h2o":
        # seed scores from prompt-time attn averaged across layers/heads
        avg = torch.stack([a[0].mean(0).mean(0) for a in out.attentions]).mean(0)
        scores = avg.clone()

    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    gen = [nxt.item()]

    for _ in range(max_new - 1):
        need_attn = (policy == "h2o")
        # position hack: after eviction, cache_position must match cache size
        # not the real token index, otherwise RoPE for the new query gets applied
        # at position "real_index" while cached keys sit at their original spots
        pos = torch.tensor([[_n(kv)]], dtype=torch.long)
        out = mdl(nxt, past_key_values=kv, use_cache=True,
                  output_attentions=need_attn, cache_position=pos[0])
        kv = out.past_key_values

        if policy == "h2o":
            step_attn = torch.stack([a[0, :, -1, :] for a in out.attentions]).mean(0)
            scores = bump(scores, step_attn)

        n = _n(kv)
        if n > cap:
            if policy == "slide":
                kv = slide(kv, cap)
            elif policy == "sink":
                kv = sink_slide(kv, cap, sinks=sinks)
            elif policy == "h2o":
                kv, scores = h2o(kv, cap, scores)

        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        gen.append(nxt.item())

    return gen


if False:
    pass
