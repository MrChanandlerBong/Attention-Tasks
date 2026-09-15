import torch


def _n(kv):
    return kv.layers[0].keys.shape[2]


def slide(kv, cap):
    n = _n(kv)
    if n <= cap:
        return kv
    idx = torch.arange(n - cap, n)
    for L in kv.layers:
        L.keys = L.keys[:, :, idx, :]
        L.values = L.values[:, :, idx, :]
    return kv


def sink_slide(kv, cap, sinks=1):
    n = _n(kv)
    if n <= cap:
        return kv
    rem = cap - sinks
    idx = torch.cat([torch.arange(0, sinks), torch.arange(n - rem, n)])
    for L in kv.layers:
        L.keys = L.keys[:, :, idx, :]
        L.values = L.values[:, :, idx, :]
    return kv


def h2o(kv, cap, scores):
    n = _n(kv)
    if n <= cap:
        return kv, scores
    top = torch.topk(scores, cap).indices
    idx, _ = torch.sort(top)
    for L in kv.layers:
        L.keys = L.keys[:, :, idx, :]
        L.values = L.values[:, :, idx, :]
    return kv, scores[idx]


def bump(scores, step_attn):
    s = step_attn.mean(dim=0)
    if scores is None:
        return s.clone()
    if scores.shape[0] < s.shape[0]:
        pad = torch.zeros(s.shape[0] - scores.shape[0], device=s.device)
        scores = torch.cat([scores, pad])
    return scores + s
