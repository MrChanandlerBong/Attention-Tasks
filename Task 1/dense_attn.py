import torch
import math

def vanilla_attn(q, k, v, causal=False):
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)

    if causal:
        n = scores.shape[-1]
        cmask = torch.triu(torch.ones(n, n, device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(cmask, float('-inf'))

    w = torch.softmax(scores, dim=-1)
    out = torch.matmul(w, v)
    return out, w


if __name__ == "__main__":
    torch.manual_seed(42)
    b, n, d = 2, 8, 4
    q = torch.randn(b, n, d)
    k = torch.randn(b, n, d)
    v = torch.randn(b, n, d)

    out, w = vanilla_attn(q, k, v, causal=False)
    print("no causal")
    print(out.shape, w.shape)
    print("row sums:", w.sum(dim=-1))

    out_c, w_c = vanilla_attn(q, k, v, causal=True)
    print("\ncausal")
    print("q0 weights (should only hit itself):", w_c[0][0])
    print("q_last weights (sees everything):", w_c[0][-1])
