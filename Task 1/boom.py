import torch
import math

def raw_attn(q, k, v, mask):
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
    w = torch.softmax(scores, dim=-1)
    return torch.matmul(w, v), w


def safe_attn(q, k, v, mask):
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))

    dead = mask.all(dim=-1)
    if dead.any():
        for i in range(mask.shape[0]):
            if dead[i]:
                scores[:, i, :] = 0.0

    w = torch.softmax(scores, dim=-1)
    out = torch.matmul(w, v)

    if dead.any():
        out[:, dead, :] = 0.0

    return out, w


def go():
    torch.manual_seed(42)
    n, d = 8, 4
    q = torch.randn(1, n, d)
    k = torch.randn(1, n, d)
    v = torch.randn(1, n, d)

    mask = torch.zeros(n, n, dtype=torch.bool)
    causal = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    mask = mask | causal
    mask[0, :] = True
    mask[1, :] = True

    print("mask (1=blocked):")
    for i in range(n):
        row = ''.join(['#' if mask[i][j] else '.' for j in range(n)])
        print(f"  q{i}: {row}")
    print("\nrows 0 and 1 have nothing to attend to\n")

    out1, w1 = raw_attn(q, k, v, mask)
    nan_hit = torch.isnan(out1).any().item()
    bad_rows = torch.isnan(out1[0]).any(dim=-1)
    print("without fix:")
    print(f"  NaN in output? {nan_hit}")
    print(f"  bad rows: {[i for i in range(n) if bad_rows[i]]}")
    print(f"  q0 weights: {w1[0][0]}")

    out2, w2 = safe_attn(q, k, v, mask)
    print(f"\nwith fix:")
    print(f"  NaN in output? {torch.isnan(out2).any().item()}")
    print(f"  q0 weights: {w2[0][0]}")
    print(f"  q0 output: {out2[0][0]} (zeroed, no valid keys)")

    same = torch.allclose(out1[0, 2:], out2[0, 2:], atol=1e-6)
    print(f"\n  normal rows (2-7) unaffected by fix? {same}")


if __name__ == "__main__":
    go()
