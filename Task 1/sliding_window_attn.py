import torch
import math

def sw_mask(seq_len, window_size, causal=True, device='cpu'):
    m = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    for i in range(seq_len):
        if causal:
            lo = max(0, i - window_size + 1)
            hi = i + 1
        else:
            lo = max(0, i - window_size // 2)
            hi = min(seq_len, i + window_size // 2 + 1)
        m[i, lo:hi] = False
    return m


def sw_attn(q, k, v, window_size, causal=True):
    d = q.shape[-1]
    n = q.shape[-2]

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    mask = sw_mask(n, window_size, causal=causal, device=q.device)
    scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))

    dead = mask.all(dim=-1)
    if dead.any():
        print(f"heads up: {dead.sum().item()} queries got fully masked")
        for i in range(n):
            if dead[i]:
                scores[:, i, :] = 0.0

    w = torch.softmax(scores, dim=-1)
    out = torch.matmul(w, v)
    return out, w


if __name__ == "__main__":
    torch.manual_seed(42)
    b, n, d = 2, 16, 4
    win = 4

    q = torch.randn(b, n, d)
    k = torch.randn(b, n, d)
    v = torch.randn(b, n, d)

    out, w = sw_attn(q, k, v, window_size=win, causal=True)
    print(f"sliding window (w={win}, causal)")
    print(out.shape)

    print("\nq0:", w[0][0][:8])
    print("q5:", w[0][5][:8])
    print("q15:", w[0][15][10:])

    m = sw_mask(n, win, causal=True)
    print("\npattern:")
    for i in range(n):
        row = ''.join(['.' if not m[i][j] else '#' for j in range(n)])
        print(f"  q{i:2d}: {row}")
