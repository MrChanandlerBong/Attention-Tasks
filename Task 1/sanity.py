import torch
import math
import sys

def dense_with_mask(q, k, v, mask):
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
    dead = mask.all(dim=-1)
    if dead.any():
        for i in range(mask.shape[0]):
            if dead[i]:
                scores[:, i, :] = 0.0
    w = torch.softmax(scores, dim=-1)
    return torch.matmul(w, v)


def check(name, got, ref, tol=1e-5):
    diff = (got - ref).abs().max().item()
    ok = torch.allclose(got, ref, atol=tol)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} — max diff: {diff:.2e}")
    return ok


def run():
    torch.manual_seed(0)
    results = []

    for n in [8, 32, 64, 128]:
        print(f"\nseq_len = {n}")
        b, d = 2, 16
        q = torch.randn(b, n, d)
        k = torch.randn(b, n, d)
        v = torch.randn(b, n, d)

        from sliding_window_attn import sw_attn, sw_mask
        win = max(2, n // 4)
        m1 = sw_mask(n, win, causal=True)
        out1, _ = sw_attn(q, k, v, window_size=win, causal=True)
        ref1 = dense_with_mask(q, k, v, m1)
        results.append(check(f"sliding window (w={win})", out1, ref1))

        from global_window_attn import gw_attn, make_mask
        g = 2
        m2 = make_mask(n, win, num_global_tokens=g, causal=True)
        out2, _ = gw_attn(q, k, v, window_size=win, num_global_tokens=g, causal=True)
        ref2 = dense_with_mask(q, k, v, m2)
        results.append(check(f"global window (w={win}, g={g})", out2, ref2))

    total, passed = len(results), sum(results)
    print(f"\n{'='*40}")
    print(f"  {passed}/{total} tests passed")
    print("  all good." if passed == total else "  something's broken ^")
    print(f"{'='*40}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(run())
