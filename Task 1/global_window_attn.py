import torch
import math

def make_mask(seq_len, window_size, num_global_tokens=1, causal=True, device='cpu'):
    m = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    for i in range(seq_len):
        if causal:
            lo = max(0, i - window_size + 1)
            hi = i + 1
        else:
            lo = max(0, i - window_size // 2)
            hi = min(seq_len, i + window_size // 2 + 1)
        m[i, lo:hi] = False
    for g in range(min(num_global_tokens, seq_len)):
        if causal:
            m[g, :g+1] = False
        else:
            m[g, :] = False
        m[:, g] = False
    if causal:
        cmask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
        m = m | cmask
    return m


def gw_attn(q, k, v, window_size, num_global_tokens=1, causal=True):
    dim = q.shape[-1]
    n = q.shape[-2]
    raw = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dim)
    msk = make_mask(n, window_size, num_global_tokens=num_global_tokens, causal=causal, device=q.device)
    raw = raw.masked_fill(msk.unsqueeze(0), float('-inf'))
    dead_rows = msk.all(dim=-1)
    if dead_rows.any():
        print(f"heads up: {dead_rows.sum().item()} queries got fully masked")
        for i in range(n):
            if dead_rows[i]:
                raw[:, i, :] = 0.0
    probs = torch.softmax(raw, dim=-1)
    out = torch.matmul(probs, v)
    return out, probs


if __name__ == "__main__":
    torch.manual_seed(42)
    b, sl, dim = 2, 16, 4
    w = 4
    n_glob = 2
    q = torch.randn(b, sl, dim)
    k = torch.randn(b, sl, dim)
    v = torch.randn(b, sl, dim)
    out, probs = gw_attn(q, k, v, window_size=w, num_global_tokens=n_glob, causal=True)
    print(f"window + global (w={w}, globals={n_glob}, causal)")
    print(out.shape)
    mask = make_mask(sl, w, num_global_tokens=n_glob, causal=True)
    print("\nsparsity pattern (. = attend, # = masked)\n")
    for i in range(sl):
        row = ''.join(['.' if not mask[i][j] else '#' for j in range(sl)])
        tag = " <-- global" if i < n_glob else ""
        print(f"  q{i:2d}: {row}{tag}")
    from sliding_window_attn import sw_mask
    sw_m = sw_mask(sl, w, causal=True)
    sw_seen = (~sw_m[15]).sum().item()
    gw_seen = (~mask[15]).sum().item()
    print(f"\ntoken 15 visible positions:")
    print(f"  pure sliding window: {sw_seen}")
    print(f"  with {n_glob} global tokens: {gw_seen}")
    tot = sl * sl
    masked_ct = mask.sum().item()
    print(f"\nsparsity: {masked_ct}/{tot} = {masked_ct/tot*100:.1f}% masked")
