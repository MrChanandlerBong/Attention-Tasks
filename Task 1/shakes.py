import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import urllib.request

from dense_attn import vanilla_attn
from sliding_window_attn import sw_attn
from global_window_attn import gw_attn

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
PATH = "tinyshakespeare.txt"

BLOCK = 128
BATCH = 32
N_EMBED = 64
N_HEAD = 4
N_LAYER = 2
LR = 3e-4
ITERS = 1500
EVAL_EVERY = 300
EVAL_ITERS = 50
WIN = 32
GLOB = 4


def fetch():
    if not os.path.exists(PATH):
        print("grabbing tinyshakespeare...")
        urllib.request.urlretrieve(URL, PATH)
    return open(PATH, 'r', encoding='utf-8').read()


class Tok:
    def __init__(self, text):
        chars = sorted(set(text))
        self.n = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}

    def enc(self, s):
        return [self.stoi[c] for c in s]


def batch(data, device):
    ix = torch.randint(len(data) - BLOCK - 1, (BATCH,))
    x = torch.stack([data[i:i+BLOCK] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK+1] for i in ix])
    return x.to(device), y.to(device)


class Head(nn.Module):
    def __init__(self, n_embed, head_size, mode):
        super().__init__()
        self.k = nn.Linear(n_embed, head_size, bias=False)
        self.q = nn.Linear(n_embed, head_size, bias=False)
        self.v = nn.Linear(n_embed, head_size, bias=False)
        self.mode = mode

    def forward(self, x):
        k, q, v = self.k(x), self.q(x), self.v(x)
        if self.mode == 'dense':
            out, _ = vanilla_attn(q, k, v, causal=True)
        elif self.mode == 'sliding':
            out, _ = sw_attn(q, k, v, window_size=WIN, causal=True)
        else:
            out, _ = gw_attn(q, k, v, window_size=WIN, num_global_tokens=GLOB, causal=True)
        return out


class MultiHead(nn.Module):
    def __init__(self, n_embed, n_head, mode):
        super().__init__()
        hs = n_embed // n_head
        self.heads = nn.ModuleList([Head(n_embed, hs, mode) for _ in range(n_head)])
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))


class FF(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embed, 4*n_embed), nn.ReLU(), nn.Linear(4*n_embed, n_embed))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head, mode):
        super().__init__()
        self.sa = MultiHead(n_embed, n_head, mode)
        self.ff = FF(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab, n_embed, block, n_head, n_layer, mode):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, n_embed)
        self.pos_emb = nn.Embedding(block, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head, mode) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(t, device=idx.device))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            b, t, c = logits.shape
            loss = F.cross_entropy(logits.view(b*t, c), targets.view(b*t))
        return logits, loss


@torch.no_grad()
def eval_loss(model, tr, va, device):
    model.eval()
    out = {}
    for name, data in [('train', tr), ('val', va)]:
        losses = torch.zeros(EVAL_ITERS)
        for i in range(EVAL_ITERS):
            x, y = batch(data, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def train_one(mode, tr, va, vocab, device):
    print(f"\n{'='*50}\n  mode = {mode}\n{'='*50}")
    model = GPT(vocab, N_EMBED, BLOCK, N_HEAD, N_LAYER, mode).to(device)
    print(f"  params: {sum(p.numel() for p in model.parameters())/1e3:.1f}k")
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    for it in range(ITERS):
        if it % EVAL_EVERY == 0 or it == ITERS - 1:
            l = eval_loss(model, tr, va, device)
            print(f"  iter {it:5d} | train {l['train']:.4f} | val {l['val']:.4f}")
        x, y = batch(tr, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return eval_loss(model, tr, va, device)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    text = fetch()
    tok = Tok(text)
    print(f"vocab: {tok.n}, chars: {len(text)}")

    data = torch.tensor(tok.enc(text), dtype=torch.long)
    split = int(0.9 * len(data))
    tr, va = data[:split], data[split:]

    results = {}
    for mode in ['dense', 'sliding', 'global']:
        results[mode] = train_one(mode, tr, va, tok.n, device)

    print(f"\n{'='*50}\n  FINAL RESULTS\n{'='*50}")
    for mode, r in results.items():
        print(f"  {mode:<10} | train {r['train']:.4f} | val {r['val']:.4f}")


if __name__ == "__main__":
    main()
