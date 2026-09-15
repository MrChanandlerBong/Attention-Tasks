import torch
import time
import platform
import sys

from dense_attn import vanilla_attn
from sliding_window_attn import sw_attn
from global_window_attn import gw_attn


def hw_info():
    print("=" * 50)
    print(f"python: {sys.version.split()[0]}  |  torch: {torch.__version__}")
    print(f"os: {platform.system()} {platform.release()}  |  cpu: {platform.processor()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"vram: {mem:.1f} GB")
    else:
        print("gpu: none, running on CPU")
    print("=" * 50)


def bench(fn, q, k, v, kwargs, warmup=3, runs=10, device='cpu'):
    for _ in range(warmup):
        fn(q, k, v, **kwargs)
    if device == 'cuda':
        torch.cuda.synchronize()

    times, mems = [], []
    for _ in range(runs):
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(q, k, v, **kwargs)
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        mems.append(torch.cuda.max_memory_allocated() / (1024**2) if device == 'cuda' else 0)

    return sum(times) / len(times), sum(mems) / len(mems)


def run():
    hw_info()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seqs = [512, 1024, 2048, 4096, 8192]
    d, b, win, glob = 64, 1, 128, 4

    res = {'dense': {'t': [], 'm': []}, 'sliding': {'t': [], 'm': []}, 'global': {'t': [], 'm': []}}

    for n in seqs:
        print(f"\nseq_len = {n}")
        torch.manual_seed(42)
        q = torch.randn(b, n, d, device=device)
        k = torch.randn(b, n, d, device=device)
        v = torch.randn(b, n, d, device=device)

        t, m = bench(vanilla_attn, q, k, v, {'causal': True}, device=device)
        res['dense']['t'].append(t); res['dense']['m'].append(m)
        print(f"  dense:   {t*1000:8.2f} ms | {m:8.1f} MB")

        t, m = bench(sw_attn, q, k, v, {'window_size': win, 'causal': True}, device=device)
        res['sliding']['t'].append(t); res['sliding']['m'].append(m)
        print(f"  sliding: {t*1000:8.2f} ms | {m:8.1f} MB")

        t, m = bench(gw_attn, q, k, v, {'window_size': win, 'num_global_tokens': glob, 'causal': True}, device=device)
        res['global']['t'].append(t); res['global']['m'].append(m)
        print(f"  global:  {t*1000:8.2f} ms | {m:8.1f} MB")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
        for name in res:
            a1.plot(seqs, [x*1000 for x in res[name]['t']], marker='o', label=name)
        a1.set_xlabel('seq len'); a1.set_ylabel('ms'); a1.set_title('wall clock')
        a1.legend(); a1.set_yscale('log'); a1.grid(alpha=0.3)

        if device == 'cuda':
            for name in res:
                a2.plot(seqs, res[name]['m'], marker='o', label=name)
            a2.set_xlabel('seq len'); a2.set_ylabel('MB'); a2.set_title('peak GPU memory')
            a2.legend(); a2.set_yscale('log'); a2.grid(alpha=0.3)
        else:
            a2.text(0.5, 0.5, 'needs CUDA', ha='center', va='center', transform=a2.transAxes)

        plt.tight_layout()
        plt.savefig('benchmark_plot.png', dpi=150)
        print("\nsaved benchmark_plot.png")
    except ImportError:
        print("\nno matplotlib, skipping plot")

    print(f"\n{'seq_len':>8} | {'dense':>10} | {'sliding':>10} | {'global':>10}  (ms)")
    for i, n in enumerate(seqs):
        print(f"{n:>8} | {res['dense']['t'][i]*1000:>10.2f} | {res['sliding']['t'][i]*1000:>10.2f} | {res['global']['t'][i]*1000:>10.2f}")


if __name__ == "__main__":
    run()
