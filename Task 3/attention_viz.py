import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
# sdpa (default backend) silently drops attention weights even with output_attentions=True
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float32, attn_implementation="eager"
)
model.eval()

# need enough tokens that a real sink pattern actually has room to show up,
# a ten word prompt won't cut it. repeating a paragraph is a lazy but effective
# way to get length without writing 500 words by hand
para = "The history of computing is long and complicated, full of dead ends and lucky breaks. "
long_text = para * 40

inputs = tok(long_text, return_tensors="pt", truncation=True, max_length=512)
seq_len = inputs["input_ids"].shape[1]
print(f"context length used: {seq_len} tokens")

with torch.no_grad():
    out = model(**inputs, output_attentions=True)

attn = out.attentions  # tuple, one (batch, heads, seq, seq) per layer
num_layers = len(attn)
num_heads = attn[0].shape[1]
print(f"layers: {num_layers}, heads per layer: {num_heads}")

# for every layer: average attention RECEIVED by each key position,
# averaged over heads and over every query row that attended to it
mass_per_layer = np.zeros((num_layers, seq_len))
for l in range(num_layers):
    a = attn[l][0]                      # (heads, seq, seq)
    avg_over_heads = a.mean(dim=0)      # (seq, seq)
    avg_over_queries = avg_over_heads.mean(dim=0)  # (seq,)
    mass_per_layer[l] = avg_over_queries.numpy()

plt.figure(figsize=(12, 6))
plt.imshow(mass_per_layer, aspect="auto", cmap="viridis")
plt.colorbar(label="avg attention mass received")
plt.xlabel("key token position")
plt.ylabel("layer")
plt.title("attention mass received per position, across layers")
plt.tight_layout()
plt.savefig("attn_heatmap.png", dpi=150)
plt.close()

# the actual claim to check: do the first few tokens hog a disproportionate
# share of attention regardless of what they actually say
first_k = 4
sink_mass = mass_per_layer[:, :first_k].sum(axis=1)
rest_mass = mass_per_layer[:, first_k:].sum(axis=1)

plt.figure(figsize=(10, 5))
plt.plot(sink_mass, marker="o", label=f"first {first_k} tokens (summed mass)")
plt.plot(rest_mass, marker="o", label=f"remaining {seq_len - first_k} tokens (summed mass)")
plt.xlabel("layer")
plt.ylabel("summed attention mass")
plt.legend()
plt.title("sink tokens vs rest of sequence, per layer")
plt.tight_layout()
plt.savefig("sink_vs_rest.png", dpi=150)
plt.close()

for l in range(num_layers):
    frac = sink_mass[l] / (sink_mass[l] + rest_mass[l])
    print(f"layer {l:2d}: first {first_k} toks = {sink_mass[l]:.3f} mass "
          f"({frac*100:.1f}% of total, despite being {first_k}/{seq_len} of the tokens)")
