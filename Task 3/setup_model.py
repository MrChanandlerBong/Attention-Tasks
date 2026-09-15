import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
# sdpa (the default attention backend) silently drops attention weights even
# when you pass output_attentions=True -- eager is required to actually get them
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float32, attn_implementation="eager"
)
model.eval()


def get_layer_kv(past_key_values, layer_idx):
    # current transformers Cache stores each layer as a CacheLayer object with
    # plain .keys / .values tensor attributes (checked directly against the
    # installed cache_utils.py source, not assumed)
    layer = past_key_values.layers[layer_idx]
    return layer.keys, layer.values


prompt = "The quick brown fox jumps over the lazy dog and then"
inputs = tok(prompt, return_tensors="pt")

with torch.no_grad():
    out = model(**inputs, use_cache=True, output_attentions=True)

kv = out.past_key_values
attn = out.attentions

print(f"num layers in cache: {len(kv)}")
print(f"num attention tensors: {len(attn)}")

k0, v0 = get_layer_kv(kv, 0)
print(f"layer0 key shape (batch, heads, seq, head_dim): {tuple(k0.shape)}")
print(f"layer0 val shape: {tuple(v0.shape)}")
print(f"layer0 attn shape (batch, heads, seq, seq): {tuple(attn[0].shape)}")

# generate one more token and feed the SAME cache back in, instead of recomputing
# from scratch -- this is the whole point of past_key_values existing
next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

with torch.no_grad():
    out2 = model(next_tok, past_key_values=kv, use_cache=True, output_attentions=True)

k0_after, _ = get_layer_kv(out2.past_key_values, 0)
print(f"layer0 key shape after one generation step: {tuple(k0_after.shape)}")
# seq dim should be +1 vs before -- confirms the cache actually grew instead of
# you having fed the whole prompt through again from scratch
