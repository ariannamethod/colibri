"""Build a TINY Mistral Small 4 (mistral4) random-weight ORACLE for C-engine parity.

Real architecture (MLA q/kv-LoRA + softmax router, NO e_score_bias, NO DSA, YaRN rope),
tiny dimensions. Saves weights+config to mistral4_tiny/ and a greedy + teacher-forcing
reference to ref_mistral4.json.

The sequence runs PAST original_max_position_embeddings (=16) so the YaRN frequency
correction is exercised -- the C engine must reproduce it or tf_pred diverges on the
late positions. Router is plain softmax + norm_topk (shape of olmoe.c:319-353), so the
C parity target is the mistral4 arch-branch, not the GLM sigmoid/noaux path.

Python here is sanctioned OFFLINE tooling (Oleg, 2026-07-15). The inference engine
stays pure C -- this script only produces the golden reference.
"""
import json
import torch
from transformers import Mistral4Config, Mistral4ForCausalLM

torch.manual_seed(1234)

cfg = Mistral4Config(
    vocab_size=256,
    hidden_size=128,
    intermediate_size=256,         # dense MLP (unused: first_k_dense_replace=0)
    moe_intermediate_size=32,      # per-expert intermediate
    num_hidden_layers=4,           # all sparse -- no dense prefix (unlike GLM's 3)
    first_k_dense_replace=0,
    num_attention_heads=4,
    num_key_value_heads=4,
    n_routed_experts=8,
    num_experts_per_tok=4,
    n_shared_experts=1,
    q_lora_rank=64,
    kv_lora_rank=32,
    qk_nope_head_dim=64,
    qk_rope_head_dim=64,           # even -> interleave ok
    v_head_dim=128,
    n_group=1,
    topk_group=1,
    norm_topk_prob=True,
    routed_scaling_factor=1.0,
    # YaRN: original_max=16, max_position=32 -> implicit factor 2.0 == explicit (no mismatch warning)
    rope_parameters={
        "rope_type": "yarn",
        "factor": 2.0,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "original_max_position_embeddings": 16,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "llama_4_scaling_beta": 0.1,
        "rope_theta": 10000.0,
    },
    max_position_embeddings=32,
    tie_word_embeddings=False,
    rms_norm_eps=1e-5,
    attention_bias=False,
)
cfg._attn_implementation = "eager"

model = Mistral4ForCausalLM(cfg).eval()
# Non-trivial random weights (default init is tiny). mistral4 has no e_score_bias.
with torch.no_grad():
    for _, p in model.named_parameters():
        if p.dim() >= 2:
            p.normal_(0, 0.05)

print("=== state_dict tensors (names used by the C loader) ===")
for n, p in model.state_dict().items():
    print(f"  {n:55s} {tuple(p.shape)}")

# arbitrary ids; prompt(12) + 20 new = 32 tokens -> positions 16..31 cross original_max=16
prompt = [3, 14, 159, 26, 53, 58, 200, 11, 77, 240, 5, 99]
# RAW greedy (pure argmax), NOT model.generate(): the generate() wrapper diverges from
# plain arch-greedy (P0 finding 2026-07-15 -- it flips the first decode token even with a
# clean generation_config), while the C engine reproduces raw argmax. Verified: raw-greedy
# satisfies tf_pred[i]==full[i+1] 20/20; model.generate() only 8/20.
full = prompt[:]
for _ in range(20):
    with torch.no_grad():
        nxt = model(torch.tensor([full]), use_cache=False).logits[0, -1].argmax().item()
    full.append(int(nxt))

# teacher-forcing: single forward over the full sequence -> argmax per position.
# Validates the C engine's PREFILL separately from decode: tf_pred[i] == full[i+1]
# holds for i >= len(prompt)-1 under greedy.
with torch.no_grad():
    lg = model(torch.tensor([full]), use_cache=False).logits[0]   # [seq, vocab]
tf_pred = lg.argmax(-1).tolist()

print("prompt :", prompt)
print("full   :", full)
print("tf_pred:", tf_pred)

model.save_pretrained("mistral4_tiny", safe_serialization=True)
json.dump(cfg.to_dict(), open("mistral4_tiny/config.json", "w"))
json.dump(
    {"prompt_ids": prompt, "full_ids": full, "tf_pred": tf_pred},
    open("ref_mistral4.json", "w"),
)
print("\nsaved: mistral4_tiny/ (weights + config) and ref_mistral4.json")
