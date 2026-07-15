"""Convert a Mistral Small 4 (mistral4) checkpoint into the Colibri container layout.

The engine loads experts PER-EXPERT (model.layers.N.mlp.experts.E.{gate,up,down}_proj.weight),
but mistral4 packs them 3D: mlp.experts.gate_up_proj (E, 2*moe_inter, D) and
mlp.experts.down_proj (E, D, moe_inter). This unpacks the 3D tensors into the
per-expert names the engine reads. Everything else (MLA, shared experts, router
mlp.gate.weight, norms, embed, lm_head) is copied as-is.

gate_up_proj[e] rows [0:I] = gate_proj, [I:2I] = up_proj (torch chunk(2, dim=-1) after
linear, verbatim modeling_mistral4.Mistral4Experts.forward).

Usage: python convert_mistral4.py <src_dir> <dst_dir> [--f32]
  --f32: write float32 (P0/tiny). Default keeps source dtype. (int4/.qs path: later, P1.)

Offline conversion tooling (Colibri Python class). The inference engine is pure C.
"""
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    if len(sys.argv) < 3:
        print("usage: convert_mistral4.py <src_dir> <dst_dir> [--f32]", file=sys.stderr)
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    force_f32 = "--f32" in sys.argv[3:]
    os.makedirs(dst, exist_ok=True)

    cfg = json.load(open(f"{src}/config.json"))
    E = cfg["n_routed_experts"]
    I = cfg["moe_intermediate_size"]

    out = {}
    n_exp_tensors = 0
    with safe_open(f"{src}/model.safetensors", framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if force_f32 and t.dtype != torch.float32:
                t = t.to(torch.float32)
            if "mlp.experts.gate_up_proj" in k:
                layer = k.split(".")[2]
                gu = t  # (E, 2I, D)
                assert gu.shape[0] == E and gu.shape[1] == 2 * I, f"gate_up shape {tuple(gu.shape)}"
                for e in range(E):
                    out[f"model.layers.{layer}.mlp.experts.{e}.gate_proj.weight"] = gu[e, :I, :].contiguous()
                    out[f"model.layers.{layer}.mlp.experts.{e}.up_proj.weight"] = gu[e, I:2 * I, :].contiguous()
                    n_exp_tensors += 2
            elif "mlp.experts.down_proj" in k:
                layer = k.split(".")[2]
                dp = t  # (E, D, I)
                assert dp.shape[0] == E and dp.shape[2] == I, f"down shape {tuple(dp.shape)}"
                for e in range(E):
                    out[f"model.layers.{layer}.mlp.experts.{e}.down_proj.weight"] = dp[e].contiguous()
                    n_exp_tensors += 1
            else:
                out[k] = t

    save_file(out, f"{dst}/model.safetensors")
    json.dump(cfg, open(f"{dst}/config.json", "w"))
    print(f"wrote {len(out)} tensors ({n_exp_tensors} per-expert) to {dst}/model.safetensors")
    # sanity: report a couple of expert shapes
    g0 = out["model.layers.0.mlp.experts.0.gate_proj.weight"].shape
    u0 = out["model.layers.0.mlp.experts.0.up_proj.weight"].shape
    d0 = out["model.layers.0.mlp.experts.0.down_proj.weight"].shape
    print(f"expert[0] gate={tuple(g0)} up={tuple(u0)} down={tuple(d0)}  (expect gate/up=(I,D), down=(D,I))")
    print(f"gate_up_proj present in container: {any('gate_up_proj' in k for k in out)}  (expect False)")


if __name__ == "__main__":
    main()
