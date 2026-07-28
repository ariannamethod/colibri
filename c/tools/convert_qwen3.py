#!/usr/bin/env python3
"""Convert a Qwen3-30B-A3B-Base (qwen3_moe) HF BF16 checkpoint into the Colibri container.

Colibri's engine loads a model directory (SNAP=<dir>): config.json + one-or-more *.safetensors
(globbed by extension, lexicographically sorted, NO index.json read) + tokenizer.json. A weight
ships PRE-QUANTIZED iff a sibling tensor "<name>.qs" exists; the engine then infers int8/int4/int2
purely from the weight's byte count vs the config-derived dims O,I (glm.c:1099-1102):
    nbytes == O*I            -> int8
    nbytes == O*((I+1)/2)    -> int4   (two nibbles/byte)
    else                     -> int2

Precision policy (T5c): MoE experts and attention q/k/v/o -> int4, byte-exact to glm.c pack_int4
(glm.c:807-819) so the bytes are decoded verbatim by glm.c's matmul_i4 kernels (glm.c:377-380);
lm_head -> Q6_K, byte-exact to the notorch kernel nt_q6_k_rows (notorch.c:5136-5164) so the head
is read by nt_qmatvec dtype 14 with no repacking. Router (mlp.gate.weight — the loader requires it
full precision), all norms and embed_tokens (a lookup, never streamed per frame, so quantizing it
does not buy tempo) are copied VERBATIM.

The Q6_K head carries an EXPLICIT marker: config.json of the container gets "lm_head_dtype": 14
(the GGUF type code). Byte count alone cannot signal it — the loader's int8/int4/int2 inference
keys off O*I / O*((I+1)/2), and Q6_K is neither. The C frontend reads the marker AND asserts
nbytes == O*(I/256)*210, so a truncated or mismatched shard is a fatal detect, not silent garbage.

int4 format (per output row o, I input cols), byte-exact to glm.c:807-819:
    amax = max_i |w[o,i]| ;  scale[o] = max(amax/7, 1e-8)                (qmax = 7 for 4 bits)
    v    = clip(round(w[o,i]/scale[o]), -8, 7)                          (round-half-even == lrintf)
    byte[o, i>>1] = (v_even + 8) | ((v_odd + 8) << 4)                   (low nibble even col, high odd)
    row stride (I+1)/2 bytes; for odd I the trailing high nibble is a pad = (0+8) = 8 (ignored on decode)
    weight  -> uint8 [O, (I+1)/2]     scale (".qs") -> float32 [O]

Shard-by-shard and resumable: an output shard already present (non-empty) is skipped.

Offline conversion tooling (Python). The inference engine stays pure C.

Usage:
  python convert_qwen3.py --model <bf16_dir> --out <container_dir> [--ebits 4|8]
  python convert_qwen3.py --verify <container_dir> --model <bf16_dir> [--samples 8]
                          [--oracle-gguf <Q4_K_M.gguf>]   # cross-check the Q6_K head
"""
import argparse, json, re, shutil, sys
from pathlib import Path

try:
    import torch
    from safetensors.torch import load_file, save_file, safe_open
except ImportError as exc:
    sys.exit(f"missing deps: {exc}. need: torch, safetensors")

# Precision policy (T5c resident quant). Experts and attention -> int4 (the per-token bandwidth
# ballast); lm_head -> Q6_K. The head IS the output token distribution, so it stays above int4;
# Q6_K is the format the oracle's own head uses, at 0.82 B/weight against int8's 1.0 — 311MB ->
# 255MB off the per-token stream without dropping a precision class. Router (mlp.gate.weight —
# the loader requires it full precision), all norms, and embed_tokens (a lookup, never read
# per-frame, so quant does not help tempo) are copied verbatim.
EXPERT_RE = re.compile(r"model\.layers\.\d+\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight$")
ATTN_RE   = re.compile(r"model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$")

LM_HEAD_DTYPE_Q6K = 14   # GGUF type code, written to config.json as "lm_head_dtype"

def tensor_kind(name, ebits, resident, head_only=False):
    # head_only: kernel-isolation container — ONLY the head is quantized, everything else stays
    # verbatim, so any token flip vs the f32 oracle is attributable to the Q6_K head alone.
    if head_only: return 'q6k' if name == 'lm_head.weight' else 'copy'
    if EXPERT_RE.search(name): return 'int8' if ebits == 8 else 'int4'
    if resident and ATTN_RE.search(name): return 'int4'
    if resident and name == 'lm_head.weight': return 'q6k'
    return 'copy'   # router (mlp.gate.weight), norms, embed_tokens

# Small non-tensor files the engine / tokenizer read; copied as-is when present.
AUX_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json",
             "generation_config.json", "vocab.json", "merges.txt"]


def pack_int4(w):
    """w[O,I] float -> (uint8 [O,(I+1)/2], float32 [O]). Byte-exact to glm.c pack_int4 (glm.c:807-819)."""
    wf = w.float()
    O, I = wf.shape
    rb = (I + 1) // 2
    amax = wf.abs().amax(dim=1)                              # [O]
    scale = (amax / 7.0).clamp(min=1e-8)                     # s = amax/qmax, floor 1e-8
    q = torch.round(wf / scale[:, None]).clamp(-8, 7).to(torch.int16) + 8   # [-8,7] -> [0,15]
    lo = q[:, 0::2].to(torch.uint8)                          # even cols -> [O, rb]
    odd = q[:, 1::2].to(torch.uint8)                         # odd cols  -> [O, I//2]
    hi = torch.full((O, rb), 8, dtype=torch.uint8)          # pad high nibble = (v1=0)+8 = 8
    hi[:, :odd.shape[1]] = odd
    packed = lo | (hi << 4)
    return packed, scale.to(torch.float32)


def quantize_rows_int8(w):
    """w[O,I] float -> (int8 [O,I], float32 [O]). Byte-exact to glm.c quantize_rows (glm.c:794-803)."""
    wf = w.float()
    amax = wf.abs().amax(dim=1)
    scale = (amax / 127.0).clamp(min=1e-8)
    q = torch.round(wf / scale[:, None]).clamp(-128, 127).to(torch.int8)
    return q, scale.to(torch.float32)


def dequant_int4(packed, scale, I):
    """Reconstruct w_hat[O,I] from int4 container bytes, MIRRORING glm.c decode (glm.c:377-380):
       lo = (byte & 0xF) - 8 -> even col ; hi = (byte >> 4) - 8 -> odd col ; y = nibble * scale[o].
       Used only for --verify; a nibble-swap / transpose bug shows up as large reconstruction error."""
    p = packed.to(torch.int16)
    O, rb = p.shape
    lo = (p & 0xF) - 8                                       # [O, rb] -> even cols
    hi = (p >> 4) - 8                                        # [O, rb] -> odd cols
    w = torch.zeros((O, I), dtype=torch.float32)
    ncols_even = (I + 1) // 2
    ncols_odd = I // 2
    w[:, 0::2] = lo[:, :ncols_even].float()
    w[:, 1::2] = hi[:, :ncols_odd].float()
    return w * scale[:, None]


def dequant_int8(q8, scale):
    """Reconstruct w_hat[O,I] from int8 container: w_hat = int8 * scale[o]. Mirrors matmul_q."""
    return q8.to(torch.float32) * scale[:, None]


def cosd(a, b):
    """Cosine similarity accumulated in float64. f32 dot/norm over 100M+ elements (lm_head is
    311M) overflows the mantissa and returns cos>1 — a broken metric, not a container defect."""
    a = a.flatten().double(); b = b.flatten().double()
    return float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-30))


# ---- Q6_K: our quantizer, the byte-exact inverse of notorch nt_q6_k_rows (notorch.c:5136-5164).
# Super-block = 256 vals / 210 bytes: ql[128] low-4bit + qh[64] high-2bit + sc[16] int8 sub-scales
# + d f16 super-scale. Position p -> sub-block p/16 (scale sc[p/16]); dequant w = d*sc[p/16]*(q6-32).
_Q6K_IDX = None
def _q6k_idx():
    """Precompute position<->(ql byte/nibble, qh byte/shift) maps from the dequant interleave."""
    global _Q6K_IDX
    if _Q6K_IDX is not None:
        return _Q6K_IDX
    lo = [0]*128; hi = [0]*128; qhg = [[0]*64 for _ in range(4)]
    for p in range(256):
        h = p//128; hp = p % 128; g = hp//32; l = hp % 32
        qi = h*64 + l + (g % 2)*32
        (lo if g < 2 else hi)[qi] = p
        qhg[g][h*32 + l] = p
    _Q6K_IDX = (torch.tensor(lo), torch.tensor(hi), [torch.tensor(qhg[i]) for i in range(4)])
    return _Q6K_IDX


def pack_q6k(w, chunk=16384):
    """w[O,I] -> Q6_K bytes uint8 [O, (I//256)*210]. Per 16-val sub-block: symmetric scale
    absmax/31; the 16 sub-scales share a super-block d (int8 sc = round(scale/d)).

    absmax/31 (not /32): with /31 the extreme lands on +-31 exactly, with /32 the step is 3.2%
    finer but a positive extreme clips to 31 and eats absmax/32 of error on that element — which
    costs more than the finer step buys (0.0087 vs 0.0081 mean|err|/absmax). llama.cpp beats both
    by scaling off the SIGNED max, so its extreme is exact at either end; that policy difference,
    not indexing, is what --oracle-gguf measures.

    d is rounded to f16 BEFORE sc and q6 are derived from it: the kernel dequantizes with the f16
    value, so deriving the codes from the f32 one makes the quantizer and the decoder disagree and
    lifts the round-trip off the Q6 floor.

    Chunked over rows: the head is [151936,2048], and the [N,256] int32 intermediates are ~1.2GB
    apiece at full height.
    """
    lo, hi, qhg = _q6k_idx()
    O, I = w.shape
    assert I % 256 == 0, f"Q6_K needs I%256==0, got {I}"
    nb = I // 256
    out = []
    for r0 in range(0, O, chunk):
        blk = w[r0:r0+chunk].float().reshape(-1, 256)
        amax = blk.reshape(-1, 16, 16).abs().amax(dim=2)      # [N,16] per sub-block absmax
        ssub = amax / 31.0
        d = (ssub.amax(dim=1) / 127.0).to(torch.float16).float()   # [N] the value the kernel sees
        sc = torch.round(ssub / d.clamp(min=1e-30)[:, None]).clamp(0, 127).to(torch.int32)
        eff = (d[:, None] * sc).repeat_interleave(16, dim=1)  # [N,256] effective step
        q6 = (torch.round(blk / eff.clamp(min=1e-30)).clamp(-32, 31).to(torch.int32) + 32).to(torch.uint8)
        low4 = q6 & 15; hi2 = (q6 >> 4) & 3
        ql = (low4[:, lo] | (low4[:, hi] << 4)).to(torch.uint8)                 # [N,128]
        qh = (hi2[:, qhg[0]] | (hi2[:, qhg[1]] << 2) | (hi2[:, qhg[2]] << 4) | (hi2[:, qhg[3]] << 6)).to(torch.uint8)  # [N,64]
        scb = sc.to(torch.int8).view(torch.uint8)                               # [N,16]
        db = d.to(torch.float16).contiguous().view(torch.uint8).reshape(-1, 2)  # [N,2] LE
        out.append(torch.cat([ql, qh, scb, db], dim=1).reshape(-1, nb*210))
    return torch.cat(out, dim=0).contiguous()


def dequant_q6k(bytes_o, O, I, want_eff=False):
    """Reconstruct w_hat[O,I] mirroring nt_q6_k_rows, for verification. With want_eff, also
       returns the per-position step [O,I] (the Q6 quantization floor is step/4)."""
    lo, hi, qhg = _q6k_idx()
    nb = I // 256
    b = bytes_o.reshape(O*nb, 210)
    ql = b[:, :128].to(torch.int32); qh = b[:, 128:192].to(torch.int32)
    sc = b[:, 192:208].contiguous().view(torch.int8).to(torch.int32)       # [N,16]
    d = b[:, 208:210].contiguous().view(torch.float16).to(torch.float32).reshape(-1)   # [N]
    N = b.shape[0]
    low4 = torch.zeros((N, 256), dtype=torch.int32); hi2 = torch.zeros((N, 256), dtype=torch.int32)
    low4[:, lo] = ql & 0x0F; low4[:, hi] = ql >> 4
    for g in range(4):
        hi2[:, qhg[g]] = (qh >> (2*g)) & 3
    L = (low4 | (hi2 << 4)) - 32
    eff = (d[:, None] * sc).repeat_interleave(16, dim=1)                    # [N,256]
    w = (L.float() * eff).reshape(O, I)
    return (w, eff.reshape(O, I)) if want_eff else w


def verify_head_q6k(src_shard, out_shard, out_dir, oracle_gguf=None, chunk=8192):
    """Q6_K head proof in three independent legs:
       (a) round-trip — our bytes -> our decode -> per-row relL1 vs BF16, measured against the Q6
           floor (step/4). An index bug is O(1) error, not a floor-sized one, so it cannot hide;
       (b) indexing — our decode of the ORACLE's Q6_K bytes vs gguf-py's reference dequantizer.
           Exact agreement proves our position map is ggml's (== notorch's nt_q6_k_rows, which is
           the same interleave), independently of our quantizer;
       (c) policy — per-row relL1 of ours and of the oracle's bytes, both against BF16. That gap
           is the scaling policy (our absmax vs llama's signed-max, +- an imatrix we cannot see),
           and it is reported as a number.
       Cosine is accumulated in float64: an f32 dot over 311M elements overflows and returns >1."""
    hn = "lm_head.weight"
    cfg = json.loads((Path(out_dir) / "config.json").read_text())
    assert cfg.get("lm_head_dtype") == LM_HEAD_DTYPE_Q6K, \
        f'config.json lm_head_dtype={cfg.get("lm_head_dtype")!r}, expected {LM_HEAD_DTYPE_Q6K}'

    with safe_open(str(src_shard), framework="pt") as fs, safe_open(str(out_shard), framework="pt") as fo:
        O, I = fs.get_slice(hn).get_shape()
        got = list(fo.get_slice(hn).get_shape())
        exp = [O, (I // 256) * 210]
        assert got == exp, f"lm_head Q6_K bytes {got} != O*(I/256)*210 {exp}"
        print(f"  lm_head Q6_K: O={O} I={I} nbytes={O*exp[1]} == O*(I/256)*210, "
              f"config lm_head_dtype={LM_HEAD_DTYPE_Q6K} OK")

        # Aggregate err/floor, not a mean of per-row ratios: a row whose f16 super-scale underflows
        # has floor 0, and the ratio for it is +inf — one such row would swamp a mean and hide a
        # real packing bug. Underflow is counted separately and named for what it is.
        rel = torch.empty(O, dtype=torch.float64)
        err_sum = flo_sum = 0.0
        dead = 0
        dot = na = nb_ = 0.0
        for r0 in range(0, O, chunk):
            r1 = min(r0 + chunk, O)
            w0 = fs.get_slice(hn)[r0:r1].float()
            wh, eff = dequant_q6k(fo.get_slice(hn)[r0:r1], r1 - r0, I, want_eff=True)
            den = w0.abs().mean(dim=1).clamp(min=1e-12)
            rel[r0:r1] = ((wh - w0).abs().mean(dim=1) / den).double()
            err_sum += float((wh - w0).abs().double().sum())
            flo_sum += float((eff.double() / 4).sum())
            dead += int(((eff.amax(dim=1) == 0) & (w0.abs().amax(dim=1) > 0)).sum())
            a, b = wh.double(), w0.double()
            dot += float((a * b).sum()); na += float((a * a).sum()); nb_ += float((b * b).sum())
        cos = dot / (na ** 0.5 * nb_ ** 0.5 + 1e-300)
        agg = err_sum / max(flo_sum, 1e-300)
        print(f"  (a) round-trip vs BF16, N={O} rows: relL1 mean={rel.mean():.6f} max={rel.max():.6f} | "
              f"err/Q6-floor aggregate={agg:.4f} | f16-d underflow rows={dead} | cos(f64)={cos:.8f}")
        assert agg <= 1.15, f"round-trip {agg:.4f}x above the Q6 floor — packing bug"
        # A sub-block whose absmax < 31*127*5.96e-8 = 2.35e-4 underflows the f16 super-scale and
        # decodes to zero. Inherent to Q6_K (llama.cpp shares it), but on the head it would pin a
        # token's logit at 0, so it is a stop-and-report, not a warning.
        assert dead == 0, f"{dead} rows underflow the f16 super-scale (subblock absmax < 2.35e-4)"

        if not oracle_gguf:
            print("  (b,c) skipped: no --oracle-gguf")
            return
        import numpy as np
        from gguf import GGUFReader, GGMLQuantizationType
        from gguf.quants import dequantize as ref_dequant
        rd = GGUFReader(oracle_gguf)
        ot = next((x for x in rd.tensors if x.name == "output.weight"), None)
        assert ot is not None, "output.weight not in the oracle GGUF"
        assert ot.tensor_type == GGMLQuantizationType.Q6_K, f"oracle head is {ot.tensor_type}, not Q6_K"
        ob = torch.from_numpy(np.ascontiguousarray(ot.data).view(np.uint8).reshape(-1))
        assert ob.numel() == O * exp[1], f"oracle bytes {ob.numel()} != ours {O*exp[1]}"
        ob = ob.reshape(O, exp[1])
        print(f"  oracle output.weight: {ot.tensor_type.name} ne={list(ot.shape)} bytes={ob.numel()}")

        # Run the reference decoder against BOTH the oracle's bytes and the bytes we ship: the
        # first proves our position map is ggml's, the second proves the container we just wrote
        # is read identically by the reference implementation.
        n = min(4096, O)
        for tag, bb in (("oracle", ob[:n].contiguous()),
                        ("OUR container", fo.get_slice(hn)[0:n].contiguous())):
            mine = dequant_q6k(bb, n, I)
            ref = torch.from_numpy(
                ref_dequant(bb.numpy(), GGMLQuantizationType.Q6_K).astype(np.float32)).reshape(n, I)
            mx = float((mine - ref).abs().max())
            print(f"  (b) our decode vs gguf-py reference on {tag} bytes, N={n} rows: "
                  f"max|diff|={mx:.3e} {'EXACT' if mx == 0.0 else 'MISMATCH'}")
            assert mx == 0.0, f"Q6_K position map disagrees with the reference on {tag} bytes"

        orel = torch.empty(O, dtype=torch.float64)
        odot = ona = onb = 0.0
        for r0 in range(0, O, chunk):
            r1 = min(r0 + chunk, O)
            w0 = fs.get_slice(hn)[r0:r1].float()
            oh = dequant_q6k(ob[r0:r1].contiguous(), r1 - r0, I)
            den = w0.abs().mean(dim=1).clamp(min=1e-12)
            orel[r0:r1] = ((oh - w0).abs().mean(dim=1) / den).double()
            a, b = oh.double(), w0.double()
            odot += float((a * b).sum()); ona += float((a * a).sum()); onb += float((b * b).sum())
        ocos = odot / (ona ** 0.5 * onb ** 0.5 + 1e-300)
        print(f"  (c) vs BF16, N={O} rows: ours relL1 mean={rel.mean():.6f} cos={cos:.8f} | "
              f"oracle relL1 mean={orel.mean():.6f} cos={ocos:.8f} | "
              f"ours/oracle={float(rel.mean()/orel.mean()):.4f} | "
              f"rows where ours is closer: {int((rel < orel).sum())}/{O}")
        print("  (c) gap source = scaling policy (our absmax/31 vs llama signed-max), NOT indexing "
              "(proved exact in (b)). imatrix: no log on this box — UNCONFIRMED.")


def convert(src_dir, out_dir, ebits, resident, head_only=False):
    src, out = Path(src_dir), Path(out_dir)
    if not (src / "config.json").is_file():
        sys.exit(f"config.json missing in {src}")
    out.mkdir(parents=True, exist_ok=True)

    for fn in AUX_FILES:
        if (src / fn).is_file():
            shutil.copy2(src / fn, out / fn)
            print(f"  copied {fn}")

    # Explicit head marker. Byte count cannot signal Q6_K (the loader only infers int8/int4/int2),
    # so the container states it and the C frontend cross-checks nbytes == O*(I/256)*210.
    if tensor_kind("lm_head.weight", ebits, resident, head_only) == 'q6k':
        cfg = json.loads((out / "config.json").read_text())
        cfg["lm_head_dtype"] = LM_HEAD_DTYPE_Q6K
        (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"  config.json: lm_head_dtype={LM_HEAD_DTYPE_Q6K} (Q6_K)")

    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        sys.exit(f"no safetensors in {src}")

    nq = {'int4': 0, 'int8': 0, 'q6k': 0}
    bytes_src = bytes_q = 0
    for si, shard in enumerate(shards, 1):
        out_shard = out / shard.name
        if out_shard.is_file() and out_shard.stat().st_size > 0:
            print(f"[{si}/{len(shards)}] {shard.name}: exists, skip (resume)")
            continue
        print(f"[{si}/{len(shards)}] {shard.name} ...", end=" ", flush=True)
        tensors = load_file(str(shard))
        outt = {}
        for name, t in tensors.items():
            kind = tensor_kind(name, ebits, resident, head_only)
            if kind == 'copy':
                outt[name] = t.contiguous()
                continue
            if kind == 'q6k':
                q, s = pack_q6k(t), None     # scales live inside the block: no ".qs" sibling
            elif kind == 'int8':
                q, s = quantize_rows_int8(t)
            else:
                q, s = pack_int4(t)
            outt[name] = q.contiguous()
            if s is not None:
                outt[name + ".qs"] = s.contiguous()
            nq[kind] += 1
            bytes_src += t.numel() * t.element_size()
            bytes_q += q.numel() * q.element_size() + (s.numel() * 4 if s is not None else 0)
        save_file(outt, str(out_shard))
        print("ok")

    ratio = bytes_q / max(bytes_src, 1) * 100
    print(f"\ndone: {nq['int4']} tensors -> int4, {nq['int8']} -> int8, {nq['q6k']} -> Q6_K")
    print(f"quantized storage: {bytes_src/1e9:.2f} GB -> {bytes_q/1e9:.2f} GB ({ratio:.1f}%)")
    print(f"container: {out}   (SNAP={out} for the engine)")


def verify(out_dir, src_dir, samples, oracle_gguf=None):
    """Round-trip proof: for sampled expert tensors, dequant the container bytes with the
       glm.c-mirroring decode and compare to the ORIGINAL bf16 weight. Also asserts the byte
       layout the loader infers int4 from: nbytes == O*((I+1)/2), scale is F32[O]."""
    src, out = Path(src_dir), Path(out_dir)
    out_shards = sorted(out.glob("*.safetensors"))
    if not out_shards:
        sys.exit(f"no container shards in {out}")

    # index: expert weight name -> (out_shard, src_shard)
    def index(dir_):
        idx = {}
        for sh in sorted(dir_.glob("*.safetensors")):
            with safe_open(str(sh), framework="pt") as f:
                for k in f.keys():
                    idx[k] = sh
        return idx

    oidx, sidx = index(out), index(src)
    q4_names = [k for k in oidx if (EXPERT_RE.search(k) or ATTN_RE.search(k)) and (k + ".qs") in oidx]
    if not q4_names:
        sys.exit("no int4 tensors found in container")
    step = max(1, len(q4_names) // samples)
    picks = q4_names[::step][:samples]

    worst = 0.0
    print(f"verifying {len(picks)} sampled int4 tensors of {len(q4_names)} (experts+attn):")
    for name in picks:
        with safe_open(str(sidx[name]), framework="pt") as f:
            w0 = f.get_tensor(name).float()
        O, I = w0.shape
        with safe_open(str(oidx[name]), framework="pt") as f:
            packed = f.get_tensor(name)
            scale = f.get_tensor(name + ".qs")
        # byte-count contract the loader keys off (glm.c:1099-1102)
        nbytes = packed.numel() * packed.element_size()
        assert nbytes == O * ((I + 1) // 2), f"{name}: nbytes {nbytes} != O*((I+1)/2) {O*((I+1)//2)}"
        assert scale.dtype == torch.float32 and scale.numel() == O, f"{name}: bad .qs {scale.dtype} {scale.numel()}"
        w_hat = dequant_int4(packed, scale.float(), I)
        denom = w0.abs().mean().clamp(min=1e-9)
        relL1 = float((w_hat - w0).abs().mean() / denom)
        # theoretical uniform-quant L1 floor for this row-symmetric int4: step==scale, mean|err|~step/4.
        # Correct int4 lands ON this floor; per-row int4 rel-L1 is ~0.15-0.20 on real weights (I large ->
        # amax reaches farther -> coarser step), NOT a defect.
        floor = float((scale.float()[:, None].expand_as(w0) / 4).mean() / denom)
        cos = cosd(w_hat, w0)
        # Grounded bug-detector: a transpose / nibble-swap scrambles element positions -> relL1 >> floor
        # and cos << 0.9. Correct packing -> relL1 == floor and cos ~ 0.98. Absolute rel-error is NOT a
        # bug signal (it is the int4 quant floor), so we test error-vs-floor and cosine, not a magic number.
        ok = (relL1 <= 1.15 * floor + 1e-6) and (cos >= 0.97)
        worst = max(worst, relL1)
        print(f"  {name}: O={O} I={I} nbytes={nbytes} relL1={relL1:.4f} floor={floor:.4f} cos={cos:.5f} {'OK' if ok else 'BUG'}")
        if not ok:
            sys.exit(f"FAIL {name}: relL1 {relL1:.4f} vs floor {floor:.4f}, cos {cos:.5f} — packing/indexing bug")
    print(f"\nall {len(picks)} sampled int4 tensors on the quant floor, cos >= 0.97 — byte-correct "
          f"(worst relL1 {worst:.4f})")
    hn = "lm_head.weight"
    if hn in oidx and (hn + ".qs") in oidx:
        with safe_open(str(sidx[hn]), framework="pt") as f:
            w0 = f.get_tensor(hn).float()
        O, I = w0.shape
        with safe_open(str(oidx[hn]), framework="pt") as f:
            pk = f.get_tensor(hn); sc = f.get_tensor(hn + ".qs")
        nb = pk.numel() * pk.element_size()
        assert nb == O * I, f"lm_head nbytes {nb} != int8 {O*I}"
        wh = dequant_int8(pk, sc.float())
        relL1 = float((wh - w0).abs().mean() / w0.abs().mean().clamp(min=1e-9))
        cos = cosd(wh, w0)
        ok = cos >= 0.999
        print(f"  lm_head int8: O={O} I={I} nbytes={nb} relL1={relL1:.4f} cos={cos:.5f} {'OK' if ok else 'BUG'}")
        if not ok:
            sys.exit(f"lm_head int8 packing bug (cos {cos:.5f})")
    elif hn in oidx:
        verify_head_q6k(sidx[hn], oidx[hn], out, oracle_gguf)
    print("VERIFY PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="source HF bf16 checkpoint dir")
    ap.add_argument("--out", help="output container dir (convert mode)")
    ap.add_argument("--ebits", type=int, default=4, choices=(4, 8), help="expert quant bits (default 4)")
    ap.add_argument("--verify", metavar="CONTAINER", help="verify a produced container against --model")
    ap.add_argument("--samples", type=int, default=8, help="quantized tensors to sample in --verify")
    ap.add_argument("--experts-only", action="store_true",
                    help="quantize experts only, copy attention+head verbatim (pre-T5 container)")
    ap.add_argument("--head-only", action="store_true",
                    help="quantize ONLY lm_head to Q6_K, copy everything else verbatim "
                         "(kernel-isolation container: a flip is then the head's alone)")
    ap.add_argument("--oracle-gguf", metavar="GGUF",
                    help="Q4_K_M gguf whose output.weight is Q6_K; cross-checks the head in --verify")
    a = ap.parse_args()
    if a.verify:
        verify(a.verify, a.model, a.samples, a.oracle_gguf)
    elif a.out:
        convert(a.model, a.out, a.ebits, resident=not a.experts_only, head_only=a.head_only)
    else:
        ap.error("need --out (convert) or --verify CONTAINER")


if __name__ == "__main__":
    main()
