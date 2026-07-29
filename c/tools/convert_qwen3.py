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
EXPERT_DTYPE_Q4K  = 12   # GGUF type code, written to config.json as "expert_dtype" (T5f)

# T5f move 3 — the oracle's "Q4_K_M" is a MIX, not uniform Q4_K. Read out of the GGUF itself
# (which blk.N carry type 14), not derived from ggml's mixing rules:
#   ffn_down_exps and attn_v are Q6_K on exactly these 24 layers, Q4_K on the other 24.
# Reproducing the recipe from data is the point; the layer list is evidence, not a formula.
MIX_Q6_LAYERS = frozenset([0,1,2,3,4,5, 8,11,14,17,20,23,26,29,32,35,38,41, 42,43,44,45,46,47])
DOWN_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.\d+\.down_proj\.weight$")
VPROJ_RE = re.compile(r"model\.layers\.(\d+)\.self_attn\.v_proj\.weight$")

def _mix_is_q6(name):
    """True iff this tensor sits where the oracle puts Q6_K. Both roles have I%256==0
    (down_proj I=moe_inter=768, v_proj I=hidden=2048), so Q6_K is representable."""
    m = DOWN_RE.search(name) or VPROJ_RE.search(name)
    return bool(m) and int(m.group(1)) in MIX_Q6_LAYERS

def tensor_kind(name, ebits, resident, head_only=False, eq4k=False, aq4k=False, mix=False):
    # head_only: kernel-isolation container — ONLY the head is quantized, everything else stays
    # verbatim, so any token flip vs the f32 oracle is attributable to the Q6_K head alone.
    if head_only: return 'q6k' if name == 'lm_head.weight' else 'copy'
    # eq4k (T5f): experts move from symmetric per-row int4 to Q4_K. Measured cost of the old
    # policy was +1.64 PPL against the oracle's Q4_K on the same text; the price is 4.5 bits
    # per weight against 4.0, i.e. ~113 MB more per token.
    if mix and _mix_is_q6(name): return 'q6k'     # move 3: the oracle's Q6_K positions
    if EXPERT_RE.search(name): return 'q4k' if eq4k else ('int8' if ebits == 8 else 'int4')
    # T5f move 2: attention was the other half of the measured gap (+1.21 PPL from the T4b
    # decomposition), and every attn projection has I divisible by 256 (2048 and 4096), so the
    # same Q4_K path applies. Router stays verbatim — the loader requires it full precision.
    if resident and ATTN_RE.search(name): return 'q4k' if aq4k else 'int4'
    if resident and name == 'lm_head.weight': return 'q6k'
    return 'copy'   # router (mlp.gate.weight), norms, embed_tokens

# Small non-tensor files the engine / tokenizer read; copied as-is when present.
AUX_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json",
             "generation_config.json", "vocab.json", "merges.txt"]


def pack_int4(w, rmse=False):
    """w[O,I] float -> (uint8 [O,(I+1)/2], float32 [O]). Byte-exact to glm.c pack_int4 (glm.c:807-819).

    rmse=False keeps the shipped amax/7 scale. rmse=True is T5f move 0: the BYTES and the
    format are untouched, only the per-row scale is chosen better. amax/7 is set by the single
    most extreme weight in the row, so one outlier coarsens all 2047 others. ggml's
    make_qx_quants instead sweeps candidate scales and keeps the one minimising squared error;
    for a fixed code vector q the optimal scale is sumlx/suml2, and maximising sumlx^2/suml2
    picks the best candidate. Same 4 bits, same row layout, same decoder — free accuracy.
    """
    wf = w.float()
    O, I = wf.shape
    rb = (I + 1) // 2
    amax = wf.abs().amax(dim=1)                              # [O]
    scale = (amax / 7.0).clamp(min=1e-8)                     # s = amax/qmax, floor 1e-8
    if rmse:
        nmax = 7.0
        amax_s = amax.clamp(min=1e-8)
        best = torch.full((O,), -1.0)
        best_q = torch.round(wf / scale[:, None]).clamp(-8, 7)
        for it in range(-9, 10):
            iscale = (0.1 * it + nmax) / amax_s              # [O]
            q = torch.round(wf * iscale[:, None]).clamp(-8, 7)
            sumlx = (wf * q).sum(dim=1)
            suml2 = (q * q).sum(dim=1)
            score = torch.where(suml2 > 0, sumlx * sumlx / suml2.clamp(min=1e-30),
                                torch.zeros_like(suml2))
            upd = (score > best) & (suml2 > 0)
            best = torch.where(upd, score, best)
            scale = torch.where(upd, sumlx / suml2.clamp(min=1e-30), scale)
            best_q = torch.where(upd[:, None], q, best_q)
        # scale may legitimately come out negative (sumlx<0); the decoder multiplies, so the
        # sign rides along. Guard only against an exact zero.
        scale = torch.where(scale.abs() < 1e-12, torch.full_like(scale, 1e-8), scale)
        q = best_q.clamp(-8, 7).to(torch.int16) + 8
        lo = q[:, 0::2].to(torch.uint8)
        odd = q[:, 1::2].to(torch.uint8)
        hi = torch.full((O, rb), 8, dtype=torch.uint8)
        hi[:, :odd.shape[1]] = odd
        return (lo | (hi << 4)), scale.to(torch.float32)
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


# ---- Q4_K: inverse of the notorch kernel nt_q4_k_rows + nt_get_scale_min_k4.
# Super-block = 256 values / 144 bytes: d f16 | dmin f16 | sc[12] | qs[128].
# sc[12] packs eight (scale,min) pairs at 6 bits each. Position p: sub-block is = p/32,
# nibble at qs[(p/64)*32 + p%32] (low when p%64 < 32, high otherwise), and
#     w[p] = d * s[is] * q[p] - dmin * m[is],  q in [0,15], s,m in [0,63].
# Asymmetric with an explicit min — that is exactly why it beats our symmetric per-row int4
# at the same nibble width (it costs 4.5 bits/weight against 4.0, the price of the sub-scales).
# Inverting nt_get_scale_min_k4 for j = 0..7:
#     j<4 : sc[j]   = s[j] | ((s[j+4]>>4)<<6)  ;  sc[j+4] = m[j] | ((m[j+4]>>4)<<6)
#     j>=4: sc[j+4] = (s[j]&0xF) | ((m[j]&0xF)<<4)
_Q4K_IDX = None
def _q4k_idx():
    """qs byte i holds the low nibble of one position and the high nibble of another."""
    global _Q4K_IDX
    if _Q4K_IDX is not None:
        return _Q4K_IDX
    lo = [0]*128; hi = [0]*128
    for i in range(128):
        base = (i // 32) * 64 + (i % 32)
        lo[i] = base            # p%64 < 32  -> low nibble
        hi[i] = base + 32       # p%64 >= 32 -> high nibble
    _Q4K_IDX = (torch.tensor(lo), torch.tensor(hi))
    return _Q4K_IDX


def _qkx2_scale_min(sub, nstep=8):
    """(scale, min) per 32-value sub-block by weighted least squares — the make_qkx2_quants
    criterion, read out of ggml-quants.c and reimplemented here vectorised.

    sub[N,8,32] -> scales[N,8], mins[N,8] (mins non-negative, as the format stores them).

    Why this beats plain min/max, which is what T5f shipped first: min/max lets the two
    extreme samples of a sub-block fix the whole affine grid. Here the codes are frozen for
    a candidate start scale and the (scale, min) pair is then FITTED to all 32 samples by
    weighted least squares, with w = av_x + |x| so the large-magnitude entries — the ones a
    dot product actually notices — pull harder. min is clamped at 0 from above, exactly as
    the reference does, because the format encodes -min as a non-negative number.
    nstep candidates are swept; each is accepted only if it lowers the weighted error, so the
    result can never be worse than the starting grid.
    """
    N = sub.shape[0]
    x = sub
    av_x = torch.sqrt((x * x).mean(dim=2))                    # [N,8]
    w = av_x[..., None] + x.abs()                             # [N,8,32]
    mn0 = x.amin(dim=2).clamp(max=0.0)                        # min<=0, per the reference
    mx0 = x.amax(dim=2)
    rng = (mx0 - mn0)
    flat = rng <= 0                                           # constant sub-block
    rng_s = rng.clamp(min=1e-30)

    sum_w = w.sum(dim=2)
    sum_x = (w * x).sum(dim=2)

    scale = rng_s / 15.0
    mn = mn0.clone()
    L0 = torch.round((15.0 / rng_s)[..., None] * (x - mn0[..., None])).clamp(0, 15)
    best = (w * (scale[..., None] * L0 + mn0[..., None] - x) ** 2).sum(dim=2)

    for it in range(nstep + 1):
        iscale = (-1.0 + 0.1 * it + 15.0) / rng_s             # [N,8]
        L = torch.round(iscale[..., None] * (x - mn0[..., None])).clamp(0, 15)
        sum_l  = (w * L).sum(dim=2)
        sum_l2 = (w * L * L).sum(dim=2)
        sum_xl = (w * L * x).sum(dim=2)
        D = sum_w * sum_l2 - sum_l * sum_l
        ok = D > 0
        Ds = torch.where(ok, D, torch.ones_like(D))
        ts = (sum_w * sum_xl - sum_x * sum_l) / Ds
        tm = (sum_l2 * sum_x - sum_l * sum_xl) / Ds
        pos = tm > 0                                          # min must not go positive
        ts = torch.where(pos, sum_xl / sum_l2.clamp(min=1e-30), ts)
        tm = torch.where(pos, torch.zeros_like(tm), tm)
        err = (w * (ts[..., None] * L + tm[..., None] - x) ** 2).sum(dim=2)
        upd = ok & (err < best)
        best  = torch.where(upd, err, best)
        scale = torch.where(upd, ts, scale)
        mn    = torch.where(upd, tm, mn)

    scales = torch.where(flat, torch.zeros_like(scale), scale).clamp(min=0.0)
    mins   = torch.where(flat, (-mn0).clamp(min=0.0), (-mn).clamp(min=0.0))
    return scales, mins


def pack_q4k(w, chunk=4096, nstep=8):
    """w[O,I] -> Q4_K bytes uint8 [O, (I//256)*144]. Per 32-value sub-block: an affine
    (scale, min) pair; the eight pairs of a super-block share d and dmin, quantized to 6 bits.
    d and dmin are rounded to f16 BEFORE the codes are derived from them — the kernel
    dequantizes with the f16 values, so deriving codes from the f32 ones would put the
    round-trip above the format floor (the lesson from the Q6_K super-scale)."""
    lo_pos, hi_pos = _q4k_idx()
    O, I = w.shape
    assert I % 256 == 0, f"Q4_K needs I%256==0, got {I}"
    nb = I // 256
    out = []
    for r0 in range(0, O, chunk):
        blk = w[r0:r0+chunk].float().reshape(-1, 256)
        N = blk.shape[0]
        sub = blk.reshape(N, 8, 32)
        scales, mins = _qkx2_scale_min(sub, nstep=nstep)
        d    = (scales.amax(dim=1) / 63.0).to(torch.float16).float()   # [N]
        dmin = (mins.amax(dim=1)   / 63.0).to(torch.float16).float()
        ls = torch.round(scales / d.clamp(min=1e-30)[:, None]).clamp(0, 63).to(torch.int32)
        lm = torch.round(mins   / dmin.clamp(min=1e-30)[:, None]).clamp(0, 63).to(torch.int32)
        eff_s = (d[:, None] * ls).repeat_interleave(32, dim=1)
        eff_m = (dmin[:, None] * lm).repeat_interleave(32, dim=1)
        q = torch.round((blk + eff_m) / eff_s.clamp(min=1e-30)).clamp(0, 15).to(torch.int32)
        qs = (q[:, lo_pos] | (q[:, hi_pos] << 4)).to(torch.uint8)      # [N,128]
        sc = torch.zeros((N, 12), dtype=torch.int32)
        for j in range(4):
            sc[:, j]     = (ls[:, j] & 63) | ((ls[:, j+4] >> 4) << 6)
            sc[:, j+4]   = (lm[:, j] & 63) | ((lm[:, j+4] >> 4) << 6)
            sc[:, j+8]   = (ls[:, j+4] & 0x0F) | ((lm[:, j+4] & 0x0F) << 4)
        db  = d.to(torch.float16).contiguous().view(torch.uint8).reshape(-1, 2)
        dmb = dmin.to(torch.float16).contiguous().view(torch.uint8).reshape(-1, 2)
        out.append(torch.cat([db, dmb, sc.to(torch.uint8), qs], dim=1).reshape(-1, nb*144))
    return torch.cat(out, dim=0).contiguous()


def dequant_q4k(bytes_o, O, I, want_eff=False):
    """Mirror of nt_q4_k_rows, for verification. want_eff also returns the per-position step
    (the Q4_K floor is step/2 for a uniform affine grid)."""
    lo_pos, hi_pos = _q4k_idx()
    nb = I // 256
    b = bytes_o.reshape(O*nb, 144)
    N = b.shape[0]
    d    = b[:, 0:2].contiguous().view(torch.float16).to(torch.float32).reshape(-1)
    dmin = b[:, 2:4].contiguous().view(torch.float16).to(torch.float32).reshape(-1)
    sc   = b[:, 4:16].to(torch.int32)
    qs   = b[:, 16:144].to(torch.int32)
    ls = torch.zeros((N, 8), dtype=torch.int32); lm = torch.zeros((N, 8), dtype=torch.int32)
    for j in range(8):
        if j < 4:
            ls[:, j] = sc[:, j] & 63
            lm[:, j] = sc[:, j+4] & 63
        else:
            ls[:, j] = (sc[:, j+4] & 0x0F) | ((sc[:, j-4] >> 6) << 4)
            lm[:, j] = (sc[:, j+4] >> 4)   | ((sc[:, j]   >> 6) << 4)
    q = torch.zeros((N, 256), dtype=torch.int32)
    q[:, lo_pos] = qs & 0x0F
    q[:, hi_pos] = qs >> 4
    eff_s = (d[:, None] * ls).repeat_interleave(32, dim=1)
    eff_m = (dmin[:, None] * lm).repeat_interleave(32, dim=1)
    w = (q.float() * eff_s - eff_m).reshape(O, I)
    return (w, eff_s.reshape(O, I)) if want_eff else w


def verify_experts_q4k(oidx, sidx, out_dir, samples=8, oracle_gguf=None):
    """Q4_K expert proof, same three legs that gated the Q6_K head:
       (a) round-trip — our bytes -> our decode -> per-row relL1 vs BF16 against the Q4_K
           floor (step/2 for a uniform affine grid). An index bug is O(1) error, not floor-sized;
       (b) indexing — our decode of the ORACLE's Q4_K bytes vs gguf-py's reference dequantizer.
           Exact agreement proves our position map is ggml's, independently of our quantizer;
       (c) byte contract — nbytes == O*(I/256)*144 and no stray .qs sibling.
       Cosine in float64 throughout."""
    cfg = json.loads((Path(out_dir) / "config.json").read_text())
    assert cfg.get("expert_dtype") == EXPERT_DTYPE_Q4K, \
        f'config.json expert_dtype={cfg.get("expert_dtype")!r}, expected {EXPERT_DTYPE_Q4K}'
    # T5f move 3: a container may MIX the two block formats. The byte count identifies which
    # one a tensor is (144 vs 210 per 256 values), exactly as the engine's blockq_kind_of does,
    # so verification never needs a layer list either.
    names = sorted(k for k in oidx if EXPERT_RE.search(k) or VPROJ_RE.search(k))
    if not names:
        sys.exit("no expert/v_proj tensors in container")
    step = max(1, len(names) // samples)
    picks = names[::step][:samples]
    print(f"verifying {len(picks)} sampled block-quant tensors of {len(names)}:")
    worst = 0.0
    seen = set()
    for name in picks:
        with safe_open(str(sidx[name]), framework="pt") as f:
            w0 = f.get_tensor(name).float()
        O, I = w0.shape
        with safe_open(str(oidx[name]), framework="pt") as f:
            pk = f.get_tensor(name)
        nb = pk.numel() * pk.element_size()
        nblk = O * (I // 256)
        if   nb == nblk * 144: kind, wh, eff = 'Q4_K', *dequant_q4k(pk, O, I, want_eff=True)
        elif nb == nblk * 210: kind, wh, eff = 'Q6_K', *dequant_q6k(pk, O, I, want_eff=True)
        else: sys.exit(f"{name}: {nb} B is neither Q4_K ({nblk*144}) nor Q6_K ({nblk*210})")
        assert (name + ".qs") not in oidx, f"{name}: unexpected .qs sibling on a {kind} tensor"
        seen.add(kind)
        den = w0.abs().mean().clamp(min=1e-12)
        rel = float((wh - w0).abs().mean() / den)
        floor = float((eff / (2 if kind == 'Q4_K' else 4)).mean() / den)
        cos = cosd(wh, w0)
        dead = int(((eff.amax(dim=1) == 0) & (w0.abs().amax(dim=1) > 0)).sum())
        worst = max(worst, rel)
        ok = rel <= 1.05 * floor and cos >= 0.99
        print(f"  {name}: {kind} O={O} I={I} nbytes={nb} relL1={rel:.4f} floor={floor:.4f} "
              f"cos={cos:.5f} dead={dead} {'OK' if ok else 'BUG'}")
        if not ok:
            sys.exit(f"FAIL {name}: relL1 {rel:.4f} vs floor {floor:.4f}, cos {cos:.5f}")
    print(f"all {len(picks)} sampled on the quant floor (worst relL1 {worst:.4f}); "
          f"formats present in sample: {sorted(seen)}")
    if not oracle_gguf:
        print("  (indexing leg skipped: no --oracle-gguf)")
        return
    import numpy as np
    from gguf import GGUFReader, GGMLQuantizationType
    from gguf.quants import dequantize as ref_dequant
    rd = GGUFReader(oracle_gguf)
    ot = next((x for x in rd.tensors if x.tensor_type == GGMLQuantizationType.Q4_K), None)
    assert ot is not None, "no Q4_K tensor in the oracle GGUF to check indexing against"
    ob = torch.from_numpy(np.ascontiguousarray(ot.data).view(np.uint8).reshape(-1))
    nblk = ob.numel() // 144
    n = min(4096, nblk)
    sub = ob[:n*144].contiguous()
    ours = dequant_q4k(sub, n, 256)
    ref = torch.from_numpy(
        ref_dequant(sub.numpy(), GGMLQuantizationType.Q4_K).astype(np.float32)).reshape(n, 256)
    mx = float((ours - ref).abs().max())
    print(f"  our decode vs gguf-py reference on oracle Q4_K bytes ({ot.name}, {n} blocks): "
          f"max|diff|={mx:.3e} {'EXACT' if mx == 0.0 else 'MISMATCH'}")
    assert mx == 0.0, "our Q4_K position map disagrees with the reference — indexing bug"


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


def convert(src_dir, out_dir, ebits, resident, head_only=False, eq4k=False, rmse=False, nstep=12, aq4k=False, mix=False):
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
    cfg = json.loads((out / "config.json").read_text())
    cfg_changed = False
    if tensor_kind("lm_head.weight", ebits, resident, head_only, eq4k, aq4k, mix) == 'q6k':
        cfg["lm_head_dtype"] = LM_HEAD_DTYPE_Q6K; cfg_changed = True
        print(f"  config.json: lm_head_dtype={LM_HEAD_DTYPE_Q6K} (Q6_K)")
    if eq4k:
        # Same reason as the head: a Q4_K expert is 144 B per 256 values, which matches
        # neither O*I nor O*((I+1)/2), so byte count cannot signal it. The container states
        # it and the loader cross-checks nbytes == O*(I/256)*144.
        cfg["expert_dtype"] = EXPERT_DTYPE_Q4K; cfg_changed = True
        print(f"  config.json: expert_dtype={EXPERT_DTYPE_Q4K} (Q4_K)")
    if aq4k:
        cfg["attn_dtype"] = EXPERT_DTYPE_Q4K; cfg_changed = True
        print(f"  config.json: attn_dtype={EXPERT_DTYPE_Q4K} (Q4_K)")
    if cfg_changed:
        (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        sys.exit(f"no safetensors in {src}")

    nq = {'int4': 0, 'int8': 0, 'q6k': 0, 'q4k': 0}
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
            kind = tensor_kind(name, ebits, resident, head_only, eq4k, aq4k, mix)
            if kind == 'copy':
                outt[name] = t.contiguous()
                continue
            if kind == 'q6k':
                q, s = pack_q6k(t), None     # scales live inside the block: no ".qs" sibling
            elif kind == 'q4k':
                q, s = pack_q4k(t, nstep=nstep), None   # (scale,min) pairs live inside the block
            elif kind == 'int8':
                q, s = quantize_rows_int8(t)
            else:
                q, s = pack_int4(t, rmse=rmse)
            outt[name] = q.contiguous()
            if s is not None:
                outt[name + ".qs"] = s.contiguous()
            nq[kind] += 1
            bytes_src += t.numel() * t.element_size()
            bytes_q += q.numel() * q.element_size() + (s.numel() * 4 if s is not None else 0)
        save_file(outt, str(out_shard))
        print("ok")

    ratio = bytes_q / max(bytes_src, 1) * 100
    print(f"\ndone: {nq['int4']} tensors -> int4, {nq['int8']} -> int8, "
          f"{nq['q6k']} -> Q6_K, {nq['q4k']} -> Q4_K")
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
    _cfg = json.loads((out / "config.json").read_text())
    if _cfg.get("expert_dtype") == EXPERT_DTYPE_Q4K:
        verify_experts_q4k(oidx, sidx, out, samples, oracle_gguf)
    q4_names = [k for k in oidx if (EXPERT_RE.search(k) or ATTN_RE.search(k)) and (k + ".qs") in oidx]
    if not q4_names:
        if _cfg.get("expert_dtype") == EXPERT_DTYPE_Q4K:
            q4_names = []          # Q4_K experts carry no .qs; attention may still be int4
        else:
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
    ap.add_argument("--experts-q4k", action="store_true",
                    help="T5f: quantize experts to Q4_K (affine, 4.5 bit/w) instead of "
                         "symmetric per-row int4 (4.0 bit/w)")
    ap.add_argument("--oracle-mix", action="store_true",
                    help="T5f move 3: put down_proj and v_proj in Q6_K on the same 24 layers the\n                          oracle GGUF does (layer list read from the file, not derived)")
    ap.add_argument("--attn-q4k", action="store_true",
                    help="T5f move 2: attention q/k/v/o to Q4_K as well (all have I%256==0)")
    ap.add_argument("--q4k-nstep", type=int, default=12,
                    help="candidate scales swept by the Q4_K weighted (scale,min) search. "
                         "Measured on a real-sized expert: 0 -> relL1 0.078564, 12 -> 0.075690, "
                         "20 -> 0.075023 (the reference uses 20); 12 keeps 99%% of the gain at "
                         "60%% of the conversion time")
    ap.add_argument("--int4-rmse", action="store_true",
                    help="T5f move 0: pick the int4 per-row scale by rmse search instead of "
                         "amax/7. Bytes and format unchanged, only the scale is better chosen")
    ap.add_argument("--oracle-gguf", metavar="GGUF",
                    help="Q4_K_M gguf whose output.weight is Q6_K; cross-checks the head in --verify")
    a = ap.parse_args()
    if a.verify:
        verify(a.verify, a.model, a.samples, a.oracle_gguf)
    elif a.out:
        convert(a.model, a.out, a.ebits, resident=not a.experts_only, head_only=a.head_only,
                eq4k=a.experts_q4k, rmse=a.int4_rmse, nstep=a.q4k_nstep, aq4k=a.attn_q4k, mix=a.oracle_mix)
    else:
        ap.error("need --out (convert) or --verify CONTAINER")


if __name__ == "__main__":
    main()
