/* qwen3moe.c — Colibri front-end for Qwen3-MoE (Arianna body = Qwen3-30B-A3B-Base).
 *
 * GRAFT of three validated sources:
 *   - forward math from the T2 parity gate qwen3moe_p0.c (GQA + per-head q/k-norm BEFORE
 *     rotate_half RoPE + softmax top-k router with norm_topk + per-expert SwiGLU),
 *   - streaming scaffold from olmoe.c (resident dense f32, per-layer LRU expert cache,
 *     persistent KV cache, ref.json harness),
 *   - int4 fused-decode dot matmul_i4 byte-exact from glm.c (2 nibbles/byte, per-row scale,
 *     decode happens INSIDE the dot — the expert stays int4 in the cache, never materialized
 *     to an f32 temp per token; that materialize is exactly what made an f16 path 8x slower),
 *   - Q6_K lm_head through the VENDORED notorch kernel nt_qmatvec (GGUF dtype 14, under
 *     vendor/notorch — the build never reaches into a sibling checkout). 210 B / 256 values
 *     with sub-scales inside the block: 0.82 B/weight against int8's 1.0, i.e. 311MB -> 255MB
 *     off the per-token stream. The head is flagged by config "lm_head_dtype": 14 and NEVER
 *     inferred from byte count (O*I / O*((I+1)/2) cannot express Q6_K). nt_qmatvec_i8 carries
 *     no Q6_K (notorch.c:5393 is dtype 2/8 only), so the head's idot path is deliberately
 *     gone: f32 activation, exact dequant-in-register.
 *
 * Container path is chosen by the container, not a flag: an expert stored U8 with a sibling
 * <name>.qs is int4 (matmul_i4); a plain F32/BF16 expert is loaded full precision (matmul).
 * So one binary runs the tiny f32 checkpoint (wiring parity vs qwen3moe_p0.c's ref.json) and
 * the int4 30B container (production breath).
 *
 * Accumulator: -DACC_DOUBLE makes dense matmul double-accumulate (numerically identical to
 * qwen3moe_p0.c) for the tiny wiring gate — any mismatch there is a real wiring bug, not float
 * noise. Default is float + AVX2 (dense too: lm_head is 311M MAC/token) for the 30B tempo gate.
 * matmul_i4 is always float+SIMD (quant already moves argmaxes; exact-parity lives in the f32 path).
 *
 * Build (prod, polygon): make qwen3moe        (-O3 -march=native -fopenmp, links vendor/notorch)
 * Build (wiring parity):  make qwen3moe_p      (-O2 -DACC_DOUBLE, single-thread; the omp pragmas
 *   are deliberately ignored there, hence -Wno-unknown-pragmas — determinism over threads)
 * Run:  SNAP=<dir> ./qwen3moe ref  <ref.json> [cap]   — teacher-forcing + greedy vs reference
 *       SNAP=<dir> ./qwen3moe bench <ntok>     [cap]   — warm decode tok/s (tempo gate)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif
#include "json.h"
#include "st.h"
#include "notorch.h"      /* vendored: nt_qmatvec packed Q6_K */

#define GGUF_Q6_K 14      /* GGUF type code; the container states it in config.json */

#ifdef ACC_DOUBLE
typedef double acc_t;
#else
typedef float acc_t;
#endif

/* ---------- config (real Qwen3MoE schema, with 5.13.1-schema fallbacks — same as T2) ---------- */
typedef struct {
    int hidden, n_layers, n_heads, n_kv_heads, head_dim, n_experts, topk, moe_inter, vocab, norm_topk;
    int lm_head_dtype;   /* 0 = infer from the container; 14 = Q6_K blob */
    float eps; double theta;
} Cfg;

static double jget(jval *r, const char *k, double dflt){
    if(!r) return dflt;
    jval *v = json_get(r, k); return v && v->t==J_NUM ? v->num : dflt;
}
static void load_cfg(Cfg *c, const char *dir){
    char path[1024]; snprintf(path, sizeof path, "%s/config.json", dir);
    FILE *f=fopen(path,"rb"); if(!f){ perror(path); exit(1); }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf=malloc(n+1); if(fread(buf,1,n,f)!=(size_t)n){} buf[n]=0; fclose(f);
    char *ar=NULL; jval *r=json_parse(buf,&ar);
    c->hidden=(int)jget(r,"hidden_size",0); c->n_layers=(int)jget(r,"num_hidden_layers",0);
    c->n_heads=(int)jget(r,"num_attention_heads",0); c->n_kv_heads=(int)jget(r,"num_key_value_heads",0);
    c->head_dim=(int)jget(r,"head_dim", c->n_heads? (double)c->hidden/c->n_heads : 0);
    c->n_experts=(int)jget(r,"num_local_experts",(double)(int)jget(r,"num_experts",0));
    c->topk=(int)jget(r,"num_experts_per_tok",0);
    c->moe_inter=(int)jget(r,"moe_intermediate_size",0); c->vocab=(int)jget(r,"vocab_size",0);
    c->eps=(float)jget(r,"rms_norm_eps",1e-6);
    c->theta=jget(r,"rope_theta",0.0);
    if(c->theta==0.0){ jval *rp=json_get(r,"rope_parameters"); c->theta=jget(rp,"rope_theta",10000.0); }
    jval *nt=json_get(r,"norm_topk_prob"); c->norm_topk=(nt&&nt->t==J_BOOL)?nt->boolean:1;
    c->lm_head_dtype=(int)jget(r,"lm_head_dtype",0);
    free(ar); free(buf);
}

/* ---------- kernels ---------- */
#ifdef __AVX2__
static inline float hsum256(__m256 v){
    __m128 lo=_mm256_castps256_ps128(v), hi=_mm256_extractf128_ps(v,1);
    lo=_mm_add_ps(lo,hi);
    __m128 sh=_mm_movehl_ps(lo,lo); lo=_mm_add_ps(lo,sh);
    sh=_mm_shuffle_ps(lo,lo,0x1);  lo=_mm_add_ss(lo,sh);
    return _mm_cvtss_f32(lo);
}
#endif

/* y[S,O] = x[S,I] @ W[O,I]^T (torch nn.Linear, W row-major). acc_t = double (parity) / float (prod). */
static void matmul(float *y, const float *x, const float *W, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const float *w=W+(int64_t)o*I;
        for(int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; acc_t a=0; int i=0;
#if defined(__AVX2__) && !defined(ACC_DOUBLE)
            __m256 acc=_mm256_setzero_ps();
            for(;i+8<=I;i+=8) acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i),_mm256_loadu_ps(w+i),acc);
            a=hsum256(acc);
#endif
            for(;i<I;i++) a+=(acc_t)xs[i]*w[i];
            y[(int64_t)s*O+o]=(float)a; } }
}

/* y[O] = x[I] @ W_int4[O,I]^T, packed 2 nibbles/byte (low=even col, high=odd col), + scale[O].
 * Fused decode: nibbles are unpacked INSIDE the dot. Byte-exact to glm.c matmul_i4 (glm.c:341-380). */
static void matmul_i4(float *y, const float *x, const uint8_t *q4, const float *scale, int I, int O){
    int rb=(I+1)/2;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const uint8_t *w=q4+(int64_t)o*rb; float sc=scale[o];
        float a=0; int i=0;
#ifdef __AVX2__
        const __m128i m4=_mm_set1_epi8(0x0F); const __m256i b8=_mm256_set1_epi32(8);
        __m256 acc=_mm256_setzero_ps();
        for(;i+16<=I;i+=16){
            __m128i by=_mm_loadl_epi64((const __m128i*)(w+(i>>1)));   /* 8 byte = 16 nibble */
            __m128i lo=_mm_and_si128(by,m4), hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
            __m128i nib=_mm_unpacklo_epi8(lo,hi);                     /* nibble in column order */
            __m256 w0=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib),b8));
            __m256 w1=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(nib,8)),b8));
            acc=_mm256_fmadd_ps(_mm256_loadu_ps(x+i),   w0, acc);
            acc=_mm256_fmadd_ps(_mm256_loadu_ps(x+i+8), w1, acc);
        }
        a=hsum256(acc);
#endif
        for(;i+1<I;i+=2){ uint8_t byte=w[i>>1]; int lo=(int)(byte&0xF)-8, hi=(int)(byte>>4)-8;
            a += x[i]*(float)lo + x[i+1]*(float)hi; }
        if(i<I){ uint8_t byte=w[i>>1]; int lo=(int)(byte&0xF)-8; a += x[i]*(float)lo; }
        y[o]=a*sc;
    }
}

/* y[S,O] = x[S,I] @ W_int8[O,I]^T + scale[O], dequant-on-use. Byte-exact to glm.c matmul_q (322-338). */
static void matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const int8_t *w=q+(int64_t)o*I; float sc=scale[o];
        for(int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0; int i=0;
#ifdef __AVX2__
            __m256 acc=_mm256_setzero_ps();
            for(;i+8<=I;i+=8){ __m256i wi=_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(w+i)));
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i), _mm256_cvtepi32_ps(wi), acc); }
            a=hsum256(acc);
#endif
            for(;i<I;i++) a+=xs[i]*(float)w[i]; y[(int64_t)s*O+o]=a*sc; } }
}

/* ---------- IDOT: int8-activation integer dot (glm.c 509-722), AVX2 + scalar ---------- */
/* quantize one activation row [I] -> int8 + per-vector scale (glm.c qrow_i8 509-514) */
static inline float qrow_i8(const float *x, int8_t *q, int I){
    float amax=0; for(int i=0;i<I;i++){ float a=fabsf(x[i]); if(a>amax)amax=a; }
    float s=amax/127.f; if(s<1e-12f) s=1e-12f; float inv=1.f/s;
    for(int i=0;i<I;i++) q[i]=(int8_t)lrintf(x[i]*inv);
    return s;
}
#ifdef __AVX2__
static inline int hsum256_i32(__m256i v){
    __m128i lo=_mm256_castsi256_si128(v), hi=_mm256_extracti128_si256(v,1);
    lo=_mm_add_epi32(lo,hi); lo=_mm_hadd_epi32(lo,lo); lo=_mm_hadd_epi32(lo,lo);
    return _mm_cvtsi128_si32(lo);
}
#endif
/* dot int8*int8, sign trick |w|*sign(w,x) (glm.c dot_i8i8 530-566) */
static inline int32_t dot_i8i8(const int8_t *w, const int8_t *x, int I){
    int32_t sum=0; int i=0;
#ifdef __AVX2__
    __m256i acc=_mm256_setzero_si256(); const __m256i ones=_mm256_set1_epi16(1);
    for(;i+32<=I;i+=32){
        __m256i wv=_mm256_loadu_si256((const __m256i*)(w+i));
        __m256i xv=_mm256_loadu_si256((const __m256i*)(x+i));
        __m256i p=_mm256_maddubs_epi16(_mm256_sign_epi8(wv,wv),_mm256_sign_epi8(xv,wv));
        acc=_mm256_add_epi32(acc,_mm256_madd_epi16(p,ones));
    }
    sum=hsum256_i32(acc);
#endif
    for(;i<I;i++) sum+=(int32_t)w[i]*x[i];
    return sum;
}
/* dot int4(packed)*int8: nibble -> int8 [-8,7] then same trick (glm.c dot_i4i8 605-708) */
static inline int32_t dot_i4i8(const uint8_t *w4, const int8_t *x, int I){
    int32_t sum=0; int i=0;
#ifdef __AVX2__
    const __m128i m4=_mm_set1_epi8(0x0F); const __m256i b8=_mm256_set1_epi8(8);
    const __m256i ones=_mm256_set1_epi16(1);
    __m256i acc=_mm256_setzero_si256();
    for(;i+32<=I;i+=32){
        __m128i by=_mm_loadu_si128((const __m128i*)(w4+(i>>1)));      /* 16 byte = 32 nibble */
        __m128i lo=_mm_and_si128(by,m4), hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
        __m128i n0=_mm_unpacklo_epi8(lo,hi), n1=_mm_unpackhi_epi8(lo,hi);
        __m256i wv=_mm256_sub_epi8(_mm256_set_m128i(n1,n0),b8);
        __m256i xv=_mm256_loadu_si256((const __m256i*)(x+i));
        __m256i p=_mm256_maddubs_epi16(_mm256_sign_epi8(wv,wv),_mm256_sign_epi8(xv,wv));
        acc=_mm256_add_epi32(acc,_mm256_madd_epi16(p,ones));
    }
    sum=hsum256_i32(acc);
#endif
    for(;i+1<I;i+=2){ uint8_t b=w4[i>>1]; sum+=((int)(b&0xF)-8)*x[i]+((int)(b>>4)-8)*x[i+1]; }
    if(i<I){ uint8_t b=w4[i>>1]; sum+=((int)(b&0xF)-8)*x[i]; }
    return sum;
}
static void matmul_q_idot(float *y, const int8_t *xq, const float *sx, const int8_t *q,
                          const float *scale, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const int8_t *w=q+(int64_t)o*I; float sc=scale[o];
        for(int s=0;s<S;s++) y[(int64_t)s*O+o]=(float)dot_i8i8(w,xq+(int64_t)s*I,I)*sc*sx[s]; }
}
static void matmul_i4_idot(float *y, const int8_t *xq, const float *sx, const uint8_t *q4,
                           const float *scale, int S, int I, int O){
    int rb=(I+1)/2;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const uint8_t *w=q4+(int64_t)o*rb; float sc=scale[o];
        for(int s=0;s<S;s++) y[(int64_t)s*O+o]=(float)dot_i4i8(w,xq+(int64_t)s*I,I)*sc*sx[s]; }
}

/* idot policy (glm.c): int8 IDOT always wins; int4 IDOT gated S>=g_i4s (AVX2 no-VNNI author
 * measured S=1 doesn't pay -> g_i4s=2); attention projections stay EXACT (allow_idot=0,
 * int8-activation quant costs ~+12% perplexity on q/k/v/o). Env: IDOT, I4S, EXPERT_IDOT. */
static double now_s(void);
/* T5d step-0 profile: per-component wall time, direct measurement (every component
 * brackets its own region), printed when PROFILE=1. clock_gettime is ~25ns and there are
 * ~10 samples per layer per token, i.e. ~12us against a ~159ms token — under 0.01%, and
 * the PROFILE=0 tempo is re-measured to prove it. Counters are zeroed after warmup so
 * they describe steady-state decode, not prefill. */
static double p_qkv, p_rope, p_score, p_oproj, p_router, p_expmm, p_expload, p_head, p_norm;
static double p_e_gu, p_e_act, p_e_dn, p_e_red;   /* inside expert_mm (move 6 step 0) */
static int g_profile=0;
#define PT_T0 double _pt0 = g_profile ? now_s() : 0.0
#define PT_ADD(acc) do{ if(g_profile) (acc) += now_s()-_pt0; }while(0)

/* T5d move 1. The expert loader inherited POSIX_FADV_DONTNEED from the olmoe scaffold,
 * where dropping pages after each read kept a big container from evicting everything.
 * Here it is pure damage: the profile puts 52.5 ms/tok (33% of the token!) in expert_load
 * at hit 96.7%, i.e. ~12.7 misses x 2.36MB = ~30MB served at ~0.57 GB/s — disk speed, not
 * memory. Experts are 14.6GB against 31GB of RAM, so they belong in the page cache.
 * EXPERT_DROP=1 restores the old behaviour for a memory-tight box. */
static int g_expert_drop=0;
/* T5d move 2. Attention projections were exact on an inherited claim ("int8-act quant
 * costs ~+12% ppl", carried over from glm.c) that was never measured on THIS model. The
 * profile puts qkv 17.4 + o_proj 12.9 = 30.3 ms/tok there, the second-largest block after
 * the experts. ATTN_IDOT=1 turns on int8-activation integer dot for q/k/v/o; the isolation
 * probes below decide whether it stays on. */
static int g_attn_idot=0;
/* T5d move 5. The Q6_K head ran through nt_qmatvec (f32 activation, exact): 27.42 ms/tok
 * at 9.3 GB/s, the worst efficiency in the decode. nt_qmatvec_i8 now carries dtype 14, so
 * HEAD_IDOT=1 routes the head through the int8-activation integer dot instead. This DOES
 * change the head's arithmetic (the activation is quantized per 32), unlike the fused
 * moves — so it ships behind a flag and the tiny pair is read flip-by-flip. */
static int g_head_idot=0;
static int g_idot=1, g_i4s=2, g_expert_idot=1;   /* expert idot default ON: measured +21% at S=1
                                                    on polygon AVX2 (5.88->7.13), unlike glm's S>=2
                                                    default; EXPERT_IDOT=0 restores exact experts. */

/* resident dense weight: f32, or pre-quantized int4/int8 (packed bytes + per-row F32 scale) */
typedef struct { int kind; float *f; uint8_t *q; float *s; } QW;   /* kind: 0=f32, 4=int4, 8=int8 */
static void qmatmul_ex(float *y, const float *x, const QW *w, int S, int I, int O, int allow_idot){
    if(w->kind==0){ matmul(y,x,w->f,S,I,O); return; }
    if(w->kind==GGUF_Q6_K){
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I; float *ys=y+(int64_t)s*O;
            int rc = (allow_idot && g_head_idot)
                   ? nt_qmatvec_i8(ys, w->q, GGUF_Q6_K, xs, O, I)   /* int8 act, approximate */
                   : nt_qmatvec   (ys, w->q, GGUF_Q6_K, xs, O, I);  /* f32 act, exact */
            if(rc!=0){ fprintf(stderr,"nt_qmatvec%s: no Q6_K kernel for k=%d\n",
                               g_head_idot?"_i8":"", I); exit(1); }
        }
        return;
    }
    /* allow_idot: 0 = exact, 1 = idot if the int4 batch threshold is met, 2 = force idot
     * (decode is S=1, below g_i4s, but the expert path measured +21% there, so attention
     * gets the same right to be measured rather than assumed). */
    if(allow_idot && g_idot && (w->kind==8 || (w->kind==4 && (allow_idot>1 || S>=g_i4s)))){
        int8_t *xq=malloc((size_t)S*I); float *sx=malloc((size_t)S*sizeof(float));
        for(int s=0;s<S;s++) sx[s]=qrow_i8(x+(int64_t)s*I, xq+(int64_t)s*I, I);
        if(w->kind==8) matmul_q_idot(y,xq,sx,(const int8_t*)w->q,w->s,S,I,O);
        else           matmul_i4_idot(y,xq,sx,w->q,w->s,S,I,O);
        free(xq); free(sx); return;
    }
    if(w->kind==8) matmul_q(y,x,(const int8_t*)w->q,w->s,S,I,O);
    else           for(int s=0;s<S;s++) matmul_i4(y+(int64_t)s*O, x+(int64_t)s*I, w->q, w->s, I, O);
}
static void qmatmul(float *y, const float *x, const QW *w, int S, int I, int O){ qmatmul_ex(y,x,w,S,I,O,1); }

static void rmsnorm(float *out, const float *x, const float *w, int n, float eps){
    double v=0; for(int i=0;i<n;i++) v+=(double)x[i]*x[i]; v/=n;
    float r=(float)(1.0/sqrt(v+eps));
    for(int i=0;i<n;i++) out[i]=w[i]*(x[i]*r);
}
static void softmax(float *x, int n){
    float m=-1e30f; for(int i=0;i<n;i++) if(x[i]>m)m=x[i];
    double s=0; for(int i=0;i<n;i++){ x[i]=expf(x[i]-m); s+=x[i]; }
    for(int i=0;i<n;i++) x[i]/=(float)s;
}
static float siluf(float x){ return x/(1.0f+expf(-x)); }
/* rotate_half RoPE on one head vector (dim=hd); cosv/sinv have hd/2 entries. */
static void rope(float *v, const double *cosv, const double *sinv, int hd){
    int half=hd/2;
    for(int i=0;i<half;i++){ float a=v[i], b=v[i+half]; double c=cosv[i], s=sinv[i];
        v[i]=(float)(a*c - b*s); v[i+half]=(float)(b*c + a*s); }
}

/* ---------- model ---------- */
typedef struct { float *in_ln, *post_ln, *qn, *kn; QW q, k, v, o, gate; } Layer;

/* expert LRU slot: int4 (qg/qu/qd packed + sg/su/sd scales) OR f32 (fg/fu/fd) per m->e_int4 */
typedef struct { int eid; uint8_t *qg,*qu,*qd; float *sg,*su,*sd; float *fg,*fu,*fd; uint64_t used; } Slot;
typedef struct { Slot *slots; int n, cap; } LCache;

typedef struct {
    Cfg c;
    shards S;
    int e_int4;               /* 1: experts int4 packed (+.qs); 0: experts f32/bf16 full precision */
    float *embed, *final_ln; QW lm_head;
    Layer *L;
    LCache *cache;
    uint64_t clock, hits, miss;
    float **K, **V; int max_t;   /* KV cache: [n_layers][KV * max_t * hd] */
    double *inv;                 /* rope inv_freq[head_dim/2] */
    double dense_load_s;
} Model;

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
#if defined(__APPLE__)
static double rss_gb(void){ struct rusage r; getrusage(RUSAGE_SELF,&r); return r.ru_maxrss/(1024.0*1024.0*1024.0); }
#else
static double rss_gb(void){ struct rusage r; getrusage(RUSAGE_SELF,&r); return r.ru_maxrss/(1024.0*1024.0); }
#endif
static float *falloc(int64_t n){ float *p=malloc(n*sizeof(float)); if(!p){fprintf(stderr,"OOM %ld\n",(long)n);exit(1);} return p; }

/* dense weight -> f32 resident (BF16/F16/F32 -> f32, upcast ONCE at load, never per-matmul) */
static float *load_t(Model *m, const char *name){
    int64_t n=st_numel(&m->S,name); if(n<0){ fprintf(stderr,"missing %s\n",name); exit(1); }
    float *p=falloc(n); st_read_f32(&m->S,name,p,0); return p;
}

/* resident matmul weight [O,I] -> QW. Container decides: U8/I8 with a .qs sibling is pre-quantized
 * (int4 iff nbytes==O*((I+1)/2), else int8); otherwise full precision f32 (BF16/F16/F32 upcast once). */
static void load_qw(Model *m, QW *w, const char *name, int O, int I){
    memset(w,0,sizeof(*w));
    char qs[300]; snprintf(qs,sizeof qs,"%s.qs",name);
    st_tensor *t=st_find(&m->S,name);
    if(t && t->dtype==3 && st_numel(&m->S,qs)>0){
        int64_t nb=st_nbytes(&m->S,name);
        w->q=malloc(nb); st_read_raw(&m->S,name,w->q,0);
        w->s=falloc(O); st_read_f32(&m->S,qs,w->s,0);
        if(nb==(int64_t)O*((I+1)/2)) w->kind=4;
        else if(nb==(int64_t)O*I)    w->kind=8;
        else { fprintf(stderr,"%s: qbytes %ld unexpected (O=%d I=%d)\n",name,(long)nb,O,I); exit(1); }
    } else {
        int64_t n=st_numel(&m->S,name); if(n<0){ fprintf(stderr,"missing %s\n",name); exit(1); }
        w->kind=0; w->f=falloc(n); st_read_f32(&m->S,name,w->f,0);
    }
}

/* Q6_K head: an opaque U8 blob with NO .qs sibling — the super-scale (f16) and the 16 int8
 * sub-scales live inside each 210-byte block. Byte count cannot signal Q6_K (the container's
 * int8/int4 inference keys off O*I / O*((I+1)/2)), so config.json states "lm_head_dtype": 14
 * and the ONLY structural invariant left is nbytes == O*(I/256)*210. Every deviation is fatal
 * here: a truncated shard, a dim mismatch, a stray .qs from an older int8 container would all
 * otherwise decode into plausible-looking garbage logits. */
static void load_q6_head(Model *m, QW *w, const char *name, int O, int I){
    memset(w,0,sizeof(*w));
    char qs[300]; snprintf(qs,sizeof qs,"%s.qs",name);
    st_tensor *t=st_find(&m->S,name);
    if(!t){ fprintf(stderr,"%s: missing, but config says lm_head_dtype=%d\n",name,GGUF_Q6_K); exit(1); }
    if(t->dtype!=3){ fprintf(stderr,"%s: dtype %d — a Q6_K head must be stored U8\n",name,t->dtype); exit(1); }
    if(I%256){ fprintf(stderr,"%s: I=%d is not a multiple of 256 — Q6_K cannot represent it\n",name,I); exit(1); }
    if(st_numel(&m->S,qs)>0){ fprintf(stderr,"%s: unexpected .qs sibling on a Q6_K head\n",name); exit(1); }
    int64_t want=(int64_t)O*(I/256)*210, nb=st_nbytes(&m->S,name);
    if(nb!=want){ fprintf(stderr,"%s: Q6_K blob is %lld B, expected O*(I/256)*210 = %lld B "
                                 "(truncated shard or wrong dims)\n",name,(long long)nb,(long long)want); exit(1); }
    w->kind=GGUF_Q6_K; w->q=malloc(nb);
    if(!w->q){ fprintf(stderr,"%s: OOM %lld B\n",name,(long long)nb); exit(1); }
    st_read_raw(&m->S,name,w->q,0);
}

static void model_init(Model *m, const char *snap, int cap){
    memset(m,0,sizeof(*m));
    load_cfg(&m->c, snap);
    st_init(&m->S, snap);
    Cfg *c=&m->c;
    /* detect expert storage from the container: int4 iff experts.0 is U8/I8 with a .qs sibling */
    { char n0[256], qs[300];
      snprintf(n0,sizeof n0,"model.layers.0.mlp.experts.0.gate_proj.weight");
      snprintf(qs,sizeof qs,"%s.qs",n0);
      st_tensor *t=st_find(&m->S,n0);
      m->e_int4 = (t && t->dtype==3 && st_numel(&m->S,qs)>0);
      if(m->e_int4){                              /* confirm byte-count is int4, not int8 */
        int64_t nb=st_nbytes(&m->S,n0), O=c->moe_inter, I=c->hidden;
        if(nb!=O*((I+1)/2)){ fprintf(stderr,"expert bytes %ld != int4 %ld (int8? unsupported)\n",(long)nb,(long)(O*((I+1)/2))); exit(1); }
      }
    }
    int half=c->head_dim/2; m->inv=malloc(half*sizeof(double));
    for(int i=0;i<half;i++) m->inv[i]=1.0/pow(c->theta,(double)(2*i)/c->head_dim);

    double t0=now_s();
    int H=c->n_heads, KV=c->n_kv_heads, hd=c->head_dim, D=c->hidden;
    m->embed   = load_t(m,"model.embed_tokens.weight");
    m->final_ln= load_t(m,"model.norm.weight");
    if(c->lm_head_dtype==GGUF_Q6_K) load_q6_head(m,&m->lm_head,"lm_head.weight", c->vocab, D);
    else                            load_qw(m,&m->lm_head,"lm_head.weight", c->vocab, D);
    m->L=calloc(c->n_layers,sizeof(Layer));
    char nm[256];
    for(int l=0;l<c->n_layers;l++){ Layer *L=&m->L[l];
        #define LN(field,suffix) snprintf(nm,sizeof nm,"model.layers.%d." suffix,l); L->field=load_t(m,nm)
        LN(in_ln,"input_layernorm.weight"); LN(post_ln,"post_attention_layernorm.weight");
        LN(qn,"self_attn.q_norm.weight"); LN(kn,"self_attn.k_norm.weight");
        #undef LN
        #define LQ(field,suffix,OO,II) snprintf(nm,sizeof nm,"model.layers.%d." suffix,l); load_qw(m,&L->field,nm,OO,II)
        LQ(q,"self_attn.q_proj.weight", H*hd, D); LQ(k,"self_attn.k_proj.weight", KV*hd, D);
        LQ(v,"self_attn.v_proj.weight", KV*hd, D); LQ(o,"self_attn.o_proj.weight", D, H*hd);
        LQ(gate,"mlp.gate.weight", c->n_experts, D);
        #undef LQ
    }
    m->cache=calloc(c->n_layers,sizeof(LCache));
    for(int l=0;l<c->n_layers;l++){ m->cache[l].cap=cap; m->cache[l].slots=calloc(cap,sizeof(Slot)); }
    m->dense_load_s=now_s()-t0;
}

/* load one expert's gate/up/down into a slot (int4 raw+scale, or f32 full precision) */
static void load_expert(Model *m, int layer, int eid, Slot *s){
    Cfg *c=&m->c; char nm[256], qs[300];
    const char *suf[3]={"gate_proj","up_proj","down_proj"};
    for(int t=0;t<3;t++){
        snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.%d.%s.weight",layer,eid,suf[t]);
        int O=(t<2)?c->moe_inter:c->hidden, I=(t<2)?c->hidden:c->moe_inter;
        if(m->e_int4){
            uint8_t *q=(t==0)?s->qg:(t==1)?s->qu:s->qd;
            float   *sc=(t==0)?s->sg:(t==1)?s->su:s->sd;
            snprintf(qs,sizeof qs,"%s.qs",nm);
            st_read_raw(&m->S,nm,q,g_expert_drop);   /* packed int4 bytes, verbatim */
            st_read_f32(&m->S,qs,sc,g_expert_drop);  /* per-row F32 scales */
        } else {
            float *f=(t==0)?s->fg:(t==1)?s->fu:s->fd;
            st_read_f32(&m->S,nm,f,g_expert_drop);
            (void)O;(void)I;
        }
    }
}

static Slot *expert_get(Model *m, int layer, int eid){
    LCache *lc=&m->cache[layer];
    for(int i=0;i<lc->n;i++) if(lc->slots[i].eid==eid){ m->hits++; lc->slots[i].used=++m->clock; return &lc->slots[i]; }
    m->miss++;
    Cfg *c=&m->c;
    int64_t Ig=(int64_t)c->moe_inter*c->hidden, Id=(int64_t)c->hidden*c->moe_inter;   /* elements */
    int64_t rbg=(int64_t)c->moe_inter*((c->hidden+1)/2), rbd=(int64_t)c->hidden*((c->moe_inter+1)/2); /* int4 bytes */
    Slot *s;
    if(lc->n<lc->cap){
        s=&lc->slots[lc->n++];
        if(m->e_int4){
            /* T5d move 6b. Expert rows are 1024 B (gate/up) and 384 B (down) — both exact
             * multiples of the 64 B cache line, so a 64-B-aligned base makes EVERY row
             * line-aligned. Plain malloc gives 16 B: three quarters of the slots start
             * mid-line, and each row then touches one cache line more than it needs across
             * 906 MB/tok. Sizes are multiples of 64 by construction (moe_inter*hidden/2),
             * which aligned_alloc requires. Pure addressing: not one byte of arithmetic. */
            s->qg=aligned_alloc(64,rbg); s->qu=aligned_alloc(64,rbg); s->qd=aligned_alloc(64,rbd);
            if(!s->qg||!s->qu||!s->qd){ fprintf(stderr,"expert slot alloc failed\n"); exit(1); }
            s->sg=falloc(c->moe_inter); s->su=falloc(c->moe_inter); s->sd=falloc(c->hidden);
        } else { s->fg=falloc(Ig); s->fu=falloc(Ig); s->fd=falloc(Id); }
    } else { int lru=0; for(int i=1;i<lc->n;i++) if(lc->slots[i].used<lc->slots[lru].used) lru=i; s=&lc->slots[lru]; }
    load_expert(m,layer,eid,s);
    s->eid=eid; s->used=++m->clock;
    return s;
}

/* expert SwiGLU on one token x[D] -> accumulate into moe[D] with weight wgt.
 * idot!=0: int8-activation dot (x pre-quantized to xq/sx by the caller, shared across the
 * token's top-k experts; gg re-quantized per expert into gq). Else exact int4 / f32. */
static void expert_apply(Model *m, Slot *s, const float *x, const int8_t *xq, float sx,
                         float wgt, float *moe, float *gg, float *uu, float *dn, int8_t *gq, int idot){
    Cfg *c=&m->c; int D=c->hidden, I=c->moe_inter;
    if(m->e_int4 && idot){
        matmul_i4_idot(gg, xq, &sx, s->qg, s->sg, 1, D, I);
        matmul_i4_idot(uu, xq, &sx, s->qu, s->su, 1, D, I);
        for(int i=0;i<I;i++) gg[i]=siluf(gg[i])*uu[i];
        float sgg=qrow_i8(gg, gq, I);
        matmul_i4_idot(dn, gq, &sgg, s->qd, s->sd, 1, I, D);
    } else if(m->e_int4){
        matmul_i4(gg, x, s->qg, s->sg, D, I);
        matmul_i4(uu, x, s->qu, s->su, D, I);
        for(int i=0;i<I;i++) gg[i]=siluf(gg[i])*uu[i];
        matmul_i4(dn, gg, s->qd, s->sd, I, D);
    } else {
        matmul(gg, x, s->fg, 1, D, I);
        matmul(uu, x, s->fu, 1, D, I);
        for(int i=0;i<I;i++) gg[i]=siluf(gg[i])*uu[i];
        matmul(dn, gg, s->fd, 1, I, D);
    }
    for(int i=0;i<D;i++) moe[i]+=wgt*dn[i];
}

/* T5d move 3 — fused top-k expert pass. matmul_i4_idot opened one OpenMP region per matmul,
 * i.e. 3 matmuls x topk 8 x 48 layers = 1152 regions per token; at a few microseconds of
 * fork/join each that is milliseconds of pure overhead on a 99 ms token. Here gate+up for
 * ALL top-k experts run as one flat (expert x row) region and down as a second, so a layer
 * costs 2 regions instead of 24, the int8 activation is quantized once and read by every
 * expert, and row-level load balance is kept (8x1536 and 8x2048 rows over 6 threads).
 * The dot is integer and the reduction keeps kk order, so this must be BIT-IDENTICAL to the
 * unfused path: any token move in the tiny pair is a bug, not quantization noise. */
static void experts_fused(Model *m, Slot **sl, int topk, const int8_t *xq, float sx,
                          const float *wsel, float *moe,
                          float *gga, float *uua, int8_t *gqa, float *dna, float *sga){
    Cfg *c=&m->c; int D=c->hidden, I=c->moe_inter;
    int64_t rbg=(D+1)/2, rbd=(I+1)/2;
    int NG=topk*I;
    double _t0 = g_profile ? now_s() : 0.0;
    #pragma omp parallel for schedule(static)
    for(int t=0;t<2*NG;t++){
        int up = t>=NG, r = up? t-NG : t, kk=r/I, o=r%I;
        Slot *s=sl[kk];
        const uint8_t *w = (up? s->qu : s->qg) + (int64_t)o*rbg;
        float sc = (up? s->su : s->sg)[o];
        (up? uua : gga)[(int64_t)kk*I+o] = (float)dot_i4i8(w, xq, D)*sc*sx;
    }
    double _t1 = g_profile ? now_s() : 0.0;
    /* T5d move 6a. This stretch ran single-threaded while six cores idled: 2.79 ms/tok,
     * and it is dominated by expf inside silu, not by the quantize. SwiGLU is elementwise,
     * so it flattens over topk*I and balances exactly; qrow_i8 needs the whole row, so it
     * stays per-expert. Both keep their internal order -> bit-identical. */
    #pragma omp parallel for schedule(static)
    for(int t=0;t<topk*I;t++) gga[t]=siluf(gga[t])*uua[t];
    #pragma omp parallel for schedule(static)
    for(int kk=0;kk<topk;kk++)
        sga[kk]=qrow_i8(gga+(int64_t)kk*I, gqa+(int64_t)kk*I, I);
    double _t2 = g_profile ? now_s() : 0.0;
    #pragma omp parallel for schedule(static)
    for(int t=0;t<topk*D;t++){
        int kk=t/D, o=t%D; Slot *s=sl[kk];
        dna[(int64_t)kk*D+o] = (float)dot_i4i8(s->qd+(int64_t)o*rbd, gqa+(int64_t)kk*I, I)
                               * s->sd[o] * sga[kk];
    }
    double _t3 = g_profile ? now_s() : 0.0;
    for(int kk=0;kk<topk;kk++){                       /* kk order preserved == unfused path */
        const float *d=dna+(int64_t)kk*D; float wg=wsel[kk];
        for(int i=0;i<D;i++) moe[i]+=wg*d[i];
    }
    if(g_profile){ double _t4=now_s();
        p_e_gu+=_t1-_t0; p_e_act+=_t2-_t1; p_e_dn+=_t3-_t2; p_e_red+=_t4-_t3; }
}

/* T5d move 4 — QK score dot. It was a scalar double chain (a += (double)q[i]*k[i], hd=128):
 * the loop-carried dependency blocks vectorization outright, and the profile charged 8.63
 * ms/tok to scores. Accumulating in DOUBLE lanes (not float) is the deliberate choice: the
 * gate needs 2.3 ms of the 8.63, four partial sums already give that, and the re-order stays
 * at 1e-16 instead of dropping to float's 1e-7 near the softmax input. Still a float re-order
 * by the brief's definition, so the tiny pair is checked flip-by-flip. */
static inline float qk_dot(const float *a, const float *b, int n){
#if defined(__AVX2__) && defined(__FMA__)
    __m256d s0=_mm256_setzero_pd(), s1=_mm256_setzero_pd();
    int i=0;
    for(; i+8<=n; i+=8){
        s0=_mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(a+i)),
                           _mm256_cvtps_pd(_mm_loadu_ps(b+i)), s0);
        s1=_mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(a+i+4)),
                           _mm256_cvtps_pd(_mm_loadu_ps(b+i+4)), s1);
    }
    __m256d s=_mm256_add_pd(s0,s1);
    __m128d lo=_mm256_castpd256_pd128(s), hi=_mm256_extractf128_pd(s,1);
    lo=_mm_add_pd(lo,hi); lo=_mm_add_sd(lo,_mm_unpackhi_pd(lo,lo));
    double acc=_mm_cvtsd_f64(lo);
    for(; i<n; i++) acc+=(double)a[i]*b[i];
    return (float)acc;
#else
    double acc=0; for(int i=0;i<n;i++) acc+=(double)a[i]*b[i]; return (float)acc;
#endif
}

/* T5d move 3b — same trick as move 3, applied to the attention projections. q/k/v share the
 * SAME input row, yet each went through its own qmatmul_ex: three OpenMP regions per layer
 * and, worse, three identical qrow_i8 quantizations of that row (144 redundant passes per
 * token). Here the row is quantized once and q/k/v run as one flat region over
 * H*hd + 2*KV*hd rows. Integer dot, per-row scales unchanged -> bit-identical. Decode-only
 * (S==1) and only when all three are int4; prefill and any other layout keep the old path. */
static void attn_qkv_fused(const QW *wq, const QW *wk, const QW *wv, const int8_t *xq,
                           float sx, float *q, float *k, float *v, int D, int Hhd, int KVhd){
    int64_t rb=(D+1)/2; int n1=Hhd, n2=n1+KVhd, n3=n2+KVhd;
    #pragma omp parallel for schedule(static)
    for(int t=0;t<n3;t++){
        const QW *w; float *dst; int o;
        if(t<n1){ w=wq; dst=q; o=t; }
        else if(t<n2){ w=wk; dst=k; o=t-n1; }
        else        { w=wv; dst=v; o=t-n2; }
        dst[o]=(float)dot_i4i8(w->q+(int64_t)o*rb, xq, D)*w->s[o]*sx;
    }
}

/* attention over new tokens x[S,D]; pos_base = absolute pos of the first new token; out[S,D] */
static void attention(Model *m, Layer *l, int layer, const float *x, int S, int pos_base, float *out){
    Cfg *c=&m->c; int H=c->n_heads, KV=c->n_kv_heads, hd=c->head_dim, D=c->hidden, group=H/KV, half=hd/2;
    float *q=falloc((int64_t)S*H*hd), *k=falloc((int64_t)S*KV*hd), *v=falloc((int64_t)S*KV*hd);
    int ai = g_attn_idot ? 2 : 0;
    { PT_T0;
    if(ai && g_idot && S==1 && l->q.kind==4 && l->k.kind==4 && l->v.kind==4){
        int8_t *xq1=malloc(D); float sx1=qrow_i8(x, xq1, D);   /* quantized ONCE for all three */
        attn_qkv_fused(&l->q,&l->k,&l->v,xq1,sx1,q,k,v,D,H*hd,KV*hd);
        free(xq1);
    } else {
        qmatmul_ex(q, x, &l->q, S, D, H*hd, ai);
        qmatmul_ex(k, x, &l->k, S, D, KV*hd, ai);
        qmatmul_ex(v, x, &l->v, S, D, KV*hd, ai);
    }
    PT_ADD(p_qkv); }
    double _pr0 = g_profile ? now_s() : 0.0;
    double *cosb=malloc((size_t)S*half*sizeof(double)), *sinb=malloc((size_t)S*half*sizeof(double));
    for(int s=0;s<S;s++){ int pos=pos_base+s; for(int i=0;i<half;i++){ double a=pos*m->inv[i]; cosb[s*half+i]=cos(a); sinb[s*half+i]=sin(a); } }
    /* per-head q/k-norm (dim=hd) BEFORE rope; then write k,v into the persistent KV cache */
    for(int s=0;s<S;s++){
        const double *cp=cosb+(size_t)s*half, *sp=sinb+(size_t)s*half;
        for(int h=0;h<H;h++){ float *qh=q+((size_t)s*H+h)*hd; rmsnorm(qh,qh,l->qn,hd,c->eps); rope(qh,cp,sp,hd); }
        for(int kh=0;kh<KV;kh++){ float *kk=k+((size_t)s*KV+kh)*hd; rmsnorm(kk,kk,l->kn,hd,c->eps); rope(kk,cp,sp,hd);
            int t=pos_base+s;
            memcpy(m->K[layer]+((int64_t)kh*m->max_t+t)*hd, kk, hd*sizeof(float));
            memcpy(m->V[layer]+((int64_t)kh*m->max_t+t)*hd, v+((size_t)s*KV+kh)*hd, hd*sizeof(float)); }
    }
    if(g_profile) p_rope += now_s()-_pr0;
    float scale=1.0f/sqrtf((float)hd);
    double _ps0 = g_profile ? now_s() : 0.0;
    #pragma omp parallel for collapse(2) schedule(static)
    for(int h=0;h<H;h++){
        for(int s=0;s<S;s++){
            int kh=h/group, qpos=pos_base+s;
            const float *qh=q+((size_t)s*H+h)*hd;
            float sc[8192];
            for(int t=0;t<=qpos;t++){ const float *kk=m->K[layer]+((int64_t)kh*m->max_t+t)*hd;
                sc[t]=qk_dot(qh,kk,hd)*scale; }
            softmax(sc,qpos+1);
            float *co=out+((size_t)s*H+h)*hd; for(int i=0;i<hd;i++) co[i]=0;
            for(int t=0;t<=qpos;t++){ const float *vv=m->V[layer]+((int64_t)kh*m->max_t+t)*hd;
                float w=sc[t]; for(int i=0;i<hd;i++) co[i]+=w*vv[i]; }
        }
    }
    if(g_profile) p_score += now_s()-_ps0;
    free(q);free(k);free(v);free(cosb);free(sinb);
}

/* one forward step: new tokens ids[S] at pos_base. Fills logits[S*vocab] if want_all else only last row. */
static void step(Model *m, const int *ids, int S, int pos_base, float *logits, int want_all){
    Cfg *c=&m->c; int D=c->hidden, H=c->n_heads, hd=c->head_dim;
    float *x=falloc((int64_t)S*D);
    for(int s=0;s<S;s++) memcpy(x+(size_t)s*D, m->embed+(int64_t)ids[s]*D, D*sizeof(float));
    float *nrm=falloc((int64_t)S*D), *att=falloc((int64_t)S*H*hd), *ao=falloc((int64_t)S*D);
    float *rl=falloc(c->n_experts), *gg=falloc(c->moe_inter), *uu=falloc(c->moe_inter), *dn=falloc(D), *moe=falloc(D);
    int8_t *xq=malloc(D), *gq=malloc(c->moe_inter);
    Slot **sl=malloc((size_t)c->topk*sizeof(Slot*));
    float *gga=falloc((int64_t)c->topk*c->moe_inter), *uua=falloc((int64_t)c->topk*c->moe_inter);
    float *dna=falloc((int64_t)c->topk*D), *sga=falloc(c->topk);
    int8_t *gqa=malloc((size_t)c->topk*c->moe_inter);
    int e_idot = g_idot && g_expert_idot && m->e_int4;
    for(int l=0;l<c->n_layers;l++){ Layer *L=&m->L[l];
        { PT_T0; for(int s=0;s<S;s++) rmsnorm(nrm+(size_t)s*D, x+(size_t)s*D, L->in_ln, D, c->eps); PT_ADD(p_norm); }
        attention(m,L,l,nrm,S,pos_base,att);
        { PT_T0; qmatmul_ex(ao, att, &L->o, S, H*hd, D, g_attn_idot?2:0); PT_ADD(p_oproj); }
        for(int64_t j=0;j<(int64_t)S*D;j++) x[j]+=ao[j];
        for(int s=0;s<S;s++){
            float *xn=nrm+(size_t)s*D; rmsnorm(xn, x+(size_t)s*D, L->post_ln, D, c->eps);
            { PT_T0; qmatmul(rl, xn, &L->gate, 1, D, c->n_experts); softmax(rl, c->n_experts); PT_ADD(p_router); }
            int idx[256]; float wsel[256];
            for(int kk=0;kk<c->topk;kk++){ int best=-1; float bv=-1e30f;
                for(int e=0;e<c->n_experts;e++){ int taken=0; for(int z=0;z<kk;z++) if(idx[z]==e){taken=1;break;}
                    if(!taken && rl[e]>bv){bv=rl[e];best=e;} }
                idx[kk]=best; wsel[kk]=rl[best]; }
            if(c->norm_topk){ float sm=0; for(int kk=0;kk<c->topk;kk++) sm+=wsel[kk]; sm+=1e-20f;
                for(int kk=0;kk<c->topk;kk++) wsel[kk]/=sm; }
            for(int i=0;i<D;i++) moe[i]=0;
            float sx = e_idot ? qrow_i8(xn, xq, D) : 0.f;
            /* LRU lookup first and single-threaded: expert_get mutates clock/used. */
            double _e0 = g_profile ? now_s() : 0.0;
            for(int kk=0;kk<c->topk;kk++) sl[kk]=expert_get(m,l,idx[kk]);
            double _e1 = g_profile ? now_s() : 0.0;
            if(g_profile) p_expload += _e1-_e0;
            if(e_idot) experts_fused(m,sl,c->topk,xq,sx,wsel,moe,gga,uua,gqa,dna,sga);
            else for(int kk=0;kk<c->topk;kk++)
                     expert_apply(m,sl[kk],xn,xq,sx,wsel[kk],moe,gg,uu,dn,gq,0);
            if(g_profile) p_expmm += now_s()-_e1;
            for(int i=0;i<D;i++) x[(size_t)s*D+i]+=moe[i];
        }
    }
    for(int s=0;s<S;s++){
        if(!want_all && s!=S-1) continue;
        float *xn=nrm; rmsnorm(xn, x+(size_t)s*D, m->final_ln, D, c->eps);
        { PT_T0; qmatmul(logits+(size_t)(want_all?s:0)*c->vocab, xn, &m->lm_head, 1, D, c->vocab); PT_ADD(p_head); }
    }
    free(x);free(nrm);free(att);free(ao);free(rl);free(gg);free(uu);free(dn);free(moe);free(xq);free(gq);
    free(sl);free(gga);free(uua);free(dna);free(sga);free(gqa);
}

static void kv_alloc(Model *m, int max_t){
    Cfg *c=&m->c; m->max_t=max_t;
    m->K=calloc(c->n_layers,sizeof(float*)); m->V=calloc(c->n_layers,sizeof(float*));
    for(int l=0;l<c->n_layers;l++){ m->K[l]=falloc((int64_t)c->n_kv_heads*max_t*c->head_dim);
        m->V[l]=falloc((int64_t)c->n_kv_heads*max_t*c->head_dim); }
}
static void kv_free(Model *m){ Cfg *c=&m->c; for(int l=0;l<c->n_layers;l++){ free(m->K[l]); free(m->V[l]); } free(m->K); free(m->V); m->K=m->V=NULL; }

static int argmax(const float *v, int n){ int b=0; for(int i=1;i<n;i++) if(v[i]>v[b])b=i; return b; }
static int *jarr(jval *r, const char *k, int *n){ jval *a=json_get(r,k); *n=a->len; int *o=malloc(a->len*sizeof(int));
    for(int i=0;i<a->len;i++) o[i]=(int)a->kids[i]->num;
    return o; }

/* ---------- ref: teacher-forcing (all positions) + greedy, vs ref.json ---------- */
static int run_ref(Model *m, const char *refpath){
    Cfg *c=&m->c;
    FILE *f=fopen(refpath,"rb"); if(!f){ perror(refpath); return 2; }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf=malloc(n+1); if(fread(buf,1,n,f)!=(size_t)n){} buf[n]=0; fclose(f);
    char *ar=NULL; jval *r=json_parse(buf,&ar);
    int np,nf,ntf; int *prompt=jarr(r,"prompt_ids",&np); int *full=jarr(r,"full_ids",&nf); int *tfp=jarr(r,"tf_pred",&ntf); (void)ntf;

    /* teacher-forcing: one prefill over the full sequence, argmax at every position */
    kv_alloc(m, nf);
    float *lg=falloc((int64_t)nf*c->vocab);
    step(m, full, nf, 0, lg, 1);
    /* Name every divergence with its margin: under quantization a flip is legal, but a
     * near-tie (tiny margin) and a confident disagreement are different animals, and only
     * the margin separates them. */
    int ok_tf=0;
    for(int i=0;i<nf;i++){ const float *row=lg+(size_t)i*c->vocab; int a=argmax(row,c->vocab);
        if(a==tfp[i]){ ok_tf++; continue; }
        fprintf(stderr,"  tf flip @%d: got %d (logit %.6f) vs ref %d (%.6f), margin %.3e\n",
                i,a,(double)row[a],tfp[i],(double)row[tfp[i]],(double)(row[a]-row[tfp[i]]));
    }
    kv_free(m); free(lg);
    fprintf(stderr,"teacher-forcing argmax: %d/%d match tf_pred\n", ok_tf, nf);

    /* greedy: prefill prompt, then decode one token at a time (persistent KV) */
    kv_alloc(m, nf);
    int *gen=malloc(nf*sizeof(int)); memcpy(gen,prompt,np*sizeof(int)); int glen=np;
    float *last=falloc(c->vocab);
    step(m, prompt, np, 0, last, 0);
    while(glen<nf){ int nx=argmax(last,c->vocab);
        if(nx!=full[glen])
            fprintf(stderr,"  greedy flip @%d: got %d (logit %.6f) vs ref %d (%.6f), margin %.3e\n",
                    glen,nx,(double)last[nx],full[glen],(double)last[full[glen]],
                    (double)(last[nx]-last[full[glen]]));
        gen[glen++]=nx; if(glen>=nf) break; int one=nx; step(m,&one,1,glen-1,last,0); }
    int ok_g=0; for(int i=np;i<nf;i++) if(gen[i]==full[i]) ok_g++;
    kv_free(m); free(last); free(gen);
    fprintf(stderr,"greedy: %d/%d match full_ids (generated tail)\n", ok_g, nf-np);

    printf("qwen3moe: tf %d/%d, greedy %d/%d\n", ok_tf,nf, ok_g,nf-np);
    free(buf); free(ar);
    return (ok_tf==nf && ok_g==nf-np) ? 0 : 2;
}

/* ---------- bench: warm decode tok/s (tempo gate, llama-bench class: warmup then steady decode) ---------- */
static void run_bench(Model *m, int ntok){
    Cfg *c=&m->c;
    /* A MoE with an expert LRU has a cold phase llama.cpp does not have (it mmaps the whole
     * file), so 8 tokens is not "warm": at 256 tok the profile still charged 26 ms/tok to
     * first-touch expert reads, and that is prime cost, not steady state. WARMUP runs the
     * LRU hot, then the KV is reset so the measured window starts at position 0 — attention
     * scores are O(T) (8.5 ms at 256, 31.8 at 1024), so measuring at positions 256..512
     * would confound the tempo with context growth. Warm-up run, then generate from an empty
     * context: that is exactly llama-bench's tg256. Default 8 keeps the historical number. */
    int warm=8; { const char *e=getenv("WARMUP"); if(e && atoi(e)>0) warm=atoi(e); }
    int npre=1; { const char *e=getenv("PREFILL"); if(e && atoi(e)>0) npre=atoi(e); }
    int total=ntok; if(npre+total+2>=c->vocab) total=c->vocab-npre-2;
    float *last=falloc(c->vocab);
    double t_warm=0, t_run=0; int cnt=0;

    kv_alloc(m, warm+2);                                   /* phase 1: warm the expert LRU */
    { int seed=1; step(m,&seed,1,0,last,0);
      double tw0=now_s();
      for(int s=0;s<warm;s++){ int nx=argmax(last,c->vocab); step(m,&nx,1,s+1,last,0); }
      t_warm=now_s()-tw0; }
    kv_free(m);
    m->hits=m->miss=0;                                     /* hit% of the measured window */
    p_qkv=p_rope=p_score=p_oproj=p_router=p_expmm=p_expload=p_head=p_norm=0;
    p_e_gu=p_e_act=p_e_dn=p_e_red=0;

    kv_alloc(m, npre+total+2);                             /* phase 2: measured, from pos 0 */
    int *pre=calloc((size_t)npre,sizeof(int));
    if(!pre){ fprintf(stderr,"OOM prefill %d\n",npre); exit(1); }
    for(int i=0;i<npre;i++) pre[i]=(i*7+1)%c->vocab;
    double tp0=now_s(); step(m,pre,npre,0,last,0); double t_pre=now_s()-tp0;
    int *seq=calloc((size_t)(total>0?total:1),sizeof(int));
    if(!seq){ fprintf(stderr,"OOM seq\n"); exit(1); }
    for(int s=0;s<total;s++){ int nx=argmax(last,c->vocab); seq[s]=nx; double t=now_s();
        step(m,&nx,1,npre+s,last,0); t_run+=now_s()-t; cnt++; }
    double tps=cnt? cnt/t_run : 0;
    if(npre>1) printf("bench: prefill %.2f tok/s  (%d tok, %.3fs)\n", npre/t_pre, npre, t_pre);
    { printf("bench: first 16 generated ids:"); for(int i=0;i<16&&i<cnt;i++) printf(" %d",seq[i]);
      printf("\n"); }
    printf("bench: warm decode %.2f tok/s  (%d tok, %.2fs; warmup %d tok %.2fs) | RSS %.2f GB | hit %.1f%%\n",
        tps, cnt, t_run, warm, t_warm, rss_gb(),
        (m->hits+m->miss)? 100.0*m->hits/(m->hits+m->miss):0.0);
    if(g_profile && cnt){
        double k=1e3/cnt, per=t_run*1e3/cnt;
        double sum=p_qkv+p_rope+p_score+p_oproj+p_router+p_expmm+p_expload+p_head+p_norm;
        printf("profile ms/tok: qkv %.2f | rope+kv %.2f | scores %.2f | o_proj %.2f | router %.2f | "
               "expert_mm %.2f | expert_load %.2f | head %.2f | norms %.2f\n",
               p_qkv*k,p_rope*k,p_score*k,p_oproj*k,p_router*k,p_expmm*k,p_expload*k,p_head*k,p_norm*k);
        printf("profile: sum %.2f ms/tok vs measured %.2f ms/tok (unaccounted %.2f ms, %.1f%%)\n",
               sum*k, per, per-sum*k, 100.0*(per-sum*k)/per);
        printf("  inside expert_mm: gate+up %.2f | silu+qrow %.2f | down %.2f | reduce %.2f"
               " (sum %.2f of %.2f)\n", p_e_gu*k, p_e_act*k, p_e_dn*k, p_e_red*k,
               (p_e_gu+p_e_act+p_e_dn+p_e_red)*k, p_expmm*k);
    }
    kv_free(m); free(last); free(pre); free(seq);
}

int main(int argc, char **argv){
    const char *snap=getenv("SNAP");
    if(!snap){ fprintf(stderr,"set SNAP=<container dir>\n"); return 1; }
    { const char *e;
      if((e=getenv("IDOT")))        g_idot=atoi(e);
      if((e=getenv("I4S")))         g_i4s=atoi(e);
      if((e=getenv("EXPERT_IDOT"))) g_expert_idot=atoi(e);
      if((e=getenv("PROFILE")))     g_profile=atoi(e);
      if((e=getenv("EXPERT_DROP"))) g_expert_drop=atoi(e);
      if((e=getenv("ATTN_IDOT")))   g_attn_idot=atoi(e);
      if((e=getenv("HEAD_IDOT")))   g_head_idot=atoi(e); }
    if(argc<3){ fprintf(stderr,"usage: SNAP=<dir> %s ref <ref.json> [cap] | bench <ntok> [cap]\n",argv[0]); return 1; }
    const char *mode=argv[1];
    int cap = argc>3 ? atoi(argv[3]) : 0;

    Cfg probe; load_cfg(&probe, snap);
    if(cap<=0) cap = probe.n_experts;                     /* default: all experts resident/layer */

    Model m; model_init(&m, snap, cap);
    fprintf(stderr,"cfg: hidden=%d L=%d H=%d KV=%d hd=%d E=%d topk=%d moe_inter=%d vocab=%d theta=%g norm_topk=%d experts=%s cap=%d\n",
        m.c.hidden,m.c.n_layers,m.c.n_heads,m.c.n_kv_heads,m.c.head_dim,m.c.n_experts,m.c.topk,m.c.moe_inter,m.c.vocab,m.c.theta,m.c.norm_topk,
        m.e_int4?"int4":"f32", cap);
    fprintf(stderr,"resident dense loaded in %.1fs | RSS %.2f GB\n", m.dense_load_s, rss_gb());

    int rc=0;
    if(!strcmp(mode,"ref"))       rc=run_ref(&m, argv[2]);
    else if(!strcmp(mode,"bench"))run_bench(&m, atoi(argv[2]));
    else { fprintf(stderr,"unknown mode %s\n",mode); rc=1; }
    return rc;
}
