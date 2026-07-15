/* mistral4_p0.c — minimal self-contained f32 forward of Mistral Small 4 (mistral4),
 * for the P0 parity gate of the Colibri->Arianna port.
 *
 * NOT the production engine: all weights resident, naive MLA (no absorption), no
 * streaming/cache/CUDA. Its only job is to prove the arch-branch (MLA + interleaved
 * YaRN rope + llama4 attn-scale + softmax router + 3D experts) reproduces the torch
 * reference token-exact, isolated from infra. Semantics ported verbatim from
 * transformers modeling_mistral4.py (read from source, not guessed).
 *
 * Build:  cc -O2 -o mistral4_p0 mistral4_p0.c -lm
 * Run:    ./mistral4_p0 <mistral4_tiny_dir> <ref_mistral4.json>
 *
 * Checklist (P0 gate): cc 0 warnings | inv_freq[:5]==torch | tf 32/32 | greedy 20/20.
 * Python is nowhere here: this is the pure-C engine side. (Oleg 2026-07-15)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "json.h"
#include "st.h"

typedef struct {
    int hidden, n_layers, n_heads, n_kv_heads, n_experts, topk, moe_inter;
    int q_lora, kv_lora, qk_nope, qk_rope, v_head, qk_head, n_shared, vocab, first_dense;
    float eps, routed_scale;
    double theta, factor, beta_fast, beta_slow, mscale, mscale_all_dim, llama4_beta;
    int orig_max;
    float attn_scaling;   /* mscale-derived cos/sin factor (1.0 here) */
} Cfg;

static double jget(jval *r, const char *k, double dflt){
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
    c->n_experts=(int)jget(r,"n_routed_experts",0); c->topk=(int)jget(r,"num_experts_per_tok",0);
    c->moe_inter=(int)jget(r,"moe_intermediate_size",0); c->q_lora=(int)jget(r,"q_lora_rank",0);
    c->kv_lora=(int)jget(r,"kv_lora_rank",0); c->qk_nope=(int)jget(r,"qk_nope_head_dim",0);
    c->qk_rope=(int)jget(r,"qk_rope_head_dim",0); c->v_head=(int)jget(r,"v_head_dim",0);
    c->n_shared=(int)jget(r,"n_shared_experts",0); c->vocab=(int)jget(r,"vocab_size",0);
    c->first_dense=(int)jget(r,"first_k_dense_replace",0);
    c->eps=(float)jget(r,"rms_norm_eps",1e-5); c->routed_scale=(float)jget(r,"routed_scaling_factor",1.0);
    c->qk_head=c->qk_nope+c->qk_rope;
    jval *rp=json_get(r,"rope_parameters");
    c->theta=jget(rp,"rope_theta",10000.0); c->factor=jget(rp,"factor",1.0);
    c->beta_fast=jget(rp,"beta_fast",32.0); c->beta_slow=jget(rp,"beta_slow",1.0);
    c->mscale=jget(rp,"mscale",1.0); c->mscale_all_dim=jget(rp,"mscale_all_dim",1.0);
    c->llama4_beta=jget(rp,"llama_4_scaling_beta",0.0);
    c->orig_max=(int)jget(rp,"original_max_position_embeddings",0);
    double gm=1.0, gd=1.0;
    if(c->factor>1.0){ gm=0.1*c->mscale*log(c->factor)+1.0; gd=0.1*c->mscale_all_dim*log(c->factor)+1.0; }
    c->attn_scaling=(float)(gm/gd);
    free(ar); free(buf);
}

/* YaRN inv_freq (dim = qk_rope), verbatim _compute_yarn_parameters */
static void yarn_inv_freq(const Cfg *c, double *inv, int half){
    int dim=c->qk_rope; double base=c->theta, factor=c->factor;
    double lo, hi;
    #define FCD(nr) ( (dim*log((double)c->orig_max/((nr)*2.0*M_PI))) / (2.0*log(base)) )
    lo=floor(FCD(c->beta_fast)); hi=ceil(FCD(c->beta_slow));
    if(lo<0) lo=0;
    if(hi>dim-1) hi=dim-1;
    #undef FCD
    for(int i=0;i<half;i++){
        double pf = pow(base, (double)(2*i)/dim);
        double extrap = 1.0/pf, interp = 1.0/(factor*pf);
        double denom = (hi==lo) ? (hi+0.001-lo) : (hi-lo);
        double ramp = (i-lo)/denom; if(ramp<0) ramp=0; if(ramp>1) ramp=1;
        double extrap_factor = 1.0 - ramp;
        inv[i] = interp*(1.0-extrap_factor) + extrap*extrap_factor;
    }
}

static void rmsnorm(float *out, const float *x, const float *w, int n, float eps){
    double v=0; for(int i=0;i<n;i++) v+=(double)x[i]*x[i]; v/=n;
    float r=(float)(1.0/sqrt(v+eps));
    for(int i=0;i<n;i++) out[i]=w[i]*(x[i]*r);
}
/* y[O] = x[I] @ W[O,I]^T  (torch nn.Linear, W row-major [O,I]) */
static void matmul(float *y, const float *x, const float *W, int O, int I){
    for(int o=0;o<O;o++){ double a=0; const float *w=W+(int64_t)o*I; for(int i=0;i<I;i++) a+=(double)x[i]*w[i]; y[o]=(float)a; }
}
static void softmax(float *x, int n){
    float m=-1e30f; for(int i=0;i<n;i++) if(x[i]>m)m=x[i];
    double s=0; for(int i=0;i<n;i++){ x[i]=expf(x[i]-m); s+=x[i]; }
    for(int i=0;i<n;i++) x[i]/=(float)s;
}
static float siluf(float x){ return x/(1.0f+expf(-x)); }

typedef struct {
    float *in_ln, *post_ln;
    float *q_a, *q_a_ln, *q_b, *kv_a, *kv_a_ln, *kv_b, *o_proj;
    float *gate, *gate_up, *down, *sh_gate, *sh_up, *sh_down;
} Layer;
typedef struct { float *embed, *final_ln, *lm_head; Layer *L; } Weights;

static float *rd(shards *S, const char *name){
    int64_t n=st_numel(S,name); if(n<0){ fprintf(stderr,"missing %s\n",name); exit(1); }
    float *p=malloc(n*sizeof(float)); st_read_f32(S,name,p,0); return p;
}
static void load_weights(Weights *w, shards *S, const Cfg *c){
    w->embed=rd(S,"model.embed_tokens.weight");
    w->final_ln=rd(S,"model.norm.weight");
    w->lm_head=rd(S,"lm_head.weight");
    w->L=calloc(c->n_layers,sizeof(Layer));
    char nm[256];
    for(int l=0;l<c->n_layers;l++){ Layer *L=&w->L[l];
        #define P(field,suffix) snprintf(nm,sizeof nm,"model.layers.%d." suffix,l); L->field=rd(S,nm);
        P(in_ln,"input_layernorm.weight") P(post_ln,"post_attention_layernorm.weight")
        P(q_a,"self_attn.q_a_proj.weight") P(q_a_ln,"self_attn.q_a_layernorm.weight")
        P(q_b,"self_attn.q_b_proj.weight") P(kv_a,"self_attn.kv_a_proj_with_mqa.weight")
        P(kv_a_ln,"self_attn.kv_a_layernorm.weight") P(kv_b,"self_attn.kv_b_proj.weight")
        P(o_proj,"self_attn.o_proj.weight")
        P(gate,"mlp.gate.weight") P(gate_up,"mlp.experts.gate_up_proj") P(down,"mlp.experts.down_proj")
        P(sh_gate,"mlp.shared_experts.gate_proj.weight") P(sh_up,"mlp.shared_experts.up_proj.weight")
        P(sh_down,"mlp.shared_experts.down_proj.weight")
        #undef P
    }
}

static void forward(const Cfg *c, const Weights *w, const int *ids, int S, float *logits_out,
                    const double *inv_freq){
    int D=c->hidden, H=c->n_heads, qkh=c->qk_head, half=c->qk_rope/2;
    double *cosb=malloc((size_t)S*half*sizeof(double)), *sinb=malloc((size_t)S*half*sizeof(double));
    for(int p=0;p<S;p++) for(int i=0;i<half;i++){ double a=p*inv_freq[i]; cosb[p*half+i]=cos(a)*c->attn_scaling; sinb[p*half+i]=sin(a)*c->attn_scaling; }
    float *x=malloc((size_t)S*D*sizeof(float));
    for(int p=0;p<S;p++) memcpy(x+(size_t)p*D, w->embed+(int64_t)ids[p]*D, D*sizeof(float));

    float *nrm=malloc(D*sizeof(float));
    float *qv=malloc((size_t)H*qkh*sizeof(float));
    float *kcache=malloc((size_t)S*H*qkh*sizeof(float));
    float *vcache=malloc((size_t)S*H*c->v_head*sizeof(float));
    float *q_lat=malloc(c->q_lora*sizeof(float)), *q_latn=malloc(c->q_lora*sizeof(float));
    float *kv_lat=malloc((c->kv_lora+c->qk_rope)*sizeof(float)), *kv_latn=malloc(c->kv_lora*sizeof(float));
    float *kv_full=malloc((size_t)H*(c->qk_nope+c->v_head)*sizeof(float));
    float *attn=malloc((size_t)H*c->v_head*sizeof(float));
    float *aout=malloc(D*sizeof(float));
    float *scores=malloc(S*sizeof(float));
    float *rl=malloc(c->n_experts*sizeof(float));
    float *gu=malloc(2*c->moe_inter*sizeof(float)), *dn=malloc(D*sizeof(float));
    float *shg=malloc(c->moe_inter*sizeof(float)), *shu=malloc(c->moe_inter*sizeof(float));
    float *moe=malloc(D*sizeof(float));
    float *qtmp=malloc(c->qk_rope*sizeof(float)), *krot=malloc(c->qk_rope*sizeof(float));

    for(int l=0;l<c->n_layers;l++){ const Layer *L=&w->L[l];
        for(int p=0;p<S;p++){
            rmsnorm(nrm, x+(size_t)p*D, L->in_ln, D, c->eps);
            matmul(q_lat, nrm, L->q_a, c->q_lora, D);
            rmsnorm(q_latn, q_lat, L->q_a_ln, c->q_lora, c->eps);
            matmul(qv, q_latn, L->q_b, H*qkh, c->q_lora);
            matmul(kv_lat, nrm, L->kv_a, c->kv_lora+c->qk_rope, D);
            rmsnorm(kv_latn, kv_lat, L->kv_a_ln, c->kv_lora, c->eps);
            matmul(kv_full, kv_latn, L->kv_b, H*(c->qk_nope+c->v_head), c->kv_lora);
            float *k_rope=kv_lat+c->kv_lora;   /* shared across heads */
            for(int i=0;i<half;i++){ float k1=k_rope[2*i],k2=k_rope[2*i+1];
                krot[i]=(float)(k1*cosb[p*half+i]-k2*sinb[p*half+i]);
                krot[half+i]=(float)(k2*cosb[p*half+i]+k1*sinb[p*half+i]); }
            double l4 = 1.0 + c->llama4_beta*log(1.0+floor((double)p/c->orig_max));
            for(int hh=0;hh<H;hh++){
                float *q=qv+(size_t)hh*qkh;
                float *qrot=q+c->qk_nope;
                for(int i=0;i<half;i++){ float a1=qrot[2*i],a2=qrot[2*i+1];
                    qtmp[i]=(float)(a1*cosb[p*half+i]-a2*sinb[p*half+i]);
                    qtmp[half+i]=(float)(a2*cosb[p*half+i]+a1*sinb[p*half+i]); }
                memcpy(qrot,qtmp,c->qk_rope*sizeof(float));
                for(int i=0;i<qkh;i++) q[i]=(float)(q[i]*l4);
                float *kdst=kcache+(((size_t)p*H+hh)*qkh);
                float *kn=kv_full+(size_t)hh*(c->qk_nope+c->v_head);
                memcpy(kdst, kn, c->qk_nope*sizeof(float));
                memcpy(kdst+c->qk_nope, krot, c->qk_rope*sizeof(float));
                float *vdst=vcache+(((size_t)p*H+hh)*c->v_head);
                memcpy(vdst, kn+c->qk_nope, c->v_head*sizeof(float));
            }
            float scale=1.0f/sqrtf((float)qkh);
            for(int hh=0;hh<H;hh++){
                float *q=qv+(size_t)hh*qkh;
                for(int j=0;j<=p;j++){ float *k=kcache+(((size_t)j*H+hh)*qkh); double a=0;
                    for(int i=0;i<qkh;i++) a+=(double)q[i]*k[i];
                    scores[j]=(float)(a*scale); }
                softmax(scores,p+1);
                float *ao=attn+(size_t)hh*c->v_head;
                for(int i=0;i<c->v_head;i++) ao[i]=0;
                for(int j=0;j<=p;j++){ float *v=vcache+(((size_t)j*H+hh)*c->v_head); float wgt=scores[j];
                    for(int i=0;i<c->v_head;i++) ao[i]+=wgt*v[i]; }
            }
            matmul(aout, attn, L->o_proj, D, H*c->v_head);
            for(int i=0;i<D;i++) x[(size_t)p*D+i]+=aout[i];

            rmsnorm(nrm, x+(size_t)p*D, L->post_ln, D, c->eps);
            matmul(rl, nrm, L->gate, c->n_experts, D);
            softmax(rl, c->n_experts);
            int idx[64]; float wsel[64];
            for(int k=0;k<c->topk;k++){ int best=-1; float bv=-1e30f;
                for(int e=0;e<c->n_experts;e++){ int taken=0; for(int z=0;z<k;z++) if(idx[z]==e){taken=1;break;}
                    if(!taken && rl[e]>bv){bv=rl[e];best=e;} }
                idx[k]=best; wsel[k]=rl[best]; }
            float sm=0; for(int k=0;k<c->topk;k++) sm+=wsel[k]; sm+=1e-20f;
            for(int k=0;k<c->topk;k++) wsel[k]=wsel[k]/sm*c->routed_scale;
            for(int i=0;i<D;i++) moe[i]=0;
            int I=c->moe_inter;
            for(int k=0;k<c->topk;k++){ int e=idx[k]; const float *gw=L->gate_up+(int64_t)e*2*I*D; const float *dw=L->down+(int64_t)e*D*I;
                matmul(gu, nrm, gw, 2*I, D);
                for(int i=0;i<I;i++) gu[i]=siluf(gu[i])*gu[I+i];
                matmul(dn, gu, dw, D, I);
                for(int i=0;i<D;i++) moe[i]+=wsel[k]*dn[i];
            }
            matmul(shg, nrm, L->sh_gate, I, D); matmul(shu, nrm, L->sh_up, I, D);
            for(int i=0;i<I;i++) shg[i]=siluf(shg[i])*shu[i];
            matmul(dn, shg, L->sh_down, D, I);
            for(int i=0;i<D;i++) x[(size_t)p*D+i]+=moe[i]+dn[i];
        }
    }
    for(int p=0;p<S;p++){
        rmsnorm(nrm, x+(size_t)p*D, w->final_ln, D, c->eps);
        matmul(logits_out+(size_t)p*c->vocab, nrm, w->lm_head, c->vocab, D);
    }
    free(cosb);free(sinb);free(x);free(nrm);free(qv);free(kcache);free(vcache);
    free(q_lat);free(q_latn);free(kv_lat);free(kv_latn);free(kv_full);free(attn);free(aout);
    free(scores);free(rl);free(gu);free(dn);free(shg);free(shu);free(moe);free(qtmp);free(krot);
}

static int argmax(const float *v, int n){ int b=0; for(int i=1;i<n;i++) if(v[i]>v[b])b=i; return b; }

static int *jarr(jval *r, const char *k, int *n){
    jval *a=json_get(r,k); *n=a->len; int *o=malloc(a->len*sizeof(int));
    for(int i=0;i<a->len;i++) o[i]=(int)a->kids[i]->num;
    return o;
}

int main(int argc, char **argv){
    if(argc<3){ fprintf(stderr,"usage: %s <tiny_dir> <ref.json>\n",argv[0]); return 1; }
    Cfg c; load_cfg(&c, argv[1]);
    fprintf(stderr,"cfg: hidden=%d L=%d H=%d E=%d topk=%d qk=%d/%d v=%d kvlora=%d yarn(factor=%g orig=%d beta=%g/%g) attn_scaling=%g llama4=%g\n",
        c.hidden,c.n_layers,c.n_heads,c.n_experts,c.topk,c.qk_nope,c.qk_rope,c.v_head,c.kv_lora,
        c.factor,c.orig_max,c.beta_fast,c.beta_slow,c.attn_scaling,c.llama4_beta);
    int half=c.qk_rope/2; double *inv=malloc(half*sizeof(double)); yarn_inv_freq(&c,inv,half);
    fprintf(stderr,"inv_freq[:5]:"); for(int i=0;i<5&&i<half;i++) fprintf(stderr," %.5f",inv[i]);
    fprintf(stderr,"  (torch: 1.0 0.65616 0.42176 0.26356 0.15811)\n");

    shards S; st_init(&S, argv[1]);
    Weights w; load_weights(&w,&S,&c);

    FILE *rf=fopen(argv[2],"rb"); if(!rf){ perror(argv[2]); return 1; }
    fseek(rf,0,SEEK_END); long rn=ftell(rf); fseek(rf,0,SEEK_SET);
    char *rb=malloc(rn+1); if(fread(rb,1,rn,rf)!=(size_t)rn){} rb[rn]=0; fclose(rf);
    char *rar=NULL; jval *rr=json_parse(rb,&rar);
    int np, nf, ntf; int *prompt=jarr(rr,"prompt_ids",&np); int *full=jarr(rr,"full_ids",&nf); int *tfp=jarr(rr,"tf_pred",&ntf);
    (void)ntf;

    float *lg=malloc((size_t)nf*c.vocab*sizeof(float));
    forward(&c,&w,full,nf,lg,inv);
    int ok_tf=0;
    for(int i=0;i<nf;i++){ int am=argmax(lg+(size_t)i*c.vocab,c.vocab); if(am==tfp[i]) ok_tf++; }
    fprintf(stderr,"teacher-forcing argmax: %d/%d match tf_pred\n", ok_tf, nf);

    int L=nf; int *gen=malloc(L*sizeof(int)); memcpy(gen,prompt,np*sizeof(int)); int glen=np;
    while(glen<L){ float *g=malloc((size_t)glen*c.vocab*sizeof(float)); forward(&c,&w,gen,glen,g,inv);
        int nx=argmax(g+(size_t)(glen-1)*c.vocab,c.vocab); free(g); gen[glen++]=nx; }
    int ok_g=0; for(int i=np;i<L;i++) if(gen[i]==full[i]) ok_g++;
    fprintf(stderr,"greedy: %d/%d match full_ids (generated tail)\n", ok_g, L-np);

    printf("P0 mistral4: tf %d/%d, greedy %d/%d\n", ok_tf,nf, ok_g,L-np);
    return (ok_tf==nf && ok_g==L-np) ? 0 : 2;
}
