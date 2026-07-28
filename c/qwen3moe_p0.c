/* qwen3moe_p0.c — minimal self-contained f32 forward of Qwen3-MoE (qwen3moe),
 * P0 parity gate of the Colibri->Arianna port (body = Qwen3-30B-A3B-Base).
 *
 * NOT the production engine: all weights resident, naive GQA, no streaming/cache/CUDA.
 * Only job: prove the arch (GQA + q/k-norm + plain rotate_half RoPE + softmax router +
 * per-expert SwiGLU) reproduces the torch reference token-exact. Semantics probed from
 * transformers Qwen3Moe / the saved safetensors, not guessed. Arch vs mistral4_p0:
 * MLA->GQA+q/k-norm, YaRN->plain rope, no shared expert, no routed_scale; experts are
 * separate per-expert gate/up/down (checkpoint contract, same as olmoe/glm).
 *
 * Build: cc -O2 -Wall -o qwen3moe_p0 qwen3moe_p0.c -lm
 * Run:   ./qwen3moe_p0 <qwen3moe_tiny_dir> <ref_qwen3moe.json>
 * Gate:  cc 0 warn | tf N/N | greedy 20/20.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "json.h"
#include "st.h"

typedef struct {
    int hidden, n_layers, n_heads, n_kv_heads, head_dim, n_experts, topk, moe_inter, vocab, norm_topk;
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
    free(ar); free(buf);
}

static void rmsnorm(float *out, const float *x, const float *w, int n, float eps){
    double v=0; for(int i=0;i<n;i++) v+=(double)x[i]*x[i]; v/=n;
    float r=(float)(1.0/sqrt(v+eps));
    for(int i=0;i<n;i++) out[i]=w[i]*(x[i]*r);
}
/* y[O] = x[I] @ W[O,I]^T (torch nn.Linear, W row-major [O,I]) */
static void matmul(float *y, const float *x, const float *W, int O, int I){
    for(int o=0;o<O;o++){ double a=0; const float *w=W+(int64_t)o*I; for(int i=0;i<I;i++) a+=(double)x[i]*w[i]; y[o]=(float)a; }
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

typedef struct {
    float *in_ln, *post_ln, *q_proj, *k_proj, *v_proj, *o_proj, *q_norm, *k_norm, *gate;
    float **eg, **eu, **ed;   /* per-expert gate/up/down, arrays of n_experts pointers */
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
        P(q_proj,"self_attn.q_proj.weight") P(k_proj,"self_attn.k_proj.weight")
        P(v_proj,"self_attn.v_proj.weight") P(o_proj,"self_attn.o_proj.weight")
        P(q_norm,"self_attn.q_norm.weight") P(k_norm,"self_attn.k_norm.weight")
        P(gate,"mlp.gate.weight")
        #undef P
        L->eg=malloc(c->n_experts*sizeof(float*));
        L->eu=malloc(c->n_experts*sizeof(float*));
        L->ed=malloc(c->n_experts*sizeof(float*));
        for(int e=0;e<c->n_experts;e++){
            snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.%d.gate_proj.weight",l,e); L->eg[e]=rd(S,nm);
            snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.%d.up_proj.weight",l,e);   L->eu[e]=rd(S,nm);
            snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.%d.down_proj.weight",l,e); L->ed[e]=rd(S,nm);
        }
    }
}

static void forward(const Cfg *c, const Weights *w, const int *ids, int S, float *logits_out, const double *inv){
    int D=c->hidden, H=c->n_heads, KV=c->n_kv_heads, hd=c->head_dim, half=hd/2, group=H/KV, I=c->moe_inter, E=c->n_experts;
    double *cosb=malloc((size_t)S*half*sizeof(double)), *sinb=malloc((size_t)S*half*sizeof(double));
    for(int p=0;p<S;p++) for(int i=0;i<half;i++){ double a=p*inv[i]; cosb[p*half+i]=cos(a); sinb[p*half+i]=sin(a); }
    float *x=malloc((size_t)S*D*sizeof(float));
    for(int p=0;p<S;p++) memcpy(x+(size_t)p*D, w->embed+(int64_t)ids[p]*D, D*sizeof(float));

    float *nrm=malloc(D*sizeof(float));
    float *q=malloc((size_t)H*hd*sizeof(float));
    float *k=malloc((size_t)KV*hd*sizeof(float));
    float *v=malloc((size_t)KV*hd*sizeof(float));
    float *kcache=malloc((size_t)S*KV*hd*sizeof(float));
    float *vcache=malloc((size_t)S*KV*hd*sizeof(float));
    float *attn=malloc((size_t)H*hd*sizeof(float));
    float *aout=malloc(D*sizeof(float));
    float *scores=malloc(S*sizeof(float));
    float *rl=malloc(E*sizeof(float));
    float *gg=malloc(I*sizeof(float)), *uu=malloc(I*sizeof(float)), *dn=malloc(D*sizeof(float)), *moe=malloc(D*sizeof(float));

    for(int l=0;l<c->n_layers;l++){ const Layer *L=&w->L[l];
        for(int p=0;p<S;p++){
            const double *cp=cosb+(size_t)p*half, *sp=sinb+(size_t)p*half;
            rmsnorm(nrm, x+(size_t)p*D, L->in_ln, D, c->eps);
            matmul(q, nrm, L->q_proj, H*hd, D);
            matmul(k, nrm, L->k_proj, KV*hd, D);
            matmul(v, nrm, L->v_proj, KV*hd, D);
            for(int h=0;h<H;h++){ float *qh=q+(size_t)h*hd; rmsnorm(qh,qh,L->q_norm,hd,c->eps); rope(qh,cp,sp,hd); }
            for(int kh=0;kh<KV;kh++){ float *kk=k+(size_t)kh*hd; rmsnorm(kk,kk,L->k_norm,hd,c->eps); rope(kk,cp,sp,hd);
                memcpy(kcache+(((size_t)p*KV+kh)*hd), kk, hd*sizeof(float));
                memcpy(vcache+(((size_t)p*KV+kh)*hd), v+(size_t)kh*hd, hd*sizeof(float)); }
            float scale=1.0f/sqrtf((float)hd);
            for(int h=0;h<H;h++){ int kh=h/group; float *qh=q+(size_t)h*hd;
                for(int j=0;j<=p;j++){ float *kk=kcache+(((size_t)j*KV+kh)*hd); double a=0;
                    for(int i=0;i<hd;i++){ a+=(double)qh[i]*kk[i]; }
                    scores[j]=(float)(a*scale); }
                softmax(scores,p+1);
                float *ao=attn+(size_t)h*hd; for(int i=0;i<hd;i++) ao[i]=0;
                for(int j=0;j<=p;j++){ float *vv=vcache+(((size_t)j*KV+kh)*hd); float wgt=scores[j];
                    for(int i=0;i<hd;i++) ao[i]+=wgt*vv[i]; } }
            matmul(aout, attn, L->o_proj, D, H*hd);
            for(int i=0;i<D;i++) x[(size_t)p*D+i]+=aout[i];

            rmsnorm(nrm, x+(size_t)p*D, L->post_ln, D, c->eps);
            matmul(rl, nrm, L->gate, E, D);
            softmax(rl, E);
            int idx[256]; float wsel[256];
            for(int kk=0;kk<c->topk;kk++){ int best=-1; float bv=-1e30f;
                for(int e=0;e<E;e++){ int taken=0; for(int z=0;z<kk;z++) if(idx[z]==e){taken=1;break;}
                    if(!taken && rl[e]>bv){bv=rl[e];best=e;} }
                idx[kk]=best; wsel[kk]=rl[best]; }
            if(c->norm_topk){ float sm=0; for(int kk=0;kk<c->topk;kk++) sm+=wsel[kk]; sm+=1e-20f;
                for(int kk=0;kk<c->topk;kk++) wsel[kk]/=sm; }
            for(int i=0;i<D;i++) moe[i]=0;
            for(int kk=0;kk<c->topk;kk++){ int e=idx[kk];
                matmul(gg, nrm, L->eg[e], I, D);
                matmul(uu, nrm, L->eu[e], I, D);
                for(int i=0;i<I;i++){ gg[i]=siluf(gg[i])*uu[i]; }
                matmul(dn, gg, L->ed[e], D, I);
                for(int i=0;i<D;i++){ moe[i]+=wsel[kk]*dn[i]; } }
            for(int i=0;i<D;i++) x[(size_t)p*D+i]+=moe[i];
        }
    }
    for(int p=0;p<S;p++){ rmsnorm(nrm, x+(size_t)p*D, w->final_ln, D, c->eps);
        matmul(logits_out+(size_t)p*c->vocab, nrm, w->lm_head, c->vocab, D); }
    free(cosb);free(sinb);free(x);free(nrm);free(q);free(k);free(v);free(kcache);free(vcache);
    free(attn);free(aout);free(scores);free(rl);free(gg);free(uu);free(dn);free(moe);
}

static int argmax(const float *v, int n){ int b=0; for(int i=1;i<n;i++) if(v[i]>v[b])b=i; return b; }
static int *jarr(jval *r, const char *k, int *n){ jval *a=json_get(r,k); *n=a->len; int *o=malloc(a->len*sizeof(int));
    for(int i=0;i<a->len;i++){ o[i]=(int)a->kids[i]->num; }
    return o; }

int main(int argc, char **argv){
    if(argc<3){ fprintf(stderr,"usage: %s <tiny_dir> <ref.json>\n",argv[0]); return 1; }
    Cfg c; load_cfg(&c, argv[1]);
    fprintf(stderr,"cfg: hidden=%d L=%d H=%d KV=%d hd=%d E=%d topk=%d moe_inter=%d vocab=%d theta=%g norm_topk=%d\n",
        c.hidden,c.n_layers,c.n_heads,c.n_kv_heads,c.head_dim,c.n_experts,c.topk,c.moe_inter,c.vocab,c.theta,c.norm_topk);
    int half=c.head_dim/2; double *inv=malloc(half*sizeof(double));
    for(int i=0;i<half;i++) inv[i]=1.0/pow(c.theta,(double)(2*i)/c.head_dim);
    fprintf(stderr,"inv_freq[:5]:"); for(int i=0;i<5&&i<half;i++) fprintf(stderr," %.6g",inv[i]); fprintf(stderr,"\n");

    shards S; st_init(&S, argv[1]);
    Weights w; load_weights(&w,&S,&c);

    FILE *rf=fopen(argv[2],"rb"); if(!rf){ perror(argv[2]); return 1; }
    fseek(rf,0,SEEK_END); long rn=ftell(rf); fseek(rf,0,SEEK_SET);
    char *rb=malloc(rn+1); if(fread(rb,1,rn,rf)!=(size_t)rn){} rb[rn]=0; fclose(rf);
    char *rar=NULL; jval *rr=json_parse(rb,&rar);
    int np,nf,ntf; int *prompt=jarr(rr,"prompt_ids",&np); int *full=jarr(rr,"full_ids",&nf); int *tfp=jarr(rr,"tf_pred",&ntf); (void)ntf;

    float *lg=malloc((size_t)nf*c.vocab*sizeof(float));
    forward(&c,&w,full,nf,lg,inv);
    int ok_tf=0; for(int i=0;i<nf;i++){ if(argmax(lg+(size_t)i*c.vocab,c.vocab)==tfp[i]) ok_tf++; }
    fprintf(stderr,"teacher-forcing argmax: %d/%d match tf_pred\n", ok_tf, nf);

    int Ln=nf; int *gen=malloc(Ln*sizeof(int)); memcpy(gen,prompt,np*sizeof(int)); int glen=np;
    while(glen<Ln){ float *g=malloc((size_t)glen*c.vocab*sizeof(float)); forward(&c,&w,gen,glen,g,inv);
        int nx=argmax(g+(size_t)(glen-1)*c.vocab,c.vocab); free(g); gen[glen++]=nx; }
    int ok_g=0; for(int i=np;i<Ln;i++) if(gen[i]==full[i]) ok_g++;
    fprintf(stderr,"greedy: %d/%d match full_ids (generated tail)\n", ok_g, Ln-np);
    printf("P0 qwen3moe: tf %d/%d, greedy %d/%d\n", ok_tf,nf, ok_g,Ln-np);
    return (ok_tf==nf && ok_g==Ln-np) ? 0 : 2;
}
