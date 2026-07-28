/* qwen_tok.h — Qwen byte-level BPE for the Colibri Qwen3-MoE front-end.
 *
 * Ported from tok.h (the GLM/cl100k tokenizer, which stays untouched and is still what
 * glm.c uses) with the differences READ OUT OF tokenizer.json rather than assumed:
 *
 *   1. merges are a flat array of STRINGS ("Ġ Ġ"), split on the first space, not an
 *      array of pairs as in the GLM file. Byte-level encoding leaves no literal space
 *      inside a token, so the first space is an unambiguous separator.
 *   2. model.ignore_merges is null/false here, so a whole piece present in the vocab must
 *      still go through the merge ladder; tok.h short-circuits it because GLM sets it true.
 *   3. the Split pattern differs from cl100k in exactly one alternative: \p{N} (one digit
 *      per piece) against cl100k's \p{N}{1,3}. Confirmed against the oracle: "2026" ->
 *      four tokens 17,15,17,21.
 *
 * The pattern is not hardcoded blindly: the loader compares the file's pattern against the
 * dialects it knows and dies on an unfamiliar one, so a future checkpoint with a different
 * pre-tokenizer fails loudly instead of silently mis-tokenizing.
 *
 * NOT implemented: the NFC normalizer declared in the file. Full NFC needs composition
 * tables this tree does not carry. Inputs already in NFC (every corpus we feed it, and
 * every file produced by a normal editor) are unaffected, and the parity gate covers
 * precomposed Cyrillic explicitly. A decomposed input would diverge — asserted, not hidden.
 */
#ifndef QWEN_TOK_H
#define QWEN_TOK_H
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include "json.h"
#include "tok_unicode.h"

typedef struct { const char *k; int klen; int v; int used; } qment;
typedef struct { qment *e; int cap; } qhmap;
static uint64_t qtk_fnv(const char *s, int n){ uint64_t h=1469598103934665603ULL;
    for(int i=0;i<n;i++){ h^=(unsigned char)s[i]; h*=1099511628211ULL; } return h; }
static void qhm_init(qhmap *m, int cap){ m->cap=cap; m->e=(qment*)calloc(cap,sizeof(qment)); }
static void qhm_put(qhmap *m, const char *k, int klen, int v){
    uint64_t h=qtk_fnv(k,klen)&(m->cap-1);
    while(m->e[h].used){ if(m->e[h].klen==klen && !memcmp(m->e[h].k,k,klen)){ m->e[h].v=v; return; } h=(h+1)&(m->cap-1); }
    m->e[h].k=k; m->e[h].klen=klen; m->e[h].v=v; m->e[h].used=1;
}
static int qhm_get(qhmap *m, const char *k, int klen){
    uint64_t h=qtk_fnv(k,klen)&(m->cap-1);
    while(m->e[h].used){ if(m->e[h].klen==klen && !memcmp(m->e[h].k,k,klen)) return m->e[h].v; h=(h+1)&(m->cap-1); }
    return -1;
}

typedef struct { char *str; int len; int id; } QSpecial;
typedef struct {
    qhmap vocab, merges;
    char **id2str; int *id_added; int n_ids;
    QSpecial *sp; int nsp;
    uint32_t byte2cp[256]; int byte2cp_len[256]; char byte2str[256][3];
    int16_t cp2byte[1024];
    int digit_run;        /* max digits per piece: 1 (Qwen) or 3 (cl100k) — from the file */
    int ignore_merges;    /* from model.ignore_merges */
} QTok;

static int qu8_next(const unsigned char *s, int len, int i, uint32_t *cp){
    unsigned char c=s[i];
    if(c<0x80){ *cp=c; return 1; }
    if((c>>5)==0x6 && i+1<len){ *cp=((c&0x1F)<<6)|(s[i+1]&0x3F); return 2; }
    if((c>>4)==0xE && i+2<len){ *cp=((c&0x0F)<<12)|((s[i+1]&0x3F)<<6)|(s[i+2]&0x3F); return 3; }
    if((c>>3)==0x1E && i+3<len){ *cp=((c&0x07)<<18)|((s[i+1]&0x3F)<<12)|((s[i+2]&0x3F)<<6)|(s[i+3]&0x3F); return 4; }
    *cp=c; return 1;
}
static int qu8_put(char *o, uint32_t cp){
    if(cp<0x80){ o[0]=cp; return 1; }
    if(cp<0x800){ o[0]=0xC0|(cp>>6); o[1]=0x80|(cp&0x3F); return 2; }
    if(cp<0x10000){ o[0]=0xE0|(cp>>12); o[1]=0x80|((cp>>6)&0x3F); o[2]=0x80|(cp&0x3F); return 3; }
    o[0]=0xF0|(cp>>18); o[1]=0x80|((cp>>12)&0x3F); o[2]=0x80|((cp>>6)&0x3F); o[3]=0x80|(cp&0x3F); return 4;
}

static void qtk_build_bytemap(QTok *T){
    for(int i=0;i<1024;i++) T->cp2byte[i]=-1;
    int isdir[256]; memset(isdir,0,sizeof(isdir));
    for(int b=33;b<=126;b++) isdir[b]=1;
    for(int b=161;b<=172;b++) isdir[b]=1;
    for(int b=174;b<=255;b++) isdir[b]=1;
    int n=0;
    for(int b=0;b<256;b++){
        uint32_t cp = isdir[b] ? (uint32_t)b : (uint32_t)(256+n);
        if(!isdir[b]) n++;
        T->byte2cp[b]=cp;
        T->byte2cp_len[b]=qu8_put(T->byte2str[b], cp);
        if(cp<1024) T->cp2byte[cp]=b;
    }
}

static char *qtk_read_file(const char *path, long *out_n){
    FILE *f=fopen(path,"rb"); if(!f){ perror(path); exit(1); }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *b=malloc(n+1); if(fread(b,1,n,f)!=(size_t)n){} b[n]=0; fclose(f); if(out_n)*out_n=n; return b;
}
static int qcmp_sp_len(const void *a, const void *b){ return ((const QSpecial*)b)->len - ((const QSpecial*)a)->len; }

/* The one-digit alternative is what separates the Qwen dialect from cl100k. */
#define QWEN_PAT_DIGIT1 "\\p{N}|"
#define QWEN_PAT_DIGIT3 "\\p{N}{1,3}|"

static void qtok_load(QTok *T, const char *path){
    memset(T,0,sizeof(*T));
    qtk_build_bytemap(T);
    long fn; char *buf=qtk_read_file(path,&fn);
    char *arena=NULL; jval *root=json_parse(buf,&arena);
    jval *model=json_get(root,"model");
    jval *vocab=json_get(model,"vocab");
    jval *merges=json_get(model,"merges");
    jval *added=json_get(root,"added_tokens");
    if(!vocab||!merges){ fprintf(stderr,"%s: missing model.vocab/merges\n",path); exit(1); }

    /* --- dialect, read from the file, never assumed --- */
    T->digit_run=0;
    { jval *pt=json_get(root,"pre_tokenizer"); jval *seq=pt?json_get(pt,"pretokenizers"):NULL;
      const char *pat=NULL;
      if(seq && seq->t==J_ARR) for(int i=0;i<seq->len;i++){
          jval *p=json_get(seq->kids[i],"pattern"); jval *rx=p?json_get(p,"Regex"):NULL;
          if(rx && rx->t==J_STR){ pat=rx->str; break; } }
      if(!pat){ fprintf(stderr,"%s: no pre_tokenizer Split/Regex pattern\n",path); exit(1); }
      if(strstr(pat,QWEN_PAT_DIGIT1))      T->digit_run=1;
      else if(strstr(pat,QWEN_PAT_DIGIT3)) T->digit_run=3;
      else { fprintf(stderr,"%s: unrecognised pre-tokenizer pattern, refusing to guess:\n  %s\n",path,pat); exit(1); }
    }
    { jval *im=json_get(model,"ignore_merges");
      T->ignore_merges = (im && im->t==J_BOOL) ? im->boolean : 0; }

    int maxid=0;
    for(int i=0;i<vocab->len;i++){ int id=(int)vocab->kids[i]->num; if(id>maxid)maxid=id; }
    if(added) for(int i=0;i<added->len;i++){ int id=(int)json_get(added->kids[i],"id")->num; if(id>maxid)maxid=id; }
    T->n_ids=maxid+1;
    T->id2str=calloc(T->n_ids,sizeof(char*));
    T->id_added=calloc(T->n_ids,sizeof(int));

    int vc=1; while(vc < vocab->len*2) vc<<=1;
    qhm_init(&T->vocab, vc);
    for(int i=0;i<vocab->len;i++){
        const char *k=vocab->keys[i]; int id=(int)vocab->kids[i]->num;
        qhm_put(&T->vocab, k, (int)strlen(k), id);
        if(id>=0 && id<T->n_ids) T->id2str[id]=(char*)k;
    }

    /* merges: accept both shapes, detected per entry. Qwen ships "left right" strings. */
    int mc=1; while(mc < merges->len*2) mc<<=1;
    qhm_init(&T->merges, mc);
    for(int i=0;i<merges->len;i++){
        jval *pr=merges->kids[i];
        const char *l=NULL,*r=NULL; int ll=0,rl=0;
        if(pr->t==J_ARR && pr->len==2){ l=pr->kids[0]->str; r=pr->kids[1]->str; ll=(int)strlen(l); rl=(int)strlen(r); }
        else if(pr->t==J_STR){
            const char *sp=strchr(pr->str,' ');
            if(!sp){ fprintf(stderr,"merge %d has no separator: %s\n",i,pr->str); exit(1); }
            l=pr->str; ll=(int)(sp-pr->str); r=sp+1; rl=(int)strlen(r);
        } else { fprintf(stderr,"merge %d: unsupported shape\n",i); exit(1); }
        char *key=malloc(ll+1+rl); memcpy(key,l,ll); key[ll]=0; memcpy(key+ll+1,r,rl);
        qhm_put(&T->merges, key, ll+1+rl, i);
    }

    if(added){
        T->nsp=added->len; T->sp=calloc(T->nsp,sizeof(QSpecial));
        for(int i=0;i<added->len;i++){
            jval *a=added->kids[i];
            char *content=json_get(a,"content")->str; int id=(int)json_get(a,"id")->num;
            T->sp[i].str=content; T->sp[i].len=(int)strlen(content); T->sp[i].id=id;
            if(id>=0 && id<T->n_ids){ T->id2str[id]=content; T->id_added[id]=1; }
        }
        qsort(T->sp,T->nsp,sizeof(QSpecial),qcmp_sp_len);
    }
    (void)arena;
}

static void qbpe_piece(QTok *T, const unsigned char *p, int a, int b, int *out, int *no, int max){
    int nb=b-a; if(nb<=0) return;
    char *s=malloc(2*nb+1); int sl=0;
    for(int i=a;i<b;i++){ int bb=p[i]; memcpy(s+sl,T->byte2str[bb],T->byte2cp_len[bb]); sl+=T->byte2cp_len[bb]; }
    s[sl]=0;
    /* Only legal when the file says ignore_merges; Qwen does not, so the ladder runs. */
    if(T->ignore_merges){
        int whole=qhm_get(&T->vocab,s,sl);
        if(whole>=0){ if(*no<max) out[(*no)++]=whole; free(s); return; }
    }
    int *soff=malloc((sl+1)*sizeof(int)), *slen=malloc((sl+1)*sizeof(int)); int ns=0;
    for(int i=0;i<sl;){ uint32_t cp; int k=qu8_next((const unsigned char*)s,sl,i,&cp);
        soff[ns]=i; slen[ns]=k; ns++; i+=k; }
    char *kbuf=malloc(2*sl+2);
    for(;;){
        int best=INT_MAX, bp=-1;
        for(int i=0;i+1<ns;i++){
            int ll=slen[i], rl=slen[i+1];
            memcpy(kbuf,s+soff[i],ll); kbuf[ll]=0; memcpy(kbuf+ll+1,s+soff[i+1],rl);
            int rk=qhm_get(&T->merges,kbuf,ll+1+rl);
            if(rk>=0 && rk<best){ best=rk; bp=i; }
        }
        if(bp<0) break;
        slen[bp]=soff[bp+1]+slen[bp+1]-soff[bp];
        for(int j=bp+1;j<ns-1;j++){ soff[j]=soff[j+1]; slen[j]=slen[j+1]; }
        ns--;
    }
    for(int i=0;i<ns;i++){
        int id=qhm_get(&T->vocab,s+soff[i],slen[i]);
        if(id>=0 && *no<max) out[(*no)++]=id;
    }
    free(s); free(soff); free(slen); free(kbuf);
}

static void qpretok_chunk(QTok *T, const unsigned char *p, int a, int b, int *out, int *no, int max){
    int nb=b-a; if(nb<=0) return;
    uint32_t *cp=malloc((nb+1)*sizeof(uint32_t)); int *off=malloc((nb+2)*sizeof(int)); int n=0;
    for(int i=a;i<b;){ uint32_t c; int k=qu8_next(p,b,i,&c); off[n]=i; cp[n]=c; n++; i+=k; }
    off[n]=b;
    #define QISNL(c) ((c)=='\r'||(c)=='\n')
    #define QLOW(c) (((c)>='A'&&(c)<='Z')?((c)+32):(c))
    int i=0;
    while(i<n){
        int start=i; uint32_t c=cp[i];
        if(c=='\'' && i+1<n){
            uint32_t d=QLOW(cp[i+1]);
            if(i+2<n){ uint32_t d2=QLOW(cp[i+2]);
                if((d=='r'&&d2=='e')||(d=='v'&&d2=='e')||(d=='l'&&d2=='l')){ i+=3; qbpe_piece(T,p,off[start],off[i],out,no,max); continue; } }
            if(d=='s'||d=='t'||d=='m'||d=='d'){ i+=2; qbpe_piece(T,p,off[start],off[i],out,no,max); continue; }
        }
        {
            int j=i;
            if(!is_L(c) && !QISNL(c) && !is_N(c)){ if(j+1<n && is_L(cp[j+1])) j++; else j=-1; }
            if(j>=0 && is_L(cp[j])){ while(j<n && is_L(cp[j])) j++; i=j; qbpe_piece(T,p,off[start],off[i],out,no,max); continue; }
        }
        if(is_N(c)){ int j=i,k=0; while(j<n && is_N(cp[j]) && k<T->digit_run){ j++; k++; } i=j;
            qbpe_piece(T,p,off[start],off[i],out,no,max); continue; }
        {
            int j=i;
            if(c==' ' && j+1<n && !is_S(cp[j+1]) && !is_L(cp[j+1]) && !is_N(cp[j+1])) j++;
            if(j<n && !is_S(cp[j]) && !is_L(cp[j]) && !is_N(cp[j])){
                while(j<n && !is_S(cp[j]) && !is_L(cp[j]) && !is_N(cp[j])) j++;
                while(j<n && QISNL(cp[j])) j++;
                i=j; qbpe_piece(T,p,off[start],off[i],out,no,max); continue;
            }
        }
        {
            int r=i; while(r<n && is_S(cp[r])) r++;
            if(r>i){ int last=-1; for(int j=i;j<r;j++) if(QISNL(cp[j])) last=j;
                if(last>=0){ i=last+1; qbpe_piece(T,p,off[start],off[i],out,no,max); continue; }
                int end = (r<n) ? r-1 : r;
                if(end<=i) end=i+1;
                i=end; qbpe_piece(T,p,off[start],off[i],out,no,max); continue;
            }
        }
        i++;
        qbpe_piece(T,p,off[start],off[i],out,no,max);
    }
    #undef QISNL
    #undef QLOW
    free(cp); free(off);
}

static int qtok_encode(QTok *T, const char *text, int len, int *out, int max){
    const unsigned char *p=(const unsigned char*)text; int no=0; int i=0;
    while(i<len){
        int hitpos=-1, hitlen=0, hitid=-1;
        for(int j=i;j<len && hitpos<0;j++){
            for(int k=0;k<T->nsp;k++){
                int sl=T->sp[k].len;
                if(sl>0 && j+sl<=len && !memcmp(p+j,T->sp[k].str,sl)){ hitpos=j; hitlen=sl; hitid=T->sp[k].id; break; }
            }
        }
        int chunk_end = (hitpos<0) ? len : hitpos;
        if(chunk_end>i) qpretok_chunk(T,p,i,chunk_end,out,&no,max);
        if(hitpos<0) break;
        if(no<max) out[no++]=hitid;
        i=hitpos+hitlen;
    }
    return no;
}

static int qtok_id_of(QTok *T, const char *content){
    for(int i=0;i<T->nsp;i++) if(!strcmp(T->sp[i].str,content)) return T->sp[i].id;
    return -1;
}

static int qtok_decode(QTok *T, const int *ids, int n, char *out, int max){
    int o=0;
    for(int i=0;i<n;i++){
        int id=ids[i]; if(id<0||id>=T->n_ids||!T->id2str[id]) continue;
        const char *s=T->id2str[id];
        if(T->id_added[id]){ int l=(int)strlen(s); for(int j=0;j<l && o<max;j++) out[o++]=s[j]; continue; }
        int sl=(int)strlen(s);
        for(int j=0;j<sl;){ uint32_t c; int k=qu8_next((const unsigned char*)s,sl,j,&c); j+=k;
            if(c<1024 && T->cp2byte[c]>=0 && o<max) out[o++]=(char)(unsigned char)T->cp2byte[c]; }
    }
    if(o<max) out[o]=0;
    return o;
}

#endif
