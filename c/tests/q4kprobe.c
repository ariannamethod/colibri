/* q4kprobe — does notorch's nt_qmatvec (dtype 12) read OUR Q4_K expert bytes correctly?
 * Same shape of proof as q6probe: y_q4k = nt_qmatvec(our bytes) vs y_ref = f32 matvec of the
 * original BF16 expert. Container verify already proved our DECODER matches gguf-py; this
 * proves the ENGINE kernel agrees on the same bytes.
 * usage: q4kprobe <bf16_dir> <t5f_dir> <tensor_name> <O> <I> */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "json.h"
#include "st.h"
#include "notorch.h"

int main(int argc,char**argv){
    if(argc<6){fprintf(stderr,"usage: %s <bf16> <t5f> <name> <O> <I>\n",argv[0]);return 2;}
    const char *NM=argv[3]; int O=atoi(argv[4]), I=atoi(argv[5]);
    shards A,B; st_init(&A,argv[1]); st_init(&B,argv[2]);
    int64_t nblk=(int64_t)O*(I/256), nb=st_nbytes(&B,NM);
    int dtype = (nb==nblk*144) ? 12 : (nb==nblk*210) ? 14 : -1;
    printf("%s: %lld B -> %s (Q4_K would be %lld, Q6_K %lld)\n", NM,(long long)nb,
           dtype==12?"Q4_K":dtype==14?"Q6_K":"NEITHER",(long long)(nblk*144),(long long)(nblk*210));
    if(dtype<0) return 1;
    float *W=malloc((size_t)O*I*sizeof(float)); st_read_f32(&A,NM,W,0);
    uint8_t *Q=malloc(nb); st_read_raw(&B,NM,Q,0);
    float *x=malloc(I*sizeof(float)), *yr=malloc(O*sizeof(float)), *yq=malloc(O*sizeof(float));
    srand(99);
    double worst=0;
    for(int t=0;t<4;t++){
        for(int i=0;i<I;i++) x[i]=((float)rand()/(float)RAND_MAX-0.5f)*2.0f;
        for(int o=0;o<O;o++){ double a=0; const float *w=W+(size_t)o*I;
            for(int i=0;i<I;i++) a+=(double)w[i]*x[i];
            yr[o]=(float)a; }
        if(nt_qmatvec(yq,Q,dtype,x,O,I)!=0){ fprintf(stderr,"nt_qmatvec refused dtype %d\n",dtype); return 1; }
        double num=0,den=0,dot=0,na=0,nbn=0;
        for(int o=0;o<O;o++){ num+=fabs((double)yq[o]-yr[o]); den+=fabs((double)yr[o]);
            dot+=(double)yq[o]*yr[o]; na+=(double)yq[o]*yq[o]; nbn+=(double)yr[o]*yr[o]; }
        double rel=num/den, cos=dot/(sqrt(na)*sqrt(nbn)+1e-300);
        if(rel>worst) worst=rel;
        printf("  trial %d: relL1=%.6f cos(f64)=%.10f\n",t,rel,cos);
    }
    printf("worst relL1 %.6f  -> %s\n",worst, worst<0.10?"kernel agrees (quant floor)":"KERNEL/WIRING BUG");
    return worst<0.10?0:2;
}
