/* q6probe - does notorch's nt_qmatvec (dtype 14) read OUR Q6_K head bytes correctly?
 * Block A proved our bytes == gguf-py reference decode. This proves the ENGINE kernel
 * agrees too: y_q6 = nt_qmatvec(our bytes) vs y_ref = f32 matvec of the original head.
 * usage: q6probe <f32_dir> <q6_dir> <O> <I> */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include "json.h"
#include "st.h"
#include "notorch.h"

static int amax_i(const float *v,int n){int b=0;for(int i=1;i<n;i++)if(v[i]>v[b])b=i;return b;}

static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e3 + t.tv_nsec/1e6; }

/* T5e step 0: what the 23.19 ms/tok head actually consists of, measured not guessed.
 * Thread spawn is timed as the consumer pays it — six pthreads created and joined per
 * call — and the row hand-out is probed by comparing the threaded call against the same
 * call forced single-threaded (NT_QMV_THREAD_MIN is read once and cached, so the floor
 * must be set in the environment before the first call). */
static void *noop_worker(void *p){ (void)p; return NULL; }
static double spawn_cost_ms(int nthr, int reps){
    double t0=now_ms();
    for(int r=0;r<reps;r++){
        pthread_t th[16];
        for(int t=0;t<nthr;t++) pthread_create(&th[t],NULL,noop_worker,NULL);
        for(int t=0;t<nthr;t++) pthread_join(th[t],NULL);
    }
    return (now_ms()-t0)/reps;
}

int main(int argc,char**argv){
    if(argc<5){fprintf(stderr,"usage: q6probe <f32_dir> <q6_dir> <O> <I>\n");return 2;}
    int O=atoi(argv[3]), I=atoi(argv[4]);
    shards A,B; st_init(&A,argv[1]); st_init(&B,argv[2]);
    const char *NM="lm_head.weight";
    int64_t want=(int64_t)O*(I/256)*210, nb=st_nbytes(&B,NM);
    printf("Q6 blob %lld B, expected O*(I/256)*210 = %lld B  [%s]\n",
           (long long)nb,(long long)want, nb==want?"MATCH":"MISMATCH");
    if(nb!=want) return 1;
    float *W=malloc((size_t)O*I*sizeof(float)); st_read_f32(&A,NM,W,0);
    uint8_t *Q=malloc(nb); st_read_raw(&B,NM,Q,0);
    float *x=malloc(I*sizeof(float)), *yr=malloc(O*sizeof(float)), *yq=malloc(O*sizeof(float));
    srand(1234);
    int agree=0, trials=8;
    double tsum=0, tmin=1e30;
    double worst=0;
    for(int t=0;t<trials;t++){
        for(int i=0;i<I;i++) x[i]=((float)rand()/(float)RAND_MAX-0.5f)*2.0f;
        for(int o=0;o<O;o++){ double a=0; const float *w=W+(size_t)o*I;
            for(int i=0;i<I;i++) a+=(double)w[i]*x[i];
            yr[o]=(float)a; }
        struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
        if(nt_qmatvec(yq,Q,14,x,O,I)!=0){ fprintf(stderr,"nt_qmatvec refused dtype 14\n"); return 1; }
        clock_gettime(CLOCK_MONOTONIC,&t1);
        double ms=(t1.tv_sec-t0.tv_sec)*1e3+(t1.tv_nsec-t0.tv_nsec)/1e6;
        tsum+=ms; if(ms<tmin) tmin=ms;
        double num=0,den=0,dot=0,na=0,nbn=0,mx=0;
        for(int o=0;o<O;o++){
            double d=fabs((double)yq[o]-yr[o]); num+=d; den+=fabs((double)yr[o]);
            if(d>mx) mx=d;
            dot+=(double)yq[o]*yr[o]; na+=(double)yq[o]*yq[o]; nbn+=(double)yr[o]*yr[o];
        }
        double rel=num/den, cos=dot/(sqrt(na)*sqrt(nbn)+1e-300);
        int a1=amax_i(yq,O), a2=amax_i(yr,O);
        if(a1==a2) agree++;
        else printf("    flip detail: f32 top1=%d (%.6f) vs q6 pick=%d (%.6f) -> f32 margin %.3e ; "
                    "q6 says %d (%.6f) over %d (%.6f), margin %.3e\n",
                    a2,(double)yr[a2],a1,(double)yr[a1],(double)(yr[a2]-yr[a1]),
                    a1,(double)yq[a1],a2,(double)yq[a2],(double)(yq[a1]-yq[a2]));
        if(rel>worst) worst=rel;
        printf("  trial %d: relL1=%.6f max|d|=%.3e cos(f64)=%.10f argmax q6=%d f32=%d %s\n",
               t,rel,mx,cos,a1,a2,a1==a2?"":"<-- FLIP");
    }
    printf("argmax agreement %d/%d | worst relL1 %.6f\n",agree,trials,worst);

    /* ---- phase breakdown of the head matvec ---- */
    {
        const int REP=10;
        double t0,f32_ms=0,i8_ms=0;
        for(int r=0;r<REP;r++){ t0=now_ms(); nt_qmatvec(yq,Q,14,x,O,I);    f32_ms+=now_ms()-t0; }
        for(int r=0;r<REP;r++){ t0=now_ms(); nt_qmatvec_i8(yq,Q,14,x,O,I); i8_ms+=now_ms()-t0; }
        f32_ms/=REP; i8_ms/=REP;
        double sp6=spawn_cost_ms(6,200), sp1=spawn_cost_ms(1,200);
        const char *floor_env=getenv("NT_QMV_THREAD_MIN");
        printf("\nHEAD PHASE BREAKDOWN (O=%d I=%d, %d reps, NT_QMV_THREAD_MIN=%s)\n",
               O,I,REP, floor_env?floor_env:"default 4M");
        printf("  nt_qmatvec    (f32 act) : %7.3f ms  -> %5.2f GB/s\n", f32_ms, (double)nb/(f32_ms/1e3)/1e9);
        printf("  nt_qmatvec_i8 (int8 act): %7.3f ms  -> %5.2f GB/s\n", i8_ms,  (double)nb/(i8_ms/1e3)/1e9);
        printf("  pthread spawn+join x6   : %7.3f ms  (%.1f%% of the i8 call)\n", sp6, 100.0*sp6/i8_ms);
        printf("  pthread spawn+join x1   : %7.3f ms\n", sp1);
        printf("  activation quant k=%d   : below timer resolution here; it is O(k)=%d ops\n", I, I);
    }
    printf("nt_qmatvec head cost: mean %.2f ms  min %.2f ms  -> %.2f GB/s effective over %lld B\n",
           tsum/trials, tmin, (double)nb/(tmin/1e3)/1e9, (long long)nb);
    return 0;
}
