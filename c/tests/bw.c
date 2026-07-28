// bw.c - pure streaming-read bandwidth from warm page cache.
// usage: bw <threads> <reps> <file> [file ...]
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>

#ifndef MAP_POPULATE
#define MAP_POPULATE 0   /* absent on macOS/BSD: the first timed pass populates anyway */
#endif

#define MAXF 32
#define MAXT 16

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static const uint64_t *base[MAXF];
static size_t words[MAXF];
static int nf = 0, nt = 1;

typedef struct { int id; uint64_t acc; } job;

static void *work(void *a) {
    job *j = (job *)a;
    uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int f = 0; f < nf; f++) {
        size_t n = words[f];
        size_t lo = (size_t)((double)n * j->id / nt);
        size_t hi = (size_t)((double)n * (j->id + 1) / nt);
        const uint64_t *p = base[f];
        size_t i = lo;
        for (; i + 4 <= hi; i += 4) { s0 += p[i]; s1 += p[i+1]; s2 += p[i+2]; s3 += p[i+3]; }
        for (; i < hi; i++) s0 += p[i];
    }
    j->acc = s0 + s1 + s2 + s3;
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: bw <threads> <reps> <file>...\n"); return 2; }
    nt = atoi(argv[1]); int reps = atoi(argv[2]);
    if (nt < 1 || nt > MAXT) return 2;
    size_t total = 0;
    for (int i = 3; i < argc && nf < MAXF; i++, nf++) {
        int fd = open(argv[i], O_RDONLY);
        if (fd < 0) { perror(argv[i]); return 1; }
        struct stat st; fstat(fd, &st);
        size_t len = (size_t)st.st_size & ~(size_t)7;
        void *m = mmap(NULL, len, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
        if (m == MAP_FAILED) { perror("mmap"); return 1; }
        madvise(m, len, MADV_WILLNEED);
        close(fd);
        base[nf] = (const uint64_t *)m; words[nf] = len / 8; total += len;
    }
    printf("files=%d bytes=%zu (%.3f GiB) threads=%d\n", nf, total, total / 1073741824.0, nt);
    pthread_t th[MAXT]; job jb[MAXT];
    double best = 0;
    for (int r = 0; r < reps; r++) {
        uint64_t sink = 0;
        double t0 = now();
        for (int t = 0; t < nt; t++) { jb[t].id = t; pthread_create(&th[t], NULL, work, &jb[t]); }
        for (int t = 0; t < nt; t++) { pthread_join(th[t], NULL); sink += jb[t].acc; }
        double dt = now() - t0;
        double gbs = total / dt / 1e9;
        if (gbs > best) best = gbs;
        printf("rep %d: %.4f s  %.3f GB/s  (sink %016llx)\n", r, dt, gbs, (unsigned long long)sink);
        fflush(stdout);
    }
    printf("BEST threads=%d %.3f GB/s\n", nt, best);
    return 0;
}
