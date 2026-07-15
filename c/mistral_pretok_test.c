/* mistral_pretok_test.c — isolate & verify the Mistral (Llama-3-style) pre-tokenizer
 * Split pattern against the tokenizers reference. Splits stdin lines into pieces and
 * prints each piece's [start,end) byte offsets; a Python golden compares.
 *
 * Mistral Split regex (7 alternatives, tried in order):
 *   1: [^\r\n\p{L}\p{N}]? [upgrp]* [logrp]+
 *   2: [^\r\n\p{L}\p{N}]? [upgrp]+ [logrp]*        (upgrp=[Lu Lt Lm Lo M], logrp=[Ll Lm Lo M])
 *   3: \p{N}                                        (single digit)
 *   4:  ?[^\s\p{L}\p{N}]+[\r\n/]*
 *   5: \s*[\r\n]+
 *   6: \s+(?!\S)
 *   7: \s+
 *
 * Build: cc -O2 -Wall -o mistral_pretok_test mistral_pretok_test.c -lm
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "tok_unicode.h"

static int u8_next(const unsigned char *s, int len, int i, uint32_t *cp){
    unsigned char c=s[i];
    if(c<0x80){ *cp=c; return 1; }
    if((c>>5)==0x6 && i+1<len){ *cp=((c&0x1F)<<6)|(s[i+1]&0x3F); return 2; }
    if((c>>4)==0xE && i+2<len){ *cp=((c&0x0F)<<12)|((s[i+1]&0x3F)<<6)|(s[i+2]&0x3F); return 3; }
    if((c>>3)==0x1E && i+3<len){ *cp=((c&0x07)<<18)|((s[i+1]&0x3F)<<12)|((s[i+2]&0x3F)<<6)|(s[i+3]&0x3F); return 4; }
    *cp=c; return 1;
}
#define ISNL(c) ((c)=='\r'||(c)=='\n')

/* alt1/alt2 letter matcher on codepoint array cp[0..n).
 * alt2=0 -> [upgrp]*[logrp]+ ; alt2=1 -> [upgrp]+[logrp]*. Returns end index (>j) or -1. */
static int match_letters_core(const uint32_t *cp, int j, int n, int alt2){
    int u=j; while(u<n && is_upgrp(cp[u])) u++;         /* max [upgrp] run [j,u) */
    int min_up = alt2 ? 1 : 0;
    /* greedy [upgrp]*: try largest up first, backtrack until [logrp] part valid */
    for(int up=u-j; up>=min_up; up--){
        int k=j+up;
        int lo=k; while(lo<n && is_logrp(cp[lo])) lo++;  /* [logrp] run from k */
        int min_lo = alt2 ? 0 : 1;                       /* alt1 needs logrp+ */
        if(lo-k>=min_lo && lo>j) return lo;              /* first valid (largest up) */
    }
    return -1;
}

/* full letter alternative (prefix? then alt1 or alt2). Returns end index or -1. */
static int match_letters(const uint32_t *cp, int i, int n){
    int j=i, pre=0;
    if(i<n && !ISNL(cp[i]) && !is_L(cp[i]) && !is_N(cp[i])){
        /* optional [^\r\n\p{L}\p{N}] prefix, only if a letter/mark follows */
        if(i+1<n && (is_upgrp(cp[i+1])||is_logrp(cp[i+1]))){ j=i+1; pre=1; }
        else return -1;
    }
    if(j>=n || !(is_upgrp(cp[j])||is_logrp(cp[j]))) return pre? -1 : -1;
    int e1=match_letters_core(cp,j,n,0);   /* alt1 first (regex order) */
    if(e1>j) return e1;
    int e2=match_letters_core(cp,j,n,1);   /* alt2 */
    if(e2>j) return e2;
    return -1;
}

/* one pre-token starting at cp index i; returns end index (exclusive). */
static int next_piece(const uint32_t *cp, int i, int n){
    uint32_t c=cp[i];
    /* alt 1&2: letters (with optional prefix) */
    int le=match_letters(cp,i,n); if(le>i) return le;
    /* alt 3: single digit \p{N} */
    if(is_N(c)) return i+1;
    /* alt 4:  ?[^\s\p{L}\p{N}]+[\r\n/]* */
    { int j=i;
      if(c==' ' && j+1<n && !is_S(cp[j+1]) && !is_L(cp[j+1]) && !is_N(cp[j+1])) j++;
      if(j<n && !is_S(cp[j]) && !is_L(cp[j]) && !is_N(cp[j])){
        while(j<n && !is_S(cp[j]) && !is_L(cp[j]) && !is_N(cp[j])) j++;
        while(j<n && (ISNL(cp[j])||cp[j]=='/')) j++;
        return j;
      }
    }
    /* alt 5: \s*[\r\n]+  (run of ws up to & incl the last contiguous newline) */
    { int r=i; while(r<n && is_S(cp[r])) r++;
      if(r>i){ int last=-1; for(int j=i;j<r;j++) if(ISNL(cp[j])) last=j;
        if(last>=0) return last+1;
        /* alt 6: \s+(?!\S) trailing ws ; alt 7: \s+ */
        int end=(r<n)?r-1:r; if(end<=i) end=i+1; return end;
      }
    }
    return i+1;  /* safety */
}

int main(void){
    char line[65536];
    while(fgets(line,sizeof line,stdin)){
        int len=(int)strlen(line);
        if(len>0 && line[len-1]=='\n') line[--len]=0;   /* strip trailing newline of the input framing */
        const unsigned char *p=(const unsigned char*)line;
        /* codepoint array + byte offset of each cp */
        uint32_t *cp=malloc((len+1)*sizeof(uint32_t)); int *boff=malloc((len+2)*sizeof(int)); int n=0;
        for(int i=0;i<len;){ uint32_t c; int k=u8_next(p,len,i,&c); boff[n]=i; cp[n]=c; n++; i+=k; }
        boff[n]=len;
        int i=0; int first=1;
        while(i<n){ int e=next_piece(cp,i,n); if(e<=i) e=i+1;
            if(!first) printf(" ");
            printf("%d,%d", boff[i], boff[e]);   /* byte [start,end) */
            first=0; i=e;
        }
        printf("\n");
        free(cp); free(boff);
    }
    return 0;
}
