/* qwen_tok_test — encode/decode harness for the Qwen BPE, built to be diffed against
 * llama-tokenize line by line.
 *
 *   qwen_tok_test <tokenizer.json> ids  <corpus>   one "[a, b, c]" line per input line,
 *                                                  byte-identical in shape to
 *                                                  `llama-tokenize --ids`
 *   qwen_tok_test <tokenizer.json> rt   <corpus>   round-trip: detok(tok(line)) vs line,
 *                                                  byte-exact, reports the first mismatch
 *   qwen_tok_test <tokenizer.json> info            dialect the loader read out of the file
 *
 * The corpus is read as raw lines; the trailing newline is not part of the line, which is
 * what the oracle sees when given the same line via -p.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "qwen_tok.h"

#define MAXIDS 65536

int main(int argc, char **argv){
    if(argc < 3){ fprintf(stderr,"usage: %s <tokenizer.json> ids|rt|info [corpus]\n",argv[0]); return 2; }
    QTok T; qtok_load(&T, argv[1]);
    const char *mode = argv[2];

    if(!strcmp(mode,"info")){
        printf("digit_run=%d ignore_merges=%d added_tokens=%d n_ids=%d\n",
               T.digit_run, T.ignore_merges, T.nsp, T.n_ids);
        printf("im_start=%d im_end=%d endoftext=%d\n",
               qtok_id_of(&T,"<|im_start|>"), qtok_id_of(&T,"<|im_end|>"), qtok_id_of(&T,"<|endoftext|>"));
        return 0;
    }
    if(argc < 4){ fprintf(stderr,"need a corpus file for mode %s\n",mode); return 2; }
    FILE *f=fopen(argv[3],"rb"); if(!f){ perror(argv[3]); return 1; }

    static int ids[MAXIDS];
    char *line=NULL; size_t cap=0; ssize_t nr;
    int lineno=0, bad=0;
    char *dec=malloc(1<<20);
    while((nr=getline(&line,&cap,f))>=0){
        lineno++;
        int len=(int)nr;
        while(len>0 && (line[len-1]=='\n' || line[len-1]=='\r')) len--;   /* oracle sees no EOL */
        int n=qtok_encode(&T,line,len,ids,MAXIDS);
        if(!strcmp(mode,"ids")){
            putchar('[');
            for(int i=0;i<n;i++) printf(i?", %d":"%d", ids[i]);
            printf("]\n");
        } else {
            int m=qtok_decode(&T,ids,n,dec,(1<<20)-1);
            if(m!=len || memcmp(dec,line,len)!=0){
                bad++;
                fprintf(stderr,"line %d ROUND-TRIP MISMATCH: %d bytes in, %d out\n",lineno,len,m);
                fprintf(stderr,"  in : [%.*s]\n  out: [%.*s]\n",len,line,m,dec);
            }
        }
    }
    fclose(f); free(line); free(dec);
    if(!strcmp(mode,"rt")) printf("round-trip: %d lines, %d mismatches -> %s\n",
                                  lineno, bad, bad? "FAIL":"byte-exact PASS");
    return bad? 2:0;
}
