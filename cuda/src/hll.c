#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "utils.h"

/*
Divide la matrice in blocchi da hack_size righe.
Per ogni blocco, raccoglie i suoi elementi non nulli.
Converte ogni blocco in formato ELLPack usando convert_to_ELL().
Salva tutti i blocchi in un array blocks.
*/
HLLMatrix* convert_to_HLL(int M, int N, int NZ, MatrixEntry *entries, int hack_size) {
    int num_blocks = (M + hack_size - 1) / hack_size;

    HLLMatrix *hll = (HLLMatrix*)malloc(sizeof(HLLMatrix));
    hll->num_blocks = num_blocks;
    hll->hack_size = hack_size;
    hll->blocks = (ELLPackMatrix**)malloc(num_blocks * sizeof(ELLPackMatrix*));
    
    // Ogni blocco viene convertito in ELLPack
    for (int b = 0; b < num_blocks; b++) {
        int start_row = b * hack_size;
        int block_rows = (start_row + hack_size <= M) ? hack_size : (M - start_row);

        // sotto-matrice con solo le righe del blocco
        MatrixEntry *block_entries = (MatrixEntry*)malloc(NZ * sizeof(MatrixEntry));
        int block_nz = 0;
        

        for (int i = 0; i < NZ; i++) {
            if (entries[i].row >= start_row && entries[i].row < start_row + block_rows) {
                block_entries[block_nz++] = (MatrixEntry){
                    .row = entries[i].row - start_row,
                    .col = entries[i].col,
                    .value = entries[i].value
                };
            }
        }

        if (block_nz > 0) {
            hll->blocks[b] = convert_to_ELL(block_rows, N, block_nz, block_entries);
        } else {
            // Alloca un blocco vuoto
            ELLPackMatrix *emptyBlock = (ELLPackMatrix*) malloc(sizeof(ELLPackMatrix));
            emptyBlock->rows = block_rows;
            emptyBlock->cols = N;
            emptyBlock->nnz = 0;
            emptyBlock->maxnz = 1;
            emptyBlock->values = (double**) malloc(block_rows * sizeof(double*));
            emptyBlock->col_indices = (int**) malloc(block_rows * sizeof(int*));
            for (int i = 0; i < block_rows; i++) {
                emptyBlock->values[i] = (double*) malloc(sizeof(double));
                emptyBlock->col_indices[i] = (int*) malloc(sizeof(int));
                emptyBlock->values[i][0] = 0.0;
                emptyBlock->col_indices[i][0] = -1;
            }
            hll->blocks[b] = emptyBlock;
        }

        free(block_entries);
    }
    return hll;
}

void free_HLL(HLLMatrix *H) {
    for (int b = 0; b < H->num_blocks; b++)
        free_ELL(H->blocks[b]);

    free(H->blocks);
    free(H);
}

void print_HLL(HLLMatrix *H) {
    int last_block_rows = H->blocks[H->num_blocks - 1]->rows;

    printf("HLL Matrix: %d blocchi, hack_size = %d\n", H->num_blocks, H->hack_size);
    printf("Ultimo blocco: %d righe\n", last_block_rows);
}