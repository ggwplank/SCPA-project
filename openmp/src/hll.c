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

        hll->blocks[b] = convert_to_ELL(block_rows, N, block_nz, block_entries);
        free(block_entries);
    }
    return hll;
}

void omp_hll_mult(HLLMatrix *H, double *x, double *y) {
    #pragma omp parallel for
    for (int b = 0; b < H->num_blocks; b++) {
        ELLPackMatrix *block = H->blocks[b];
        int start_row = b * H->hack_size;

        for (int i = 0; i < block->rows; i++) {
            double sum = 0.0;

            for (int j = 0; j < block->maxnz; j++) {
                if (block->col_indices[i][j] != -1)
                    sum += block->values[i][j] * x[block->col_indices[i][j]];
                
            }
            y[start_row + i] = sum;
        }
    }
}
// NOTA: Iteriamo sui blocchi, non sulle righe come facevamo con ELLPack

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
