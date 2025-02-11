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

        if (block_nz > 0)
            hll->blocks[b] = convert_to_ELL(block_rows, N, block_nz, block_entries);
        else
            hll->blocks[b] = NULL; 

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

// idea: si potrebbe creare un kernel cuda che faccia questo lavoro da solo, con un for si calcola un blocco alla volta e questo ci rallenta molto
// possiamo creare tutti quanti i pezzi qui e poi passarli al kernel per fare il ciclo direttamente da li dentro
// void matvec_hll_cuda(HLLMatrix *H, double *x, double *y) {
//     for (int b = 0; b < H->num_blocks; b++) {
//         ELLPackMatrix *block = H->blocks[b];
//         if (block == NULL) continue;  // Se il blocco è vuoto, salta questo blocco
//         int block_rows = block->rows;
//         int start_row = b * H->hack_size;
//         double *block_y = (double*)malloc(block_rows * sizeof(double));

//         // Computa il prodotto matrice-vettore per il blocco
//         matvec_ellpack_cuda(block, x, block_y);

//         // Aggiungi i risultati del blocco nel vettore y
//         for (int i = 0; i < block_rows; i++) {
//             y[start_row + i] += block_y[i];
//         }

//         free(block_y);
//     }
// }

