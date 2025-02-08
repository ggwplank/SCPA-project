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
    int num_blocks = (M + hack_size - 1) / hack_size;  // Arrotonda in eccesso
    HLLMatrix *hll = (HLLMatrix*)malloc(sizeof(HLLMatrix));
    hll->num_blocks = num_blocks;
    hll->hack_size = hack_size;
    hll->blocks = (ELLPackMatrix**)malloc(num_blocks * sizeof(ELLPackMatrix*));
    
    // Converti ogni blocco in formato ELLPack
    for (int b = 0; b < num_blocks; b++) {
        int start_row = b * hack_size;
        int block_rows = (start_row + hack_size <= M) ? hack_size : (M - start_row);

        // Creiamo una lista di entries per il blocco, allocata solo con il numero effettivo di non-zero entries
        MatrixEntry *block_entries = (MatrixEntry*)malloc(NZ * sizeof(MatrixEntry));
        int block_nz = 0;
        
        // Aggiungi entries al blocco
        for (int i = 0; i < NZ; i++) {
            if (entries[i].row >= start_row && entries[i].row < start_row + block_rows) {
                block_entries[block_nz++] = (MatrixEntry){
                    .row = entries[i].row - start_row, // Shift per il nuovo blocco
                    .col = entries[i].col,
                    .value = entries[i].value
                };
            }
        }

        // Se ci sono elementi nel blocco, convertili in formato ELL
        if (block_nz > 0) {
            hll->blocks[b] = convert_to_ELL(block_rows, N, block_nz, block_entries);
            // transpose_ELLPack(hll->blocks[b]);  // Trasposta del blocco, vedi main
        } else {
            hll->blocks[b] = NULL;  // Se il blocco è vuoto, settiamo il puntatore a NULL
        }

        free(block_entries);  // Libera la memoria per block_entries
    }
    return hll;
}


void free_HLL(HLLMatrix *H) {
    for (int b = 0; b < H->num_blocks; b++) {
        free_ELL(H->blocks[b]);
    }
    free(H->blocks);
}

void print_HLL(HLLMatrix *H) {
    printf("HLL Matrix: %d blocchi, hack_size = %d\n", H->num_blocks, H->hack_size);

    // voglio stampare quante sono le righe nell'ultimo blocco
    int last_block_rows = H->blocks[H->num_blocks - 1]->rows;
    printf("Ultimo blocco: %d righe\n", last_block_rows);

    // for (int b = 0; b < H->num_blocks; b++) {
    //     printf("\nBlocco %d:\n", b);
    //     print_ELL(H->blocks[b]);
    // }
}

// idea: si potrebbe creare un kernel cuda che faccia questo lavoro da solo, con un for si calcola un blocco alla volta e questo ci rallenta molto
// possiamo creare tutti quanti i pezzi qui e poi passarli al kernel per fare il ciclo direttamente da li dentro
void matvec_hll_cuda(HLLMatrix *H, double *x, double *y) {
    for (int b = 0; b < H->num_blocks; b++) {
        ELLPackMatrix *block = H->blocks[b];
        if (block == NULL) continue;  // Se il blocco è vuoto, salta questo blocco
        int block_rows = block->rows;
        int start_row = b * H->hack_size;
        double *block_y = (double*)malloc(block_rows * sizeof(double));

        // Computa il prodotto matrice-vettore per il blocco
        matvec_ellpack_cuda(block, x, block_y);

        // Aggiungi i risultati del blocco nel vettore y
        for (int i = 0; i < block_rows; i++) {
            y[start_row + i] += block_y[i];
        }

        free(block_y);
    }
}

