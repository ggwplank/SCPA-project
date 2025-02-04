#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include "reader.h"

void read_matrix_market(const char *filename, int *M, int *N, int *NZ, MatrixEntry **entries) {
    FILE *f;
    MM_typecode matcode;
    int i;

    // apertura file
    if ((f = fopen(filename, "r")) == NULL) {
        fprintf(stderr, "Errore: impossibile aprire il file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // lettura banner
    if (mm_read_banner(f, &matcode) != 0) {
        fprintf(stderr, "Errore: il file non è un Matrix Market valido.\n");
        exit(EXIT_FAILURE);
    }

    if (!mm_is_matrix(matcode) || !mm_is_coordinate(matcode)) {
        fprintf(stderr, "Errore: supportate solo matrici sparse in formato coordinate.\n");
        exit(EXIT_FAILURE);
    }

    // lettura dimensione matrice
    if (mm_read_mtx_crd_size(f, M, N, NZ) != 0) {
        fprintf(stderr, "Errore nella lettura della dimensione della matrice.\n");
        exit(EXIT_FAILURE);
    }

    // allocazione memoria per gli elementi
    *entries = (MatrixEntry *)malloc((*NZ) * sizeof(MatrixEntry));
    if (*entries == NULL) {
        fprintf(stderr, "Errore di allocazione memoria.\n");
        exit(EXIT_FAILURE);
    }

    // lettura elementi della matrice
    for (i = 0; i < *NZ; i++) {
        int row, col;

        double value = 1.0;  // valore di default per matrici "pattern"

        if (mm_is_pattern(matcode))
            fscanf(f, "%d %d", &row, &col);
        else
            fscanf(f, "%d %d %lf", &row, &col, &value);

        // la numerazione delle righe e colonne parte da 1 quindi occorre decrementarle
        (*entries)[i].row = row - 1;  
        (*entries)[i].col = col - 1;
        (*entries)[i].value = value;
    }

    fclose(f);

    // ricostruzione parte mancante se matrice è simmetrica
    if (mm_is_symmetric(matcode)) {
        int newNZ = (*NZ) * 2;

        MatrixEntry *expandedEntries = (MatrixEntry *)malloc(newNZ * sizeof(MatrixEntry));
        if (expandedEntries == NULL) {
            fprintf(stderr, "Errore di allocazione memoria per matrice simmetrica.\n");
            exit(EXIT_FAILURE);
        }

        int count = 0;
        for (i = 0; i < *NZ; i++) {
            expandedEntries[count++] = (*entries)[i];  // copia elemento originale
            if ((*entries)[i].row != (*entries)[i].col) {  // evitiamo di duplicare la diagonale
                expandedEntries[count].row = (*entries)[i].col;
                expandedEntries[count].col = (*entries)[i].row;
                expandedEntries[count].value = (*entries)[i].value;
                count++;
            }
        }

        free(*entries);

        *entries = expandedEntries;
        *NZ = count;
    }
}