#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"

typedef struct {
    int row;
    int col;
    double value;
} MatrixEntry;

void read_matrix_market(const char *filename, int *M, int *N, int *NZ, MatrixEntry **entries) {
    FILE *f;
    MM_typecode matcode;
    int i;

    // Apri il file
    if ((f = fopen(filename, "r")) == NULL) {
        printf("Errore: impossibile aprire il file %s\n", filename);
        exit(1);
    }

    // Leggi il banner
    if (mm_read_banner(f, &matcode) != 0) {
        printf("Errore: il file non è un Matrix Market valido.\n");
        exit(1);
    }

    // Controlla che sia una matrice
    if (!mm_is_matrix(matcode) || !mm_is_coordinate(matcode)) {
        printf("Errore: supportate solo matrici sparse in formato coordinate.\n");
        exit(1);
    }

    // Leggi dimensione della matrice
    if (mm_read_mtx_crd_size(f, M, N, NZ) != 0) {
        printf("Errore nella lettura della dimensione della matrice.\n");
        exit(1);
    }

    // Alloca memoria per gli elementi
    *entries = (MatrixEntry *)malloc((*NZ) * sizeof(MatrixEntry));
    if (*entries == NULL) {
        printf("Errore di allocazione memoria.\n");
        exit(1);
    }

    // Leggi gli elementi della matrice
    for (i = 0; i < *NZ; i++) {
        int row, col;
        double value = 1.0;  // Default per "pattern"

        if (mm_is_pattern(matcode)) {
            fscanf(f, "%d %d", &row, &col);
        } else {
            fscanf(f, "%d %d %lf", &row, &col, &value);
        }

        (*entries)[i].row = row;
        (*entries)[i].col = col;
        (*entries)[i].value = value;
    }

    fclose(f);

    if (mm_is_pattern(matcode)) {
        printf("pure chaosssssssss");
    }

    // Se la matrice è simmetrica, dobbiamo duplicare il triangolo
    if (mm_is_symmetric(matcode)) {
        int newNZ = *NZ * 2;  // Stima approssimativa (potremmo avere elementi sulla diagonale)
        MatrixEntry *expandedEntries = (MatrixEntry *)malloc(newNZ * sizeof(MatrixEntry));
        printf("behold the god of thunderrrrrrrrrrrrrrrrr");
        if (expandedEntries == NULL) {
            printf("Errore di allocazione memoria per matrice simmetrica.\n");
            exit(1);
        }

        int count = 0;
        for (i = 0; i < *NZ; i++) {
            expandedEntries[count++] = (*entries)[i];  // Copia l'elemento originale
            if ((*entries)[i].row != (*entries)[i].col) {  // Evita di duplicare la diagonale
                expandedEntries[count].row = (*entries)[i].col;
                expandedEntries[count].col = (*entries)[i].row;
                expandedEntries[count].value = (*entries)[i].value;
                count++;
            }
        }

        free(*entries);
        *entries = expandedEntries;
        *NZ = count;  // Aggiorna il numero totale di nonzeri
    }
}



int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Uso: %s <file_matrix_market>\n", argv[0]);
        return 1;
    }

    int M, N, NZ;
    MatrixEntry *entries;

    read_matrix_market(argv[1], &M, &N, &NZ, &entries);

    printf("Matrice %dx%d, nonzeri: %d\n", M, N, NZ);
    
    /*
    for (int i = 0; i < NZ; i++) {
        printf("%d %d %lf\n", entries[i].row, entries[i].col, entries[i].value);
    }
    */
    

    free(entries);
    return 0;
}