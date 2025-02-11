#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "utils.h"

ELLPackMatrix* convert_to_ELL(int M, int N, int NZ, MatrixEntry *entries) {
    ELLPackMatrix *ellpack = (ELLPackMatrix*)malloc(sizeof(ELLPackMatrix));
    ellpack->rows = M;
    ellpack->cols = N;
    ellpack->nnz = NZ;
    
    // NZ per ogni riga
    int *row_counts = (int*)calloc(M, sizeof(int));
    if (row_counts == NULL) {
        perror("Errore nell'allocazione di memoria per row_counts");
        exit(1);
    }

    for (int i = 0; i < NZ; i++)
        row_counts[entries[i].row]++;

    // lunghezza massima per le righe
    int max_row_length = 0;
    for (int i = 0; i < M; i++) {
        if (row_counts[i] > max_row_length)
            max_row_length = row_counts[i];
    }
    ellpack->maxnz = max_row_length;

    ellpack->values = (double**)malloc(M * sizeof(double*));
    if (ellpack->values == NULL) {
        perror("Errore nell'allocazione di memoria per ellpack->values");
        exit(1);
    }

    ellpack->col_indices = (int**)malloc(M * sizeof(int*));
    if (ellpack->col_indices == NULL) {
        perror("Errore nell'allocazione di memoria per ellpack->col_indices");
        exit(1);
    }

    for (int i = 0; i < M; i++) {
        ellpack->values[i] = (double *)malloc(max_row_length * sizeof(double));
        if (ellpack->values[i] == NULL) {
            perror("Errore nell'allocazione di memoria per ellpack->values[i]");
            exit(1);
        }

        ellpack->col_indices[i] = (int *)malloc(max_row_length * sizeof(int));
        if (ellpack->col_indices[i] == NULL) {
            perror("Errore nell'allocazione di memoria per ellpack->col_indices[i]");
            exit(1);
        }
        
        for (int j = 0; j < max_row_length; j++) {
            ellpack->values[i][j] = 0.0;
            ellpack->col_indices[i][j] = -1;
        }
    }

    // inserimento elementi nella matrice ELL
    int *row_counts_insert = (int*)calloc(M, sizeof(int));
    if (row_counts_insert == NULL) {
        perror("Errore nell'allocazione di memoria per row_counts_insert");
        exit(1);
    }

    for (int i = 0; i < NZ; i++) {
        int row = entries[i].row;
        int col = entries[i].col;
        double value = entries[i].value;;

        // posizione giusta per ogni elemento nella riga
        int idx = row_counts_insert[row];
        ellpack->values[row][idx] = value;
        ellpack->col_indices[row][idx] = col;
        row_counts_insert[row]++;
    }

    free(row_counts);
    free(row_counts_insert);

    return ellpack;
}

void free_ELL(ELLPackMatrix *A) {
    for (int i = 0; i < A->rows; i++) {
        free(A->values[i]);
        free(A->col_indices[i]);
    }

    free(A->values);
    free(A->col_indices);
    free(A);
}
