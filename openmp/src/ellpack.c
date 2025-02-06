#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "utils.h"

ELLPackMatrix* convert_to_ELL(int M, int N, int NZ, MatrixEntry *entries) {
    // Crea la struttura ELLMatrix
    ELLPackMatrix *ellpack = (ELLPackMatrix*)malloc(sizeof(ELLPackMatrix));
    ellpack->rows = M;
    ellpack->cols = N;
    ellpack->nnz = NZ;
    
    // Calcola il numero di non-zeri per ogni riga
    int *row_counts = (int*)calloc(M, sizeof(int));
    for (int i = 0; i < NZ; i++) {
        row_counts[entries[i].row]++;
    }

    // Trova la lunghezza massima per le righe
    int max_row_length = 0;
    for (int i = 0; i < M; i++) {
        if (row_counts[i] > max_row_length) {
            max_row_length = row_counts[i];
        }
    }
    ellpack->maxnz = max_row_length;

    // Alloca memoria per la matrice ELL
    ellpack->values = (double**)malloc(M * sizeof(double*));
    ellpack->col_indices = (int**)malloc(M * sizeof(int*));
    for (int i = 0; i < M; i++) {
        ellpack->values[i] = (double *)malloc(max_row_length * sizeof(double));
        ellpack->col_indices[i] = (int *)malloc(max_row_length * sizeof(int));
        
        // Inizializza con zero le colonne non usate
        for (int j = 0; j < max_row_length; j++) {
            ellpack->values[i][j] = 0.0;
            ellpack->col_indices[i][j] = -1;
        }
    }

    // Inserisce gli elementi nella matrice ELL
    int *row_counts_insert = (int*)calloc(M, sizeof(int));
    for (int i = 0; i < NZ; i++) {
        int row = entries[i].row;
        int col = entries[i].col;
        double value = entries[i].value;;

        // Trova la posizione giusta per ogni elemento nella riga
        int idx = row_counts_insert[row];
        ellpack->values[row][idx] = value;
        ellpack->col_indices[row][idx] = col;
        row_counts_insert[row]++;
    }

    free(row_counts);
    free(row_counts_insert);

    return ellpack;
}

void omp_ellpack_mult(ELLPackMatrix *A, double *x, double *y) {
    int M = A->rows;
    int maxnz = A->maxnz;

    int **col_indices = A->col_indices;
    double **values = A->values;

    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        y[i] = 0.0;
        for (int j = 0; j < maxnz; j++) {
            if (col_indices[i][j] != -1) {
                y[i] += values[i][j] * x[col_indices[i][j]];
            }
        }
    }
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

void print_ELL(ELLPackMatrix *A) {
    printf("ELLPack Matrix: %d x %d (maxnz = %d)\n", A->rows, A->cols, A->maxnz);

    for (int i = 0; i < A->rows; i++) {
        printf("Riga %d: ", i);
        for (int j = 0; j < A->maxnz; j++) {
            printf("(%d, %.2f) ", A->col_indices[i][j], A->values[i][j]);
        }
        printf("\n");
    }
}
