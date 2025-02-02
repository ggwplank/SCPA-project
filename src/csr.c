#include <stdio.h>
#include <stdlib.h>
#include "csr.h"

// Funzione di confronto per ordinare gli elementi per riga e poi per colonna
int compare_entries(const void *a, const void *b) {
    MatrixEntry *entryA = (MatrixEntry *)a;
    MatrixEntry *entryB = (MatrixEntry *)b;
    
    if (entryA->row == entryB->row) {
        return entryA->col - entryB->col;
    }
    return entryA->row - entryB->row;
}

// Converte una matrice in formato MatrixEntry in formato CSR
CSRMatrix* convert_to_CSR(int M, int N, int NZ, MatrixEntry *entries) {
    qsort(entries, NZ, sizeof(MatrixEntry), compare_entries); // Ordina gli elementi

    CSRMatrix *A = (CSRMatrix *)malloc(sizeof(CSRMatrix));
    if (!A) {
        printf("Errore di allocazione per la matrice CSR\n");
        exit(1);
    }

    A->rows = M;
    A->cols = N;
    A->nnz = NZ;
    A->values = (double *)malloc(NZ * sizeof(double));
    A->col_indices = (int *)malloc(NZ * sizeof(int));
    A->row_ptr = (int *)malloc((M + 1) * sizeof(int));

    if (!A->values || !A->col_indices || !A->row_ptr) {
        printf("Errore di allocazione per gli array CSR\n");
        exit(1);
    }

    for (int i = 0; i <= M; i++) A->row_ptr[i] = 0; // Inizializza row_ptr

    for (int i = 0; i < NZ; i++) {
        A->row_ptr[entries[i].row]++;
    }

    int sum = 0;
    for (int i = 0; i < M; i++) {
        int temp = A->row_ptr[i];
        A->row_ptr[i] = sum;
        sum += temp;
    }
    A->row_ptr[M] = NZ;

    for (int i = 0; i < NZ; i++) {
        int row = entries[i].row;
        int index = A->row_ptr[row];
        A->col_indices[index] = entries[i].col;
        A->values[index] = entries[i].value;
        A->row_ptr[row]++;
    }

    for (int i = M; i > 0; i--) {
        A->row_ptr[i] = A->row_ptr[i - 1];
    }
    A->row_ptr[0] = 0;

    return A;
}

// Libera la memoria della matrice CSR
void free_CSR(CSRMatrix *A) {
    if (A) {
        free(A->values);
        free(A->col_indices);
        free(A->row_ptr);
        free(A);
    }
}