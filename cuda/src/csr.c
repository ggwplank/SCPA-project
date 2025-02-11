#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "utils.h"

// nel CSR gli elementi devono essere memorizzati riga per riga
int compare_entries(const void *a, const void *b) {
    MatrixEntry *entryA = (MatrixEntry *)a;
    MatrixEntry *entryB = (MatrixEntry *)b;
    
    // se le due celle appartengono alla stessa riga, le ordiniamo per colonna
    if (entryA->row == entryB->row)
        return entryA->col - entryB->col;
    
    // altrimenti le ordiniamo per riga
    return entryA->row - entryB->row;
}

// Funzione che converte una matrice da formato MatrixEntry (r, c, v) in formato CSR
CSRMatrix* convert_to_CSR(int M, int N, int NZ, MatrixEntry *entries) {
    qsort(entries, NZ, sizeof(MatrixEntry), compare_entries);

    CSRMatrix *A = (CSRMatrix *)malloc(sizeof(CSRMatrix));
    if (!A) {
        printf("Errore di allocazione per la matrice CSR\n");
        free_CSR(A);
        exit(1);
    }

    A->rows = M;
    A->cols = N;
    A->nnz = NZ;

    A->values = (double *)malloc(NZ * sizeof(double));  // memorizza i valori non nulli
    A->col_indices = (int *)malloc(NZ * sizeof(int));   // indici di colonna dei valori non nulli
    A->row_ptr = (int *)malloc((M + 1) * sizeof(int));  // offset di inizio di ogni riga

    if (!A->values || !A->col_indices || !A->row_ptr) {
        printf("Errore di allocazione per gli array CSR\n");
        exit(1);
    }

    // inizializza tutti gli elementi di row_ptr a 0
    for (int i = 0; i <= M; i++)
        A->row_ptr[i] = 0;  // row_ptr[i] conterrà quanti elementi non nulli ci sono nella riga i-esima

    // conta quanti elementi non nulli ci sono in ogni riga
    for (int i = 0; i < NZ; i++)
        A->row_ptr[entries[i].row]++;   // row_ptr[i] ora contiene il numero di elementi nella riga i

    // calcola gli offset di inizio di ogni riga
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

    // ripristina i valori di row_ptr
    for (int i = M; i > 0; i--)
        A->row_ptr[i] = A->row_ptr[i - 1];

    A->row_ptr[0] = 0;

    return A;
}

void serial_csr_mult(CSRMatrix *A, double *x, double *y) {
    for (int i = 0; i < A->rows; i++) {
        y[i] = 0.0;

        for (int j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++)
            y[i] += A->values[j] * x[A->col_indices[j]];
        
    }
}

void free_CSR(CSRMatrix *A) {
    if (A) {
        free(A->values);
        free(A->col_indices);
        free(A->row_ptr);
        free(A);
    }
}