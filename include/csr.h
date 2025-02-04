#ifndef CSR_H
#define CSR_H

#include "reader.h" // Include MatrixEntry

typedef struct {
    int rows;        // Numero di righe
    int cols;        // Numero di colonne
    int nnz;         // Numero di elementi non nulli
    double *values;  // Array dei valori non nulli
    int *col_indices; // Indici delle colonne
    int *row_ptr;    // Puntatori alle righe
} CSRMatrix;

// Funzione per convertire un array di elementi in formato CSR
CSRMatrix* convert_to_CSR(int M, int N, int NZ, MatrixEntry *entries);

// Funzione per liberare la memoria della matrice CSR
void free_CSR(CSRMatrix *A);

// Funzione per il prodotto matrice-vettore
void csr_matrix_vector_multiply(CSRMatrix *A, double *x, double *y);

#endif