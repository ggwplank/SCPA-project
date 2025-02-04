#include <stdio.h>
#include <stdlib.h>
#include "reader.h"
#include "csr.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Uso: %s <file_matrix_market>\n", argv[0]);
        return 1;
    }

    int M, N, NZ;
    MatrixEntry *entries;

    read_matrix_market(argv[1], &M, &N, &NZ, &entries);
    printf("Matrice %dx%d, nonzeri: %d\n", M, N, NZ);
    
    CSRMatrix *A = convert_to_CSR(M, N, NZ, entries);
    
    printf("Formato CSR:\nValori: ");
    for (int i = 0; i < A->nnz; i++) printf("%lf ", A->values[i]);

    printf("\nColonne: ");
    for (int i = 0; i < A->nnz; i++) printf("%d ", A->col_indices[i]);

    printf("\nRow Ptr: ");
    for (int i = 0; i <= A->rows; i++) printf("%d ", A->row_ptr[i]);

    printf("\n");

    // Allocare e inizializzare il vettore x
    double *x = (double *)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++)
        x[i] = 1.0; // Esempio: vettore di tutti 1

    // Allocare il vettore risultato y
    double *y = (double *)malloc(M * sizeof(double));

    // Calcolare il prodotto matrice-vettore
    csr_matrix_vector_multiply(A, x, y);

    // Stampare il risultato
    printf("Risultato del prodotto matrice-vettore:\n");
    for (int i = 0; i < M; i++)
        printf("%lf ", y[i]);
    
    printf("\n");

    // Liberare la memoria
    free_CSR(A);
    free(entries);
    free(x);
    free(y);

    return 0;
}