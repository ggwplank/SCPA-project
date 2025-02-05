#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "reader.h"
#include "csr.h"
#include "matrix_operations.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Uso: %s <file_matrix_market>\n", argv[0]);
        return 1;
    }

    const char *matrix_filename = argv[1];  // --> dir/file.mtx
    const char *matrix_name = strrchr(matrix_filename, '/'); // --> /file.mtx
    matrix_name = (matrix_name) ? matrix_name + 1 : matrix_filename; // --> file.mtx

    int M, N, NZ;
    MatrixEntry *entries;
    CSRMatrix *A;

    read_and_convert_matrix(matrix_filename, &A, &M, &N, &NZ, &entries);
    
    double *x = NULL;
    generate_vector(matrix_name, M, &x);
    
    // printf("Formato CSR:\nValori: ");
    // for (int i = 0; i < A->nnz; i++) printf("%lf ", A->values[i]);

    // printf("\nColonne: ");
    // for (int i = 0; i < A->nnz; i++) printf("%d ", A->col_indices[i]);

    // printf("\nRow Ptr: ");
    // for (int i = 0; i <= A->rows; i++) printf("%d ", A->row_ptr[i]);

    multiply_and_compare(A, x, M);


    free_CSR(A);
    free(entries);
    free(x);
    
    return 0;
}