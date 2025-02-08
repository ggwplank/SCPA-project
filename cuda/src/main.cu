#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "utils.h"


int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Uso: %s <file_matrix_market> <-serial/-cudaCSR/-cudaHLL>\n", argv[0]);
        return 1;
    }

    const char *matrix_filename = argv[1];  // --> dir/file.mtx
    const char *matrix_name = strrchr(matrix_filename, '/'); // --> /file.mtx
    matrix_name = (matrix_name) ? matrix_name + 1 : matrix_filename; // --> file.mtx

    const char *mode = argv[2];

    int M, N, NZ;
    MatrixEntry *entries;

    read_matrix_market(matrix_filename, &M, &N, &NZ, &entries);
    printf("Matrice %dx%d, nonzeri: %d\n", M, N, NZ);

    CSRMatrix *A = convert_to_CSR(M, N, NZ, entries);

    ELLPackMatrix *A_ellpack = convert_to_ELL(M, N, NZ, entries);
    
    double *x = NULL;
    generate_random_vector(matrix_name, M, &x);
    
    // printf("Formato CSR:\nValori: ");
    // for (int i = 0; i < A->nnz; i++) printf("%lf ", A->values[i]);

    // printf("\nColonne: ");
    // for (int i = 0; i < A->nnz; i++) printf("%d ", A->col_indices[i]);

    // printf("\nRow Ptr: ");
    // for (int i = 0; i <= A->rows; i++) printf("%d ", A->row_ptr[i]);
 
    double *y_serial = allocate_result(M);
    serial_csr_mult(A, x, y_serial);

    if (strcmp(mode, "-serial") == 0) {
        printf("Primi 10 valori risultato moltiplicazione seriale: ");
        for (int i = 0; i < 10; i++) 
            printf("%f ", y_serial[i]);
        printf("\n");

        free(y_serial);
    }

    else if (strcmp(mode, "-cudaCSR") == 0) {
        double *y_cuda_csr = allocate_result(M);

        printf("Eseguo il prodotto matrice-vettore con CUDA...\n");
        cuda_csr_mult(A, x, y_cuda_csr);

        compare_results(y_serial, y_cuda_csr, M);
    }

    else if (strcmp(mode, "-cudaHLL") == 0) {
        double *y_cuda_hll = allocate_result(M);

        int hack_size = 32; // dobbiamo riproporzionare questo valore altrimenti andiamo fuori memoria tipo con 3000 amazon va, con 32 no =^(
        ELLPackMatrix *A_ellpack_hack = convert_to_ELL(M, N, NZ, entries);

        // HLLMatrix *A_hll = convert_to_HLL(M, N, NZ, entries, hack_size);
        // print_HLL(A_hll);
        //print_ELL(A_ellpack_hack);
        //transpose_ELLPack(A_ellpack_hack);
        print_ELL(A_ellpack_hack);

        matvec_ellpack_cuda(A_ellpack_hack, x, y_cuda_hll);

        compare_results(y_serial, y_cuda_hll, M);

        free_ELL(A_ellpack_hack);
    }

    else {
        printf("Le possibili modalità sono: -serial, -cudaCSR, -cudaHLL\n");
    }

    free_CSR(A);
    free_ELL(A_ellpack);
    free(entries);
    free(x);
    
    return 0;
}