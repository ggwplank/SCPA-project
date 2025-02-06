#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "utils.h"

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Uso: %s <file_matrix_market> <-serial/-ompCSR/-ompHLL> {num threads}\n", argv[0]);
        return 1;
    }

    const char *matrix_filename = argv[1];  // --> dir/file.mtx
    const char *matrix_name = strrchr(matrix_filename, '/'); // --> /file.mtx
    matrix_name = (matrix_name) ? matrix_name + 1 : matrix_filename; // --> file.mtx

    const char *mode = argv[2];

    int num_threads = (argc == 4) ? atoi(argv[3]) : omp_get_max_threads();

    int M, N, NZ;
    MatrixEntry *entries;

    read_matrix_market(matrix_filename, &M, &N, &NZ, &entries);
    printf("Matrice %dx%d, nonzeri: %d\n", M, N, NZ);

    CSRMatrix *A = convert_to_CSR(M, N, NZ, entries);

    //ELLMatrix *A;
    //read_and_convert_matrix_to_ellpack(matrix_filename, &A, &M, &N, &NZ, &entries);
    
    double *x = NULL;
    generate_random_vector(matrix_name, M, &x);

    // for (int i = 0; i < A->M; i++) {
    //     printf("Riga %d: ", i);
    //     for (int j = 0; j < A->NZ / A->M; j++) {  // Al massimo, ogni riga ha max_row_length non-zeri
    //         if (A->columns[i][j] != -1) {  // Se la colonna è valida (non -1)
    //             printf("(Colonna: %d, Valore: %f) ", A->columns[i][j], A->values[i][j]);
    //         }
    //     }
    //     printf("\n");
    // }


    // printf("Formato CSR:\nValori: ");
    // for (int i = 0; i < A->nnz; i++) printf("%lf ", A->values[i]);

    // printf("\nColonne: ");
    // for (int i = 0; i < A->nnz; i++) printf("%d ", A->col_indices[i]);

    // printf("\nRow Ptr: ");
    // for (int i = 0; i <= A->rows; i++) printf("%d ", A->row_ptr[i]);

    double *y_serial = allocate_result(M);
    serial_csr_mult(A, x, y_serial);

    if (strcmp(mode, "-serial") == 0) {
        // stampa dei primi valori del seriale
        printf("Primi 10 valori del seriale: ");
        for (int i = 0; i < 10; i++) 
            printf("%f ", y_serial[i]);
        printf("\n");
    }

    else if (strcmp(mode, "-ompCSR") == 0) {
        double *y_omp_csr = allocate_result(M);

        omp_set_num_threads(num_threads);
        omp_csr_mult(A, x, y_omp_csr);

        compare_results(y_serial, y_omp_csr, M);
    }

    else if (strcmp(mode, "-ompHLL") == 0) {
        double *y_omp_hll = allocate_result(M);

        omp_set_num_threads(num_threads);
        // omp_hll_mult(A, x, y_omp_hll);

        // compare_results(y_serial, y_omp_hll, M);
    }
    
    else
        printf("Le possibili modalità sono: -serial, -ompCSR, -ompHLL\n");
    
    free_CSR(A);
    free(entries);
    free(x);
    
    return 0;
}