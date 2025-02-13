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

    const char *matrix_filename = argv[1];
    const char *matrix_name = strrchr(matrix_filename, '/');
    matrix_name = (matrix_name) ? matrix_name + 1 : matrix_filename;

    const char *mode = argv[2];

    int num_threads = (argc == 3) ? atoi(argv[2]) : omp_get_max_threads();

    int M, N, NZ;
    MatrixEntry *entries;

    printf("\n\nLettura della matrice %s...\n", matrix_name);
    read_matrix_market(matrix_filename, &M, &N, &NZ, &entries);
    printf("Matrice %dx%d, nonzeri: %d\n", M, N, NZ);

    printf("Conversione in formato CSR...\n");
    CSRMatrix *A = convert_to_CSR(M, N, NZ, entries);
    
    double *x = NULL;
    printf("Generazione del vettore randomico...\n");
    generate_random_vector(matrix_name, M, &x);

    double *y_serial = allocate_result(M);
    printf("Moltiplicazione seriale...\n");
    serial_csr_mult(A, x, y_serial);

    
    printf("Calcolo delle prestazioni per la moltiplicazione seriale...\n");
    get_performances_and_save((void (*)(void *, double *, double *))serial_csr_mult,
        A, x, y_serial,
        matrix_name, M, N, NZ,
        "-serial", 1, NULL);
    printf("Calcolo terminato.\n");


    double *y_omp_csr = allocate_result(M);
    omp_set_num_threads(num_threads);
    printf("Moltiplicazione parallela con CSR e %d thread...\n", num_threads);
    get_performances_and_save((void (*)(void *, double *, double *))omp_csr_mult,
        A, x, y_omp_csr,
        matrix_name, M, N, NZ,
        "-ompCSR", num_threads, y_serial);
    free(y_omp_csr);

    double avg_nnz = (double)NZ / M;
    int hack_size = 2048 * ((int)(avg_nnz / 10) + 1);

    printf("Conversione matrice in formato HLL con hack_size = %d...\n", hack_size);
    HLLMatrix *A_hll = convert_to_HLL(M, N, NZ, entries, hack_size);
    print_HLL(A_hll);

    double *y_omp_hll = allocate_result(M);
    omp_set_num_threads(num_threads);
    printf("Moltiplicazione parallela con HLL e %d thread...\n", num_threads);
    get_performances_and_save((void (*)(void *, double *, double *))omp_hll_mult,
        A_hll, x, y_omp_hll,
        matrix_name, M, N, NZ,
        "-ompHLL", num_threads, y_serial);
    
    free(y_omp_hll);
    free_HLL(A_hll);
    
    free_CSR(A);
    free(y_serial);
    free(entries);
    free(x);
    
    return 0;
}