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

    const char *matrix_filename = argv[1];
    const char *matrix_name = strrchr(matrix_filename, '/');
    matrix_name = (matrix_name) ? matrix_name + 1 : matrix_filename;

    const char *mode = argv[2];

    int M, N, NZ;
    MatrixEntry *entries;

    printf("\nLettura della matrice...\n");
    read_matrix_market(matrix_filename, &M, &N, &NZ, &entries);
    printf("Matrice %dx%d, nonzeri: %d\n", M, N, NZ);

    printf("Conversione in formato CSR...\n");
    CSRMatrix *A = convert_to_CSR(M, N, NZ, entries);
    
    double *x = NULL;
    printf("Generazione del vettore randomico...\n");
    generate_random_vector(matrix_name, M, &x);
    
    double *y_serial = allocate_result(M);
    printf("Moltiplicazione seriale...\n");
    serial_csr_mult(A, x, y_serial,NULL);

    if (strcmp(mode, "-serial") == 0) {
        printf("Calcolo delle prestazioni per la moltiplicazione seriale...\n");

        get_performances_and_save_cuda((void (*)(void *, double *, double *, float *))serial_csr_mult,
        A, x, y_serial,
        matrix_name, M, N, NZ,
        mode, NULL);

        printf("Calcolo terminato.\n");

        free(entries);
    }

    else if (strcmp(mode, "-cudaCSR") == 0) {
        double *y_cuda_csr = allocate_result(M);

        printf("Moltiplicazione parallela con CUDA e formato CSR...\n");

        get_performances_and_save_cuda((void (*)(void *, double *, double *, float *))select_and_run_cuda_csr,
        A, x, y_cuda_csr,
        matrix_name, M, N, NZ,
        mode, y_serial);

        free(entries);
    }

    else if (strcmp(mode, "-cudaHLL") == 0) {    
        printf("Conversione matrice in formato HLL con hack_size = %d...\n", HACK_SIZE);
        HLLMatrix *A_hll = convert_to_HLL(M, N, NZ, entries, HACK_SIZE);
        print_HLL(A_hll);

        double *y_cuda_hll = allocate_result(M);
    
        get_performances_and_save_cuda((void (*)(void *, double *, double *, float *))matvec_hll_cuda,
        A_hll, x, y_cuda_hll,
        matrix_name, M, N, NZ,
        mode, y_serial);

        free_HLL(A_hll);
        free(y_cuda_hll);
    }

    else {
        printf("Le possibili modalità sono: -serial, -cudaCSR, -cudaHLL\n");
    }

    free_CSR(A);
    free(y_serial);
    free(x);
    
    return 0;
}