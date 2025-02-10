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

    printf("\nLettura della matrice...\n");
    read_matrix_market(matrix_filename, &M, &N, &NZ, &entries);
    printf("Matrice %dx%d, nonzeri: %d\n", M, N, NZ);

    printf("Conversione in formato CSR...\n");
    CSRMatrix *A = convert_to_CSR(M, N, NZ, entries);

    //ELLPackMatrix *A_ellpack = convert_to_ELL(M, N, NZ, entries);
    
    double *x = NULL;
    printf("Generazione del vettore randomico...\n");
    generate_random_vector(matrix_name, M, &x);

    double *y_serial = allocate_result(M);
    printf("Moltiplicazione seriale...\n");
    serial_csr_mult(A, x, y_serial);

    if (strcmp(mode, "-serial") == 0) {
        printf("Calcolo delle prestazioni per la moltiplicazione seriale...\n");
        get_performances_and_save((void (*)(void *, double *, double *))serial_csr_mult,
            A, x, y_serial,
            matrix_name, M, N, NZ,
            mode, 1, NULL);
    }

    else if (strcmp(mode, "-ompCSR") == 0) {
        double *y_omp_csr = allocate_result(M);

        omp_set_num_threads(num_threads);
        printf("Moltiplicazione parallela con CSR e %d thread...\n", num_threads);
        get_performances_and_save((void (*)(void *, double *, double *))omp_csr_mult,
            A, x, y_omp_csr,
            matrix_name, M, N, NZ,
            mode, num_threads, y_serial);

        free(y_omp_csr);
    }

    else if (strcmp(mode, "-ompHLL") == 0) {
        double *y_omp_hll = allocate_result(M);

        printf("Conversione matrice in formato HLL con hack_size = %d...\n", HACK_SIZE);
        HLLMatrix *A_hll = convert_to_HLL(M, N, NZ, entries, HACK_SIZE);

        print_HLL(A_hll);

        omp_set_num_threads(num_threads);
 
        printf("Moltiplicazione parallela con HLL e %d thread...\n", num_threads);
        get_performances_and_save((void (*)(void *, double *, double *))omp_hll_mult,
            A_hll, x, y_omp_hll,
            matrix_name, M, N, NZ,
            mode, num_threads, y_serial);
    
        free_HLL(A_hll);
        free(y_omp_hll);
    }

    else {
        printf("Le possibili modalità sono: -serial, -ompCSR, -ompHLL\n");
    }
    
    free_CSR(A);
    //free_ELL(A_ellpack);
    free(y_serial);
    free(entries);
    free(x);
    
    return 0;
}