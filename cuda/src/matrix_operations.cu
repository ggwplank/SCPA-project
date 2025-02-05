#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "reader.h"
#include "csr.h"
#include "csr_cuda.h"
#include "vector_generator.h"

#define EPSILON 1e-6

void read_and_convert_matrix(const char *matrix_filename, CSRMatrix **A, int *M, int *N, int *NZ, MatrixEntry **entries) {
    read_matrix_market(matrix_filename, M, N, NZ, entries);
    printf("Matrice %dx%d, nonzeri: %d\n", *M, *N, *NZ);

    *A = convert_to_CSR(*M, *N, *NZ, *entries);
}

void generate_vector(const char *matrix_name, int size, double **vector) {
    generate_random_vector(matrix_name, size, vector);
}

void compare_results(double *y_serial, double *y_parallel, int size) {
    int correct = 1;
    for (int i = 0; i < size; i++) {
        if (fabs(y_serial[i] - y_parallel[i]) > EPSILON) {
            correct = 0;
            printf("Differenza rilevata all'indice %d: seriale=%lf, parallelo=%lf\n",
                   i, y_serial[i], y_parallel[i]);
        }
    }

    if (correct)
        printf("I risultati seriale e parallelo sono uguali\n");
    else
        printf("I risultati seriale e parallelo sono diversi\n");
    
}

void multiply_and_compare(CSRMatrix *A, double *x, int M) {
    double *y_serial = (double *)malloc(M * sizeof(double));
    if (!y_serial) {
        perror("Errore di allocazione per il vettore risultato seriale");
        free_CSR(A);
        free(x);
        exit(1);
    }

    double *y_parallel = (double *)malloc(M * sizeof(double));
    if (!y_parallel) {
        perror("Errore di allocazione per il vettore risultato parallelo");
        free_CSR(A);
        free(x);
        free(y_serial);
        exit(1);
    }

    serial_csr_matrix_vector_multiply(A, x, y_serial);
    cuda_csr_matrix_vector_multiply(A, x, y_parallel);
    compare_results(y_serial, y_parallel, M);

    free(y_serial);
    free(y_parallel);
}