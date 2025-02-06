#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "reader.h"
#include "csr.h"
#include "vector_generator.h"

#define EPSILON 1e-6

ELLMatrix* convert_to_ELL(int M, int N, int NZ, MatrixEntry *entries) {
    // Crea la struttura ELLMatrix
    ELLMatrix *ell = (ELLMatrix*)malloc(sizeof(ELLMatrix));
    ell->M = M;
    ell->N = N;
    ell->NZ = NZ;
    
    // Calcola il numero di non-zeri per ogni riga
    int *row_counts = (int*)calloc(M, sizeof(int));
    for (int i = 0; i < NZ; i++) {
        row_counts[entries[i].row]++;
    }

    // Trova la lunghezza massima per le righe
    int max_row_length = 0;
    for (int i = 0; i < M; i++) {
        if (row_counts[i] > max_row_length) {
            max_row_length = row_counts[i];
        }
    }

    // Alloca memoria per la matrice ELL
    ell->values = (double**)malloc(M * sizeof(double*));
    ell->columns = (int**)malloc(M * sizeof(int*));
    for (int i = 0; i < M; i++) {
        ell->values[i] = (double*)malloc(max_row_length * sizeof(double));
        ell->columns[i] = (int*)malloc(max_row_length * sizeof(int));
        
        // Inizializza con zero e -1 per le colonne non utilizzate
        for (int j = 0; j < max_row_length; j++) {
            ell->values[i][j] = 0.0;
            ell->columns[i][j] = -1;
        }
    }

    // Inserisce gli elementi nella matrice ELL
    int *row_counts_insert = (int*)calloc(M, sizeof(int));
    for (int i = 0; i < NZ; i++) {
        int row = entries[i].row;
        int col = entries[i].col;
        double value = entries[i].value;;

        // Trova la posizione giusta per ogni elemento nella riga
        int idx = row_counts_insert[row];
        ell->values[row][idx] = value;
        ell->columns[row][idx] = col;
        row_counts_insert[row]++;
    }

    free(row_counts);
    free(row_counts_insert);

    return ell;
}

void read_and_convert_matrix(const char *matrix_filename, CSRMatrix **A, int *M, int *N, int *NZ, MatrixEntry **entries) {
    read_matrix_market(matrix_filename, M, N, NZ, entries);
    printf("Matrice %dx%d, nonzeri: %d\n", *M, *N, *NZ);

    *A = convert_to_CSR(*M, *N, *NZ, *entries);
}

void read_and_convert_matrix_to_ellpack(const char *matrix_filename, ELLMatrix **A, int *M, int *N, int *NZ, MatrixEntry **entries) {
    // Legge la matrice da un file Matrix Market
    read_matrix_market(matrix_filename, M, N, NZ, entries);
    printf("Matrice %dx%d, nonzeri: %d\n", *M, *N, *NZ);

    // Converte la matrice in formato ELLPACK
    *A = convert_to_ELL(*M, *N, *NZ, *entries);
}

void generate_vector(const char *matrix_name, int size, double **vector) {
    generate_random_vector(matrix_name, size, vector);
}

void compare_results(const double *y_serial, const double *y_parallel, int size) {
    int correct = 1;
    double max_rel_diff = 0.0;

    for (int i = 0; i < size; i++) {
        double abs_diff = fabs(y_serial[i] - y_parallel[i]);
        double max_val = fmax(fabs(y_serial[i]), fabs(y_parallel[i]));
        double rel_diff = (max_val == 0) ? 0.0 : abs_diff / max_val;

        max_rel_diff = fmax(max_rel_diff, rel_diff);

        if (abs_diff > EPSILON) {
            correct = 0;
            printf("Differenza rilevata all'indice %d: seriale=%lf, parallelo=%lf (rel_diff=%lf)\n",
                   i, y_serial[i], y_parallel[i], rel_diff);
        }
    }

    if (correct)
        printf("I risultati seriale e parallelo sono uguali\n");
    else
        printf("I risultati seriale e parallelo sono diversi (max rel diff: %lf)\n", max_rel_diff);
    
}

void multiply_and_compare(CSRMatrix *A, double *x, int M) {
    double *y_serial = (double *)malloc(M * sizeof(double));
    if (!y_serial) {
        perror("Errore di allocazione per il vettore risultato seriale");
        free_CSR(A);
        free(x);
        free(y_serial);
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
    omp_csr_matrix_vector_multiply(A, x, y_parallel);

    compare_results(y_serial, y_parallel, M);

    free(y_serial);
    free(y_parallel);
}