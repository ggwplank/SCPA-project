#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "reader.h"
#include "csr.h"
#include "vector_generator.h"

#define EPSILON 1e-6

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

    read_matrix_market(matrix_filename, &M, &N, &NZ, &entries);
    printf("Matrice %dx%d, nonzeri: %d\n", M, N, NZ);
    
    CSRMatrix *A = convert_to_CSR(M, N, NZ, entries);
    
    double *x = NULL;
    generate_random_vector(matrix_name, N, &x);
    
    // printf("Formato CSR:\nValori: ");
    // for (int i = 0; i < A->nnz; i++) printf("%lf ", A->values[i]);

    // printf("\nColonne: ");
    // for (int i = 0; i < A->nnz; i++) printf("%d ", A->col_indices[i]);

    // printf("\nRow Ptr: ");
    // for (int i = 0; i <= A->rows; i++) printf("%d ", A->row_ptr[i]);

    // Allocare il vettore risultato y
    double *y_serial = (double *)malloc(M * sizeof(double));
    if (!y_serial) {
        perror("Errore di allocazione per il vettore risultato seriale");
        free_CSR(A);
        free(entries);
        free(x);
        exit(1);
    }

    double *y_parallel = (double *)malloc(M * sizeof(double));
    if (!y_parallel) {
        perror("Errore di allocazione per il vettore risultato parallelo");
        free_CSR(A);
        free(entries);
        free(x);
        free(y_serial);
        exit(1);
    }

    // Calcolare il prodotto matrice-vettore
    serial_csr_matrix_vector_multiply(A, x, y_serial);
    omp_csr_matrix_vector_multiply(A, x, y_parallel);

    int correct = 1;
    for (int i = 0; i < M; i++) {
        if (fabs(y_serial[i] - y_parallel[i]) > EPSILON) {
            correct = 0;
            printf("Differenza rilevata all'indice %d: seriale=%lf, parallelo=%lf\n",
                   i, y_serial[i], y_parallel[i]);
        }
    }

    if (correct) {
        printf("I risultati seriale e parallelo sono uguali\n");
    } else {
        printf("I risultati seriale e parallelo sono diversi\n");
    }
    

    // Stampare il risultato
    // printf("\nRisultato del prodotto matrice-vettore:\n");
    // for (int i = 0; i < M; i++)
    //     printf("%lf ", y_serial[i]);
    
    // printf("\n");

    // Liberare la memoria
    free_CSR(A);
    free(entries);
    free(x);
    free(y_serial);
    free(y_parallel);

    return 0;
}