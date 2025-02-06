#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include "csr.h"

void read_and_convert_matrix(const char *matrix_filename, CSRMatrix **A, int *M, int *N, int *NZ, MatrixEntry **entries);
void generate_vector(const char *matrix_name, int size, double **vector);
void multiply_and_compare(CSRMatrix *A, double *x, int M);
void compare_results(double *y_serial, double *y_parallel, int size);

void read_and_convert_matrix_to_ellpack(const char *matrix_filename, ELLMatrix **A, int *M, int *N, int *NZ, MatrixEntry **entries);
ELLMatrix* convert_to_ELL(int M, int N, int NZ, MatrixEntry *entries, int max_nonzeros_in_row);
#endif