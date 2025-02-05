#include "csr.h"

__global__ void csr_matrix_vector_multiply_cuda(int rows, int *d_row_ptr, int *d_col_indices, double *d_values, double *d_x, double *d_y) ;
void cuda_csr_matrix_vector_multiply(CSRMatrix *A, double *x, double *y) ;