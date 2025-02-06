#ifndef CSR_CUDA_H
#define CSR_CUDA_H

#include "utils.h"
#include <cuda_runtime.h>

__global__ void csr_mult_cuda_kernel(int rows, int *d_row_ptr, int *d_col_indices, double *d_values, double *d_x, double *d_y);
void cuda_csr_mult(CSRMatrix *A, double *x, double *y);

#endif