#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "utils.h"

// Kernel CUDA per il prodotto matrice-vettore CSR
__global__ void csr_mult_cuda_kernel(int rows, int *d_row_ptr, int *d_col_indices, double *d_values, double *d_x, double *d_y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // Ogni thread processa una riga

    if (row < rows) {
        double sum = 0.0;

        for (int j = d_row_ptr[row]; j < d_row_ptr[row + 1]; j++)
            sum += d_values[j] * d_x[d_col_indices[j]];
        
        d_y[row] = sum;
    }
}

void cuda_csr_mult(CSRMatrix *A, double *x, double *y, float *elapsed_time) {
    int *d_row_ptr, *d_col_indices;
    double *d_values, *d_x, *d_y;

    // Allocazione memoria sulla GPU
    cudaMalloc((void **)&d_row_ptr, (A->rows + 1) * sizeof(int));
    cudaMalloc((void **)&d_col_indices, A->nnz * sizeof(int));
    cudaMalloc((void **)&d_values, A->nnz * sizeof(double));
    cudaMalloc((void **)&d_x, A->cols * sizeof(double));
    cudaMalloc((void **)&d_y, A->rows * sizeof(double));

    // Copia dati dalla CPU alla GPU
    cudaMemcpy(d_row_ptr, A->row_ptr, (A->rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, A->col_indices, A->nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, A->values, A->nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, A->cols * sizeof(double), cudaMemcpyHostToDevice);

    // Configurazione e lancio del kernel CUDA
    int blockSize = 256;
    int gridSize = (A->rows + blockSize - 1) / blockSize;

    // Configurazione per il calcolo del tempo di esecuzione
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);

    csr_mult_cuda_kernel<<<gridSize, blockSize>>>(A->rows, d_row_ptr, d_col_indices, d_values, d_x, d_y);

    // registrazione del tempo di esecuzione
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Copia il risultato dalla GPU alla CPU
    cudaMemcpy(y, d_y, A->rows * sizeof(double), cudaMemcpyDeviceToHost);

    // allocazione del tempo
    cudaEventElapsedTime(elapsed_time, start, stop);

    // Deallocazione della memoria sulla GPU
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_row_ptr);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
}
