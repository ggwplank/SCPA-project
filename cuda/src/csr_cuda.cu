#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "utils.h"

#define WARP_SIZE 32
#define BLOCK_SIZE 128  // Multiplo di 32

__global__ void csr_mult_cuda_kernel(int num_rows, int *d_row_ptr, int *d_col_indices, 
                                     double *d_values, double *d_x, double *d_y) {
    __shared__ double sdata[BLOCK_SIZE];  
    __shared__ int ptrs[BLOCK_SIZE / WARP_SIZE][2];

    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;  // ID globale del thread
    const int thread_in_warp = threadIdx.x & (WARP_SIZE - 1);     // ID del thread dentro il warp
    const int warp_id = thread_id / WARP_SIZE;                    // ID globale del warp
    const int warp_in_block = threadIdx.x / WARP_SIZE;            // Warp ID locale nel blocco
    const int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // Numero totale di warps

    // Ogni warp si occupa di una riga della matrice
    for (int row = warp_id; row < num_rows; row += num_warps) {
        // I primi due thread di ogni warp caricano gli indici di inizio/fine riga
        if (thread_in_warp < 2) {
            ptrs[warp_in_block][thread_in_warp] = d_row_ptr[row + thread_in_warp];
        }
        __syncthreads(); // Sincronizzazione per garantire che ptrs sia stato aggiornato

        int row_start = ptrs[warp_in_block][0];  
        int row_end = ptrs[warp_in_block][1];    

        // Calcolo del prodotto locale
        double sum = 0.0;
        for (int j = row_start + thread_in_warp; j < row_end; j += WARP_SIZE) {
            sum += d_values[j] * d_x[d_col_indices[j]];
        }

        // Riduzione all'interno del warp
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        // Il primo thread del warp scrive il risultato finale nella memoria globale
        if (thread_in_warp == 0) {
            d_y[row] = sum;
        }
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
    int threads_per_block = 128;  // Multiplo di 32 per i warp
    int num_warps = (A->rows + WARP_SIZE - 1) / WARP_SIZE;
    int num_blocks = (num_warps + (threads_per_block / WARP_SIZE) - 1) / (threads_per_block / WARP_SIZE);

    // Configurazione per il calcolo del tempo di esecuzione
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);

    csr_mult_cuda_kernel<<<num_blocks, threads_per_block>>>(A->rows, d_row_ptr, d_col_indices, d_values, d_x, d_y);


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

