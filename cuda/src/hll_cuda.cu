#include <cuda_runtime.h>
#include <stdio.h>

#include "utils.h"

__global__ void matvec_hll_kernel(double *values, int *col_indices, double *x, double *y, 
                                  int *block_offsets, int *block_nnz, int *block_rows, int hack_size, int num_blocks) {
    int global_row = blockIdx.x * blockDim.x + threadIdx.x;  // Indice globale della riga

    if (global_row >= num_blocks * hack_size) return; // Evita di superare i limiti

    // Trova il blocco di appartenenza
    int block_id = global_row / hack_size; 
    if (block_id >= num_blocks) return;  // Controllo di sicurezza

    int block_start = block_offsets[block_id];   // Offset della sottomatrice nel vettore flat
    int rows = block_rows[block_id];      // Numero di righe nel blocco

    int local_row = global_row % hack_size;  // Riga locale nel blocco
    if (local_row >= rows) return; // Se fuori dal blocco, esci

    double sum = 0.0;
    int maxnz = block_nnz[block_id]; // Numero di colonne non nulle del blocco

    for (int j = 0; j < maxnz; j++) {
        int col = col_indices[block_start + j * rows + local_row]; 
        if (col >= 0) {
            sum += values[block_start + j * rows  + local_row] * x[col];
        }
    }
    y[global_row] = sum;  // Scrive nel vettore risultato globale

}

void matvec_hll_cuda(HLLMatrix *H, double *x, double *y, float *elapsed_time) {
    printf("Eseguo il prodotto matrice-vettore con CUDA...\n");
    int num_blocks = H->num_blocks;
    int hack_size = H->hack_size;

    // Calcola il numero totale di righe nella matrice HLL
    int total_rows = 0;
    for (int b = 0; b < num_blocks; b++) {
        if (H->blocks[b] != NULL) {
            total_rows += H->blocks[b]->rows;
        }
    }

    // Allocazione per la struttura lineare di tutti i blocchi
    int total_values = 0;
    for (int b = 0; b < num_blocks; b++) {
        if (H->blocks[b] != NULL)
            total_values += H->blocks[b]->rows * H->blocks[b]->maxnz;
    }

    double *h_values = (double*)malloc(total_values * sizeof(double));
    int *h_col_indices = (int*)malloc(total_values * sizeof(int));
    int *h_block_offsets = (int*)malloc(num_blocks * sizeof(int));
    int *h_block_nnz = (int*)malloc(num_blocks * sizeof(int));
    int *h_block_rows = (int*)malloc(num_blocks * sizeof(int));

    // Conversione della matrice HLL in formati compatibili con CUDA
    int offset = 0;
    for (int b = 0; b < num_blocks; b++) {
        ELLPackMatrix *block = H->blocks[b];
        if (block == NULL) continue;

        h_block_offsets[b] = offset;  
        h_block_nnz[b] = block->maxnz;
        h_block_rows[b] = block->rows;

        for (int i = 0; i < block->rows; i++) {
            for (int j = 0; j < block->maxnz; j++) {
                h_values[offset + j * block->rows + i] = block->values[i][j];
                h_col_indices[offset + j * block->rows + i] = block->col_indices[i][j];
            }
        }
        offset += block->rows * block->maxnz;
    }


    // Allocazione memoria sulla GPU
    double *d_values, *d_x, *d_y;
    int *d_col_indices, *d_block_offsets, *d_block_nnz, *d_block_rows;

    cudaMalloc((void **)&d_values, total_values * sizeof(double));
    cudaMalloc((void **)&d_col_indices, total_values * sizeof(int));
    cudaMalloc((void **)&d_x, H->blocks[0]->cols * sizeof(double));
    cudaMalloc((void **)&d_y, num_blocks * hack_size * sizeof(double));
    cudaMalloc((void **)&d_block_offsets, num_blocks * sizeof(int));
    cudaMalloc((void **)&d_block_nnz, num_blocks * sizeof(int));
    cudaMalloc((void **)&d_block_rows, num_blocks * sizeof(int));

    // Copia dei dati sulla GPU
    cudaMemcpy(d_values, h_values, total_values * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, h_col_indices, total_values * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, H->blocks[0]->cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_offsets, h_block_offsets, num_blocks * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_nnz, h_block_nnz, num_blocks * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_rows, h_block_rows, num_blocks * sizeof(int), cudaMemcpyHostToDevice);

    // Configurazione del kernel
    int block_size = 256;
    int num_threads = num_blocks * hack_size;
    int grid_size = (num_threads + block_size - 1) / block_size;

    // Configurazione per il calcolo del tempo di esecuzione
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // Lancio del kernel
    matvec_hll_kernel<<<grid_size, block_size>>>(d_values, d_col_indices, d_x, d_y, d_block_offsets, d_block_nnz, d_block_rows, hack_size, num_blocks);


    // registrazione del tempo di esecuzione
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Copia del risultato dalla GPU alla CPU
    cudaMemcpy(y, d_y, total_rows * sizeof(double), cudaMemcpyDeviceToHost);

    // allocazioen del tempo
    cudaEventElapsedTime(elapsed_time, start, stop);



    // Pulizia della memoria
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_values);
    cudaFree(d_col_indices);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_block_offsets);
    cudaFree(d_block_nnz);
    cudaFree(d_block_rows);
}
