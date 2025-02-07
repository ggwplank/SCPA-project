#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "utils.h"

// Kernel CUDA per il prodotto matrice-vettore con il formato HLL
__global__ void hll_mult_cuda_kernel(int num_blocks, int hack_size, 
                                     int *d_block_row_ptrs, int *d_block_col_indices, 
                                     double *d_block_values, double *d_x, double *d_y) {
    // Ogni thread calcola il valore di una riga globale della matrice
    int global_row = blockIdx.x * blockDim.x + threadIdx.x;  
    if (global_row >= num_blocks * hack_size) return; // Se fuori dai limiti, termina

    // Determina a quale blocco appartiene la riga attuale
    int block_id = global_row / hack_size;  // Blocchi organizzati in ordine sequenziale
    int local_row = global_row % hack_size; // Riga all'interno del blocco

    // Recupera il numero massimo di elementi non nulli per riga in questo blocco
    int maxnz = d_block_row_ptrs[block_id * (hack_size + 1) + hack_size];

    double sum = 0.0;
    for (int j = 0; j < maxnz; j++) {
        // Calcola l'indice globale degli elementi del blocco
        int index = block_id * hack_size * maxnz + local_row * maxnz + j;
        int col = d_block_col_indices[index]; // Ottiene la colonna corrispondente
        if (col != -1) { // Se la colonna è valida, effettua il calcolo
            sum += d_block_values[index] * d_x[col];
        }
    }
    d_y[global_row] = sum; // Memorizza il risultato nel vettore di output
}

// Funzione host per eseguire il prodotto matrice-vettore in formato HLL
void cuda_hll_mult(HLLMatrix *H, double *x, double *y) {
    int num_blocks = H->num_blocks; // Numero di blocchi nella matrice
    int hack_size = H->hack_size;   // Numero massimo di righe per blocco
    int total_rows = num_blocks * hack_size; // Numero totale di righe nella matrice

    // Dichiarazione dei puntatori per la memoria sulla GPU
    int *d_block_row_ptrs, *d_block_col_indices;
    double *d_block_values, *d_x, *d_y;

    // Allocazione della memoria per il vettore di input e output
    cudaMalloc((void **)&d_x, H->blocks[0]->cols * sizeof(double));
    cudaMalloc((void **)&d_y, total_rows * sizeof(double));

    // Assumiamo che tutti i blocchi abbiano lo stesso maxnz
    int maxnz = H->blocks[0]->maxnz;

    // Allocazione della memoria per la rappresentazione della matrice HLL
    cudaMalloc((void **)&d_block_row_ptrs, num_blocks * (hack_size + 1) * sizeof(int));
    cudaMalloc((void **)&d_block_col_indices, num_blocks * hack_size * maxnz * sizeof(int));
    cudaMalloc((void **)&d_block_values, num_blocks * hack_size * maxnz * sizeof(double));

    // Copia del vettore x nella memoria della GPU
    cudaMemcpy(d_x, x, H->blocks[0]->cols * sizeof(double), cudaMemcpyHostToDevice);

    // Allocazione della memoria temporanea sulla CPU per organizzare i dati
    int *h_block_row_ptrs = (int *)malloc(num_blocks * (hack_size + 1) * sizeof(int));
    int *h_block_col_indices = (int *)malloc(num_blocks * hack_size * maxnz * sizeof(int));
    double *h_block_values = (double *)malloc(num_blocks * hack_size * maxnz * sizeof(double));

    // Preparazione dei dati per la GPU
    for (int b = 0; b < num_blocks; b++) {
        ELLPackMatrix *block = H->blocks[b]; // Ottiene il blocco corrente

        for (int i = 0; i < hack_size; i++) {
            // Memorizza il numero massimo di elementi non nulli per riga
            h_block_row_ptrs[b * (hack_size + 1) + i] = (i < block->rows) ? block->maxnz : 0;

            for (int j = 0; j < maxnz; j++) {
                int idx = b * hack_size * maxnz + i * maxnz + j;
                
                // Se la riga è valida, copia gli indici delle colonne e i valori
                if (i < block->rows) {
                    h_block_col_indices[idx] = block->col_indices[i][j];
                    h_block_values[idx] = block->values[i][j];
                } else { 
                    // Se la riga è fuori dai limiti del blocco, assegna valori di default
                    h_block_col_indices[idx] = -1;
                    h_block_values[idx] = 0.0;
                }
            }
        }
        // Imposta maxnz per il blocco attuale
        h_block_row_ptrs[b * (hack_size + 1) + hack_size] = maxnz;
    }

    // Copia i dati della matrice nella memoria della GPU
    cudaMemcpy(d_block_row_ptrs, h_block_row_ptrs, num_blocks * (hack_size + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_col_indices, h_block_col_indices, num_blocks * hack_size * maxnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_values, h_block_values, num_blocks * hack_size * maxnz * sizeof(double), cudaMemcpyHostToDevice);

    // Configurazione dei thread e dei blocchi
    int blockSize = 256;  // Numero di thread per blocco
    int gridSize = (total_rows + blockSize - 1) / blockSize;  // Numero di blocchi

    // Lancio del kernel CUDA per il calcolo parallelo
    hll_mult_cuda_kernel<<<gridSize, blockSize>>>(num_blocks, hack_size, 
                                                  d_block_row_ptrs, d_block_col_indices, 
                                                  d_block_values, d_x, d_y);

    // Copia il risultato del calcolo dalla GPU alla CPU
    cudaMemcpy(y, d_y, total_rows * sizeof(double), cudaMemcpyDeviceToHost);

    // Pulizia della memoria sulla GPU
    cudaFree(d_block_row_ptrs);
    cudaFree(d_block_col_indices);
    cudaFree(d_block_values);
    cudaFree(d_x);
    cudaFree(d_y);

    // Pulizia della memoria temporanea sulla CPU
    free(h_block_row_ptrs);
    free(h_block_col_indices);
    free(h_block_values);
}
