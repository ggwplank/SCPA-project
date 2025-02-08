#include <cuda_runtime.h>
#include <stdio.h>

#include "utils.h"

__global__ void matvec_kernel(double *values, int *col_indices, double *x, double *y, int rows, int maxnz) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        double sum = 0.0;
        for (int j = 0; j < maxnz; j++) {
            int col = col_indices[j * rows + row];  // Lettura coalescente
            if (col >= 0) {
                sum += values[j * rows + row] * x[col];
            }
        }
        y[row] = sum;
    }
}
    
void matvec_ellpack_cuda(ELLPackMatrix *A, double *x, double *y) {
    // Debug: Stampa della matrice ELLPack
    int M = A->rows;
    double *d_values, *d_x, *d_y;
    int *d_col_indices, *d_maxnz;
    int maxnz = A->maxnz;
    

    // Conversione da doppio puntatore a array contiguo per GPU
    double* h_values_flat = new double[M * maxnz];
    int* h_col_indices_flat = new int[M * maxnz];

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < maxnz; j++) {
            h_values_flat[i + j * M] = A->values[i][j];
            h_col_indices_flat[i + j * M] = A->col_indices[i][j];
        }
    }

    // Calcola la dimensione della memoria da allocare per la matrice e il vettore
    int size_values = A->rows * A->maxnz * sizeof(double);  // memoria per i valori della matrice
    int size_col_indices = A->rows * A->maxnz * sizeof(int); // memoria per gli indici delle colonne

    // Allocazione della memoria sulla GPU
    cudaMalloc((void **)&d_maxnz, size_values);          // numero massimo di elementi non nulli per riga
    cudaMalloc((void **)&d_values, size_values);          // Matrice valori
    cudaMalloc((void **)&d_col_indices, size_col_indices); // Indici colonna
    cudaMalloc((void **)&d_x, A->cols * sizeof(double));   // Vettore di input x
    cudaMalloc((void **)&d_y, A->rows * sizeof(double));   // Vettore di output y

    // Copia dei dati dalla memoria host alla memoria device
    cudaMemcpy(d_maxnz, &maxnz, sizeof(int), cudaMemcpyHostToDevice); // Copia del numero massimo di elementi non nulli per riga
    cudaMemcpy(d_values, h_values_flat, size_values, cudaMemcpyHostToDevice); // Copia di tutta la matrice dei valori
    cudaMemcpy(d_col_indices, h_col_indices_flat, size_col_indices, cudaMemcpyHostToDevice); // Copia degli indici delle colonne
    cudaMemcpy(d_x, x, A->cols * sizeof(double), cudaMemcpyHostToDevice); // Copia del vettore x


    // Lancio del kernel CUDA
    int block_size = 256; // Numero di thread per blocco
    int num_blocks = (A->rows + block_size - 1) / block_size; // Numero di blocchi necessari
    matvec_kernel<<<num_blocks, block_size>>>(d_values, d_col_indices, d_x, d_y, A->rows, A->maxnz);
    // Gestione degli errori del kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // Copia dei risultati dalla memoria device alla memoria host
    cudaMemcpy(y, d_y, A->rows * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // Libera la memoria allocata sulla GPU
    cudaFree(d_values);
    cudaFree(d_col_indices);
    cudaFree(d_x);
    cudaFree(d_y);
}
