#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "utils.h"

ELLPackMatrix* convert_to_ELL(int M, int N, int NZ, MatrixEntry *entries) {
    // Crea la struttura ELLMatrix
    ELLPackMatrix *ellpack = (ELLPackMatrix*)malloc(sizeof(ELLPackMatrix));
    ellpack->rows = M;
    ellpack->cols = N;
    ellpack->nnz = NZ;
    
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
    ellpack->maxnz = max_row_length;

    // Alloca memoria per la matrice ELL
    ellpack->values = (double**)malloc(M * sizeof(double*));
    ellpack->col_indices = (int**)malloc(M * sizeof(int*));
    for (int i = 0; i < M; i++) {
        ellpack->values[i] = (double *)malloc(max_row_length * sizeof(double));
        ellpack->col_indices[i] = (int *)malloc(max_row_length * sizeof(int));
        
        // Inizializza con zero le colonne non usate
        for (int j = 0; j < max_row_length; j++) {
            ellpack->values[i][j] = 0.0;
            ellpack->col_indices[i][j] = -1;
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
        ellpack->values[row][idx] = value;
        ellpack->col_indices[row][idx] = col;
        row_counts_insert[row]++;
    }

    free(row_counts);
    free(row_counts_insert);

    return ellpack;
}

void free_ELL(ELLPackMatrix *A) {
    for (int i = 0; i < A->rows; i++) {
        free(A->values[i]);
        free(A->col_indices[i]);
    }
    free(A->values);
    free(A->col_indices);
    free(A);
}

void print_ELL(ELLPackMatrix *A) {
    printf("ELLPack Matrix: %d x %d (maxnz = %d)\n", A->rows, A->cols, A->maxnz);

    for (int i = 0; i < A->rows; i++) {
        printf("Riga %d: ", i);
        for (int j = 0; j < A->maxnz; j++) {
            printf("(%d, %.2f) ", A->col_indices[i][j], A->values[i][j]);
        }
        printf("\n");
    }
}

void transpose_ELLPack(ELLPackMatrix *A) {
    // Passo 1: Calcolare il nuovo maxnz
    int *column_nnz = (int*)calloc(A->cols, sizeof(int)); // Conta gli elementi per ogni colonna

    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->maxnz; j++) {
            int col = A->col_indices[i][j];
            if (col != -1) {
                column_nnz[col]++;
            }
        }
    }

    // Trovare il massimo numero di elementi in una colonna (che diventerà maxnz della trasposta)
    int new_maxnz = 0;
    for (int i = 0; i < A->cols; i++) {
        if (column_nnz[i] > new_maxnz) {
            new_maxnz = column_nnz[i];
        }
    }

    free(column_nnz); // Non serve più

    // Passo 2: Creazione della nuova matrice trasposta
    ELLPackMatrix *transposed = (ELLPackMatrix*)malloc(sizeof(ELLPackMatrix));
    transposed->rows = A->cols;
    transposed->cols = A->rows;
    transposed->nnz = A->nnz;
    transposed->maxnz = new_maxnz; // Usare il nuovo maxnz calcolato

    // Allocazione delle nuove matrici per i valori e gli indici
    transposed->values = (double**)malloc(transposed->rows * sizeof(double*));
    transposed->col_indices = (int**)malloc(transposed->rows * sizeof(int*));

    for (int i = 0; i < transposed->rows; i++) {
        transposed->values[i] = (double*)calloc(transposed->maxnz, sizeof(double));
        transposed->col_indices[i] = (int*)calloc(transposed->maxnz, sizeof(int));

        // Inizializziamo gli indici di colonna a -1 (come in convert_to_ELL)
        for (int j = 0; j < transposed->maxnz; j++) {
            transposed->col_indices[i][j] = -1;
        }
    }

    // Passo 3: Trasposizione dei dati
    int *row_counts = (int*)calloc(transposed->rows, sizeof(int));  // Contatore di elementi per ogni riga

    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->maxnz; j++) {
            if (A->col_indices[i][j] != -1) {
                int col = A->col_indices[i][j];  // Nuova riga della matrice trasposta
                int row = i;                     // Nuova colonna della matrice trasposta

                int idx = row_counts[col]++;  // Posizione per l'elemento trasposto

                // Controllo di sicurezza per evitare accessi fuori dai limiti
                if (idx >= transposed->maxnz) {
                    printf("Errore: idx (%d) supera maxnz (%d)\n", idx, transposed->maxnz);
                    exit(EXIT_FAILURE);
                }

                transposed->values[col][idx] = A->values[i][j];
                transposed->col_indices[col][idx] = row;
            }
        }
    }

    free(row_counts);

    // Copia i dati della matrice trasposta nella struttura originale
    *A = *transposed;

    // Libera la memoria temporanea (senza liberare i valori e gli indici, ora dentro A)
    free(transposed);
}
