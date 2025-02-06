#ifndef UTILS_H
#define UTILS_H

#include "mmio.h"

// ------- Matrix generation -------

typedef struct {
    int row;
    int col;
    double value;
} MatrixEntry;

void read_matrix_market(const char *filename, int *M, int *N, int *NZ, MatrixEntry **entries);



// ------ CSR -------

typedef struct {
    int rows;        
    int cols;        
    int nnz;         
    double *values;  
    int *col_indices; 
    int *row_ptr;    
} CSRMatrix;

CSRMatrix* convert_to_CSR(int M, int N, int NZ, MatrixEntry *entries);

void serial_csr_mult(CSRMatrix *A, double *x, double *y);
void omp_csr_mult(CSRMatrix *A, double *x, double *y);

void free_CSR(CSRMatrix *A);



// ------ ELL -------

typedef struct {
    int M;          // Numero di righe
    int N;          // Numero di colonne
    int NZ;         // Numero di elementi non nulli
    double **values;   // Valori degli elementi non nulli
    int **columns;     // Indici delle colonne degli elementi non nulli
} ELLMatrix;

void read_and_convert_matrix_to_ellpack(const char *matrix_filename, ELLMatrix **A, int *M, int *N, int *NZ, MatrixEntry **entries);
ELLMatrix* convert_to_ELL(int M, int N, int NZ, MatrixEntry *entries, int max_nonzeros_in_row);



// ------ Matrix operations ------

void compare_results(double *y_serial, double *y_parallel, int size);
double * allocate_result(int M);



// ------ Vector generation ------

void generate_random_vector(const char *matrix_name, int size, double **vector);

#endif