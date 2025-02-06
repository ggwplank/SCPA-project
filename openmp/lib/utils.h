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
    int rows;         
    int cols;          
    int nnz; 
    int maxnz;
    int **col_indices;
    double **values;
} ELLPackMatrix;

ELLPackMatrix* convert_to_ELL(int M, int N, int NZ, MatrixEntry *entries);

void omp_hll_mult(ELLPackMatrix *A, double *x, double *y);

void free_ELL(ELLPackMatrix *A);



// ------ Matrix operations ------

void compare_results(double *y_serial, double *y_parallel, int size);
double * allocate_result(int M);



// ------ Vector generation ------

void generate_random_vector(const char *matrix_name, int size, double **vector);

#endif