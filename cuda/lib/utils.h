#ifndef UTILS_H
#define UTILS_H

#include "mmio.h"

// ------ Constants -------

#define PERFORMANCE_FILE "performance.csv"
#define REPETITIONS 5
#define REL_TOL 1e-6
#define ABS_TOL 1e-9
#define HACK_SIZE 2048



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

void serial_csr_mult(CSRMatrix *A, double *x, double *y, float *elapsed_time);

void cuda_csr_mult(CSRMatrix *A, double *x, double *y, float *elapsed_time);

void cuda_csr_mult_warp(CSRMatrix *A, double *x, double *y, float *elapsed_time);

void select_and_run_cuda_csr(CSRMatrix *A, double *x, double *y, float *elapsed_time);

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

void free_ELL(ELLPackMatrix *A);



// ------ HLL -------

typedef struct {
    int num_blocks;        // Numero di blocchi
    int hack_size;         // Numero di righe per blocco
    ELLPackMatrix **blocks; // Array di puntatori ai blocchi ELLPack
} HLLMatrix;

HLLMatrix* convert_to_HLL(int M, int N, int NZ, MatrixEntry *entries, int hack_size);

void matvec_hll_cuda(HLLMatrix *H, double *x, double *y, float *elapsed_time);

void free_HLL(HLLMatrix *H);

void print_HLL(HLLMatrix *H);



// ------ Matrix operations ------

void compare_results(double *y_serial, double *y_parallel, int size, int *passed);

double * allocate_result(int M);

void save_results_to_csv(const char *filename, const char *matrix_name,
    int M, int N, int NZ, 
    const char *mode,
    double time_ms,double median_time_ms, double best_time_ms,
    double gflops, double gflops_median, double best_gflops,
    int passed, int iterations);

void get_performances_and_save_cuda(
    void (*matrix_mult)(void *, double *, double *, float *), 
    void *matrix, double *x, double *y_result, 
    const char *matrix_name, int M, int N, int NZ, 
    const char *mode,
    double *y_serial
);

int compare_doubles(const void *a, const void *b);



// ------ Vector generation ------

void generate_random_vector(const char *matrix_name, int size, double **vector);

#endif