#ifndef UTILS_H
#define UTILS_H

#include "mmio.h"

// ------ Constants -------

#define PERFORMANCE_FILE "performance.csv"
#define REPETITIONS 3
#define REL_TOL 1e-6
#define ABS_TOL 1e-9
#define HACK_SIZE 128



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

void omp_ellpack_mult(ELLPackMatrix *A, double *x, double *y);

void free_ELL(ELLPackMatrix *A);

void print_ELL(ELLPackMatrix *A);



// ------ HLL -------

typedef struct {
    int num_blocks;        // Numero di blocchi
    int hack_size;         // Numero di righe per blocco
    ELLPackMatrix **blocks; // Array di puntatori ai blocchi ELLPack
} HLLMatrix;

HLLMatrix* convert_to_HLL(int M, int N, int NZ, MatrixEntry *entries, int hack_size);

void omp_hll_mult(HLLMatrix *H, double *x, double *y);

void free_HLL(HLLMatrix *H);

void print_HLL(HLLMatrix *H);



// ------ Results utils ------

void compare_results(double *y_serial, double *y_parallel, int size, int *passed, double *diff, double *rel_diff);

double * allocate_result(int M);

void save_results_to_csv(const char *filename, const char *matrix_name,
    int M, int N, int NZ, 
    const char *mode, int threads,
    double time_ms,double median_time_ms,
    double gflops, double gflops_median,
    int passed, double diff, double rel_diff, int iterations);

void get_performances_and_save(
    void (*matrix_mult)(void *, double *, double *), 
    void *matrix, double *x, double *y_result, 
    const char *matrix_name, int M, int N, int NZ, 
    const char *mode, int num_threads,
    double *y_serial
);

int compare_doubles(const void *a, const void *b);


// ------ Vector generation ------

void generate_random_vector(const char *matrix_name, int size, double **vector);

#endif