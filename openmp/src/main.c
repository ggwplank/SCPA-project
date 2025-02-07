#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "utils.h"

void save_results_to_csv(const char *filename, const char *matrix_name,
    int M, int N, int NZ, 
    const char *mode, int threads,
    double time_ms,double median_time_ms, double flops, double mflops, double gflops,
    int passed, double diff, double rel_diff, int iterations) {

    FILE *file = fopen(filename, "a");
    if (file == NULL) {
        perror("Errore nell'apertura del file CSV");
        exit(EXIT_FAILURE);
    }

    // Scrivi l'header se il file è vuoto
    fseek(file, 0, SEEK_END);
    if (ftell(file) == 0) {
        fprintf(file, "Matrix,M,N,nz,CalculationMode,Threads,CalculationTime(ms),MedianTime(ms),Flops,MFlops,GFlops,Passed,Diff,RelDiff,Iterations\n");
    }

    fprintf(file, "%s,%d,%d,%d,%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.6f,%.6f,%d\n",
            matrix_name, M, N, NZ, mode, threads, time_ms, median_time_ms, flops, mflops, gflops, passed, diff, rel_diff, iterations);

    fclose(file);
}

int compare_doubles(const void *a, const void *b) {
    double arg1 = *(const double *)a;
    double arg2 = *(const double *)b;
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Uso: %s <file_matrix_market> <-serial/-ompCSR/-ompHLL> {num threads}\n", argv[0]);
        return 1;
    }

    const char *matrix_filename = argv[1];  // --> dir/file.mtx
    const char *matrix_name = strrchr(matrix_filename, '/'); // --> /file.mtx
    matrix_name = (matrix_name) ? matrix_name + 1 : matrix_filename; // --> file.mtx

    const char *mode = argv[2];

    int num_threads = (argc == 4) ? atoi(argv[3]) : omp_get_max_threads();

    int M, N, NZ;
    MatrixEntry *entries;

    read_matrix_market(matrix_filename, &M, &N, &NZ, &entries);
    printf("Matrice %dx%d, nonzeri: %d\n", M, N, NZ);

    CSRMatrix *A = convert_to_CSR(M, N, NZ, entries);

    ELLPackMatrix *A_ellpack = convert_to_ELL(M, N, NZ, entries);
    
    double *x = NULL;
    generate_random_vector(matrix_name, M, &x);

    // for (int i = 0; i < A_ellpack->rows; i++) {
    //     printf("Riga %d: ", i);
    //     for (int j = 0; j < A_ellpack->nnz / A_ellpack->rows; j++) {  // Al massimo, ogni riga ha max_row_length non-zeri
    //         if (A_ellpack->col_indices[i][j] != -1) {  // Se la colonna è valida (non -1)
    //             printf("(Colonna: %d, Valore: %f) ", A_ellpack->col_indices[i][j], A_ellpack->values[i][j]);
    //         }
    //     }
    //     printf("\n");
    // }


    // printf("Formato CSR:\nValori: ");
    // for (int i = 0; i < A->nnz; i++) printf("%lf ", A->values[i]);

    // printf("\nColonne: ");
    // for (int i = 0; i < A->nnz; i++) printf("%d ", A->col_indices[i]);

    // printf("\nRow Ptr: ");
    // for (int i = 0; i <= A->rows; i++) printf("%d ", A->row_ptr[i]);

    double *y_serial = allocate_result(M);
    serial_csr_mult(A, x, y_serial);

    double start_time, end_time, total_time = 0.0;
    int repetitions = 10;
    double times[repetitions];

    if (strcmp(mode, "-serial") == 0) {
        for (int i = 0; i < repetitions; i++) {
            start_time = omp_get_wtime();
            serial_csr_mult(A, x, y_serial);
            end_time = omp_get_wtime();

            double elapsed_time = end_time - start_time;
            total_time += elapsed_time;
            times[i] = elapsed_time;
        }
        total_time /= repetitions;
        double time_ms = total_time * 1000;

        qsort(times, repetitions, sizeof(double), compare_doubles);
        double median_time_ms = times[repetitions / 2] * 1000;
        if (repetitions % 2 == 0) {
            median_time_ms = (times[repetitions / 2 - 1] + times[repetitions / 2]) * 1000;
        } else {
            median_time_ms = times[repetitions / 2] * 1000;
        }

        double flops = (2.0 * NZ) / total_time;
        double mflops = flops / 1e6;
        double gflops = flops / 1e9;

        printf("Primi 10 valori risultato moltiplicazione seriale: ");
        for (int i = 0; i < 10; i++) 
            printf("%f ", y_serial[i]);
        printf("\n");

        save_results_to_csv("performance.csv", matrix_name, M, N, NZ, mode, 1, time_ms, median_time_ms, flops, mflops, gflops, 1, 0.0, 0.0, repetitions);

        free(y_serial);
    }

    else if (strcmp(mode, "-ompCSR") == 0) {
        double *y_omp_csr = allocate_result(M);

        omp_set_num_threads(num_threads);

        for (int i = 0; i < repetitions; i++) {
            start_time = omp_get_wtime();
            omp_csr_mult(A, x, y_omp_csr);
            end_time = omp_get_wtime();

            double elapsed_time = end_time - start_time;
            total_time += elapsed_time;
            times[i] = elapsed_time;
        }
        total_time /= repetitions;
        double time_ms = total_time * 1000;

        qsort(times, repetitions, sizeof(double), compare_doubles);
        double median_time_ms = times[repetitions / 2] * 1000;
        if (repetitions % 2 == 0) {
            median_time_ms = (times[repetitions / 2 - 1] + times[repetitions / 2]) * 1000;
        } else {
            median_time_ms = times[repetitions / 2] * 1000;
        }

        double flops = (2.0 * NZ) / total_time;
        double mflops = flops / 1e6;
        double gflops = flops / 1e9;


        int passed;
        double diff, rel_diff;
        compare_results(y_serial, y_omp_csr, M, &passed, &diff, &rel_diff);

        save_results_to_csv("performance.csv", matrix_name, M, N, NZ, mode, num_threads, time_ms, median_time_ms, flops, mflops, gflops, passed, diff, rel_diff, repetitions);

    }

    else if (strcmp(mode, "-ompHLL") == 0) {
        double *y_omp_hll = allocate_result(M);

        int hack_size = 32;
        HLLMatrix *A_hll = convert_to_HLL(M, N, NZ, entries, hack_size);

        print_HLL(A_hll);

        omp_set_num_threads(num_threads);
        //omp_ellpack_mult(A_ellpack, x, y_omp_hll);

        for (int i = 0; i < repetitions; i++) {
            start_time = omp_get_wtime();
            omp_hll_mult(A_hll, x, y_omp_hll);
            end_time = omp_get_wtime();

            double elapsed_time = end_time - start_time;
            total_time += elapsed_time;
            times[i] = elapsed_time;
        }
        total_time /= repetitions;
        double time_ms = total_time * 1000;

        qsort(times, repetitions, sizeof(double), compare_doubles);
        double median_time_ms = times[repetitions / 2] * 1000;
        if (repetitions % 2 == 0) {
            median_time_ms = (times[repetitions / 2 - 1] + times[repetitions / 2]) * 1000;
        } else {
            median_time_ms = times[repetitions / 2] * 1000;
        }

        double flops = (2.0 * NZ) / total_time;
        double mflops = flops / 1e6;
        double gflops = flops / 1e9;

        int passed;
        double diff, rel_diff;
        compare_results(y_serial, y_omp_hll, M, &passed, &diff, &rel_diff);

        save_results_to_csv("performance.csv", matrix_name, M, N, NZ, mode, num_threads, time_ms, median_time_ms, flops, mflops, gflops, passed, diff, rel_diff, repetitions);

        free_HLL(A_hll);
    }

    else {
        printf("Le possibili modalità sono: -serial, -ompCSR, -ompHLL\n");
    }
    
    free_CSR(A);
    free_ELL(A_ellpack);
    free(entries);
    free(x);
    
    return 0;
}