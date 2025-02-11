#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

#include "utils.h"

void save_results_to_csv(const char *filename, const char *matrix_name,
    int M, int N, int NZ, 
    const char *mode, int threads,
    double time_ms,double median_time_ms, double best_time_ms,
    double gflops, double gflops_median, double best_gflops,
    int passed, int iterations) {

    FILE *file = fopen(filename, "a");
    if (file == NULL) {
        perror("Errore nell'apertura del file CSV");
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    if (ftell(file) == 0)
        fprintf(file,
            "Matrix,M,N,NZ,Mode,Threads,AvgTime(ms),MedianTime(ms),BestTime(ms),AvgGFlops,MedianGFlops,BestGFlops,Passed,Iterations\n");
    

    mode = mode + 1; // skip the '-' character

    char *dot = strrchr(matrix_name, '.'); // remove the extension
    if (dot) *dot = '\0';

    fprintf(file, "%s,%d,%d,%d,%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d\n",
        matrix_name, M, N, NZ, mode, threads, time_ms, median_time_ms, best_time_ms, gflops, gflops_median, best_gflops, passed, iterations);

    fclose(file);
}

int compare_doubles(const void *a, const void *b) {
    double arg1 = *(const double *)a;
    double arg2 = *(const double *)b;
    return (arg1 > arg2) - (arg1 < arg2);
}

void get_performances_and_save(
    void (*matrix_mult)(void *, double *, double *), 
    void *matrix, double *x, double *y_result, 
    const char *matrix_name, int M, int N, int NZ, 
    const char *mode, int num_threads,
    double *y_serial
) {
    double start_time, end_time, total_time = 0.0;
    double times[REPETITIONS];

    for (int i = 0; i < REPETITIONS; i++) {
        start_time = omp_get_wtime();
        matrix_mult(matrix, x, y_result);
        end_time = omp_get_wtime();

        double elapsed_time = end_time - start_time;
        times[i] = elapsed_time;
    }

    for (int i = 1; i < REPETITIONS; i++)
        total_time += times[i];

    total_time /= (REPETITIONS - 1); 
    double time_ms = total_time * 1000;

    qsort(times + 1, REPETITIONS - 1, sizeof(double), compare_doubles);
    double median_time = (REPETITIONS - 1) % 2 == 0 
                         ? (times[(REPETITIONS - 1) / 2] + times[(REPETITIONS - 1) / 2 + 1]) / 2
                         : times[(REPETITIONS - 1) / 2];
    double median_time_ms = median_time * 1000;
    
    double best_time = times[1];
    for (int i = 2; i < REPETITIONS; i++)
        if (times[i] < best_time)
            best_time = times[i];
    double best_time_ms = best_time * 1000;


    double flops = (2.0 * NZ) / total_time;
    double gflops = flops / 1e9;

    double flops_median = (2.0 * NZ) / median_time;
    double gflops_median = flops_median / 1e9;

    double best_flops = (2.0 * NZ) / best_time;
    double best_gflops = best_flops / 1e9;

    int passed = 1;

    if (y_serial)
        compare_results(y_serial, y_result, M, &passed);


    save_results_to_csv(PERFORMANCE_FILE, matrix_name, M, N, NZ, mode, num_threads, 
                        time_ms, median_time_ms, best_time_ms,
                        gflops, gflops_median, best_gflops,
                        passed, REPETITIONS);
}