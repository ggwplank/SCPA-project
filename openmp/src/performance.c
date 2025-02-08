#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

#include "utils.h"

void save_results_to_csv(const char *filename, const char *matrix_name,
    int M, int N, int NZ, 
    const char *mode, int threads,
    double time_ms,double median_time_ms,
    double flops, double mflops, double gflops,
    double flops_median, double mflops_median, double gflops_median,
    int passed, double diff, double rel_diff, int iterations) {

    FILE *file = fopen(filename, "a");
    if (file == NULL) {
        perror("Errore nell'apertura del file CSV");
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    if (ftell(file) == 0)
        fprintf(file,
            "Matrix,M,N,nz,CalculationMode,Threads,CalculationTime(ms),MedianTime(ms),Flops,MFlops,GFlops,Flops_Median,MFlops_Median,GFlops_Median,Passed,Diff,RelDiff,Iterations\n");
    

    mode = mode + 1; // skip the '-' character

    char *dot = strrchr(matrix_name, '.'); // remove the extension
    if (dot) *dot = '\0';

    threads = (threads > omp_get_max_threads()) ? omp_get_max_threads() : threads;
    printf("Threads: %d\n", threads);

    fprintf(file, "%s,%d,%d,%d,%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.6f,%.6f,%d\n",
            matrix_name, M, N, NZ, mode, threads, time_ms, median_time_ms, flops, mflops, gflops, flops_median, mflops_median, gflops_median, passed, diff, rel_diff, iterations);

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
        total_time += elapsed_time;
        times[i] = elapsed_time;
    }

    total_time /= REPETITIONS;
    double time_ms = total_time * 1000;

    qsort(times, REPETITIONS, sizeof(double), compare_doubles);
    double median_time = REPETITIONS % 2 == 0 
                         ? (times[REPETITIONS / 2 - 1] + times[REPETITIONS / 2]) / 2
                         : times[REPETITIONS / 2];
    double median_time_ms = median_time * 1000;

    double flops = (2.0 * NZ) / total_time;
    double mflops = flops / 1e6;
    double gflops = flops / 1e9;

    double flops_median = (2.0 * NZ) / median_time;
    double mflops_median = flops_median / 1e6;
    double gflops_median = flops_median / 1e9;


    int passed = 1;
    double diff = 0.0, rel_diff = 0.0;

    if (y_serial)
        compare_results(y_serial, y_result, M, &passed, &diff, &rel_diff);


    save_results_to_csv(PERFORMANCE_FILE, matrix_name, M, N, NZ, mode, num_threads, 
                        time_ms, median_time_ms,
                        flops, mflops, gflops,
                        flops_median, mflops_median, gflops_median,
                        passed, diff, rel_diff, REPETITIONS);
}