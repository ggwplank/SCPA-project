#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "utils.h"

double * allocate_result(int M) {
    double *y = (double *)malloc(M * sizeof(double));
    if (!y) {
        perror("Errore di allocazione per il vettore risultato");
        free(y);
    }

    return y;
}

void compare_results(double *y_serial, double *y_parallel, int size) {
    int correct = 1;
    double diff = 0.0;
    double max_rel_diff = 0.0;

    for (int i = 0; i < size; i++) {
        double abs_diff = fabs(y_serial[i] - y_parallel[i]);
        double max_val = fmax(fabs(y_serial[i]), fabs(y_parallel[i]));
        double rel_diff_val = (max_val == 0) ? 0.0 : abs_diff / max_val;

        diff = fmax(diff, abs_diff);
        max_rel_diff = fmax(max_rel_diff, rel_diff_val);

        if (abs_diff > ABS_TOL && rel_diff_val > REL_TOL) {
            correct = 0;
            printf("Differenza rilevata all'indice %d: seriale=%lf, parallelo=%lf (rel_diff=%lf)\n",
                   i, y_serial[i], y_parallel[i], rel_diff_val);
        }
    }

    if (correct)
        printf("I risultati seriale e parallelo sono uguali\n");
    else
        printf("I risultati seriale e parallelo sono diversi (max rel diff: %lf)\n", max_rel_diff);

    free(y_serial);
    free(y_parallel);   
}