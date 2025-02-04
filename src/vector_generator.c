#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Calcola un hash (che fungerà poi da seed) in base al nome della matrice considerata.
unsigned int hash_string(const char *str) {
    unsigned int hash = 5381;
    int c;

    while ((c = *str++))
        hash = ((hash << 5) + hash) + c; // hash * 33 + ASCII carattere corrente

    return hash;
}

// Genera un vettore randomico a partire da un seed fisso per garantire la riproducibilità.
void generate_random_vector(const char *matrix_name, int size, double **vector) {
    *vector = (double *)malloc(size * sizeof(double));
    if (!(*vector)) {
        perror("Errore di allocazione del vettore");
        exit(EXIT_FAILURE);
    }

    unsigned int seed = hash_string(matrix_name);
    srand(seed);

    double lower_bound = 0.1;
    double upper_bound = 2.0;

    for (int i = 0; i < size; i++)
        (*vector)[i] = lower_bound + ((double)rand() / RAND_MAX) * (upper_bound - lower_bound);
}
