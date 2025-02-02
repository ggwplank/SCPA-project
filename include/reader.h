#ifndef READER_H
#define READER_H

#include "mmio.h"

typedef struct {
    int row;
    int col;
    double value;
} MatrixEntry;

void read_matrix_market(const char *filename, int *M, int *N, int *NZ, MatrixEntry **entries);

#endif