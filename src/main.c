#include <stdio.h>
#include <stdlib.h>
#include "reader.h"
#include "csr.h"


int main(int argc, char *argv[]) {
    // if (argc < 2) {
    //     printf("Uso: %s <file_matrix_market>\n", argv[0]);
    //     return 1;
    // }

    // int M, N, NZ;
    // MatrixEntry *entries;

    // read_matrix_market(argv[1], &M, &N, &NZ, &entries);
    // printf("Matrice %dx%d, nonzeri: %d\n", M, N, NZ);
    
    // CSRMatrix *A = convert_to_CSR(M, N, NZ, entries);
    // printf("Formato CSR:\nValori: ");
    // for (int i = 0; i < A->nnz; i++) printf("%lf ", A->values[i]);
    // printf("\nColonne: ");
    // for (int i = 0; i < A->nnz; i++) printf("%d ", A->col_indices[i]);
    // printf("\nRow Ptr: ");
    // for (int i = 0; i <= A->rows; i++) printf("%d ", A->row_ptr[i]);
    // printf("\n");

    // free_CSR(A);
    // free(entries);
    // return 0;
    
    srand(time(NULL)); // Inizializza il generatore di numeri casuali
    process_directory("/home/lorenzo/DEVELOPEMENT/SCPA-project/Matrices", "random_vectors.txt");
    
    return 0;
}
