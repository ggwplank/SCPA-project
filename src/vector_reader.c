#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LENGTH 1024  // Per gestire una riga lunga di numeri

int read_vector_from_file(const char *filename, const char *matrix_name, double **vector, int *size) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Errore nell'apertura del file vettore");
        return 0;
    }

    char line[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), file)) {
        // Rimuove eventuale newline
        line[strcspn(line, "\n")] = 0;

        if (strcmp(line, matrix_name) == 0) {  // Se troviamo il nome della matrice
            if (!fgets(line, sizeof(line), file)) { // Leggiamo la riga successiva (il vettore)
                fclose(file);
                return 0;
            }

            // Contiamo il numero di valori nel vettore
            int count = 0;
            char *token = strtok(line, " ");
            while (token) {
                count++;
                token = strtok(NULL, " ");
            }

            // Allocazione del vettore
            *vector = (double *)malloc(count * sizeof(double));
            if (!(*vector)) {
                fclose(file);
                return 0;
            }

            // Rianalizziamo la riga e convertiamo i valori in double
            rewind(file);
            while (fgets(line, sizeof(line), file)) {
                line[strcspn(line, "\n")] = 0;
                if (strcmp(line, matrix_name) == 0) {
                    fgets(line, sizeof(line), file);
                    break;
                }
            }

            token = strtok(line, " ");
            for (int i = 0; i < count; i++) {
                (*vector)[i] = atof(token);
                token = strtok(NULL, " ");
            }

            *size = count;
            fclose(file);
            return 1;
        }
    }

    fclose(file);
    return 0;
}
