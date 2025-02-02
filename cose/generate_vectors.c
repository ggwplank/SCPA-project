#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <time.h>
#include "mmio.h"

#define MAX_FILENAME 256

void process_matrix(const char *filepath, FILE *output_file) {
    FILE *f;
    MM_typecode matcode;
    int M, N, NZ;
    
    if ((f = fopen(filepath, "r")) == NULL) {
        fprintf(stderr, "Errore: impossibile aprire il file %s\n", filepath);
        return;
    }
    
    if (mm_read_banner(f, &matcode) != 0) {
        fprintf(stderr, "Errore: il file %s non è un Matrix Market valido.\n", filepath);
        fclose(f);
        return;
    }
    
    if (!mm_is_matrix(matcode)) {
        fprintf(stderr, "Errore: %s non è una matrice valida.\n", filepath);
        fclose(f);
        return;
    }
    
    if (mm_read_mtx_crd_size(f, &M, &N, &NZ) != 0) {
        fprintf(stderr, "Errore nella lettura della dimensione della matrice %s.\n", filepath);
        fclose(f);
        return;
    }
    fclose(f);
    
    // Genera un vettore casuale di dimensione N

const char *filename = strrchr(filepath, '/');
if (filename) {
    filename++;  // Sposta il puntatore dopo lo '/'
} else {
    filename = filepath;  // Se non c'è '/', usa il filepath originale
}
fprintf(output_file, "%s\n", filename);
for (int i = 0; i < N; i++) {
    double value = 0.1 + ((double)rand() / RAND_MAX) * (3.0 - 0.1);
    fprintf(output_file, "%lf ", value);
}
fprintf(output_file, "\n");
}

void process_directory(const char *input_folder, const char *output_filename) {
    DIR *dir;
    struct dirent *entry;
    char filepath[MAX_FILENAME];
    FILE *output_file = fopen(output_filename, "w");
    
    if (!output_file) {
        fprintf(stderr, "Errore: impossibile creare il file di output %s\n", output_filename);
        return;
    }
    
    if ((dir = opendir(input_folder)) == NULL) {
        fprintf(stderr, "Errore: impossibile aprire la cartella %s\n", input_folder);
        fclose(output_file);
        return;
    }
    
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') continue; // Ignora file nascosti
        
        if (snprintf(filepath, MAX_FILENAME, "%s/%s", input_folder, entry->d_name) >= MAX_FILENAME) {
            fprintf(stderr, "Errore: percorso file troppo lungo per %s\n", entry->d_name);
            continue;
        }
        
        process_matrix(filepath, output_file);
    }
    
    closedir(dir);
    fclose(output_file);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Uso: %s <cartella_matrici> <output_file>\n", argv[0]);
        return 1;
    }
    
    srand(time(NULL)); // Inizializza il generatore di numeri casuali
    process_directory(argv[1], argv[2]);
    
    return 0;
}
