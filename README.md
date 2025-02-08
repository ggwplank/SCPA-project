# SCPA-project


Per trasporre una matrice ELLPack, l'idea principale è cambiare l'ordine con cui i dati vengono memorizzati nella memoria. Questo significa scambiare righe e colonne, affinché ogni **colonna** della matrice (per la moltiplicazione) sia memorizzata in modo contiguo in memoria, favorendo l'accesso ottimale quando i thread CUDA o OpenMP accedono ai dati.

### Come trasporre una matrice ELLPack:
1. **Iterare su ogni colonna e riga**:
   - Attualmente, nella matrice ELLPack, ogni riga contiene i coefficienti non zero. Se vogliamo trasporla, dobbiamo fare in modo che per ogni colonna venga memorizzata una lista di coefficienti, esattamente come avviene per le righe.
   
2. **Creare un nuovo array per le colonne**:
   - Crea un array di righe e colonne in cui la **colonna** è trattata come una **riga**. Essenzialmente dovremo riarrangiare i dati in memoria.

3. **Aggiornare la struttura**:
   - Dobbiamo cambiare la struttura dei dati della matrice `ELLPackMatrix` in modo da riflettere il nuovo formato (con righe/colonne scambiate).

### Algoritmo:

1. **Trasposizione dei dati**:
   Per ogni riga `i` e colonna `j` della matrice originale, il valore `A[i][j]` nella matrice originale dovrebbe diventare `A_transposed[j][i]` nella matrice trasposta.

2. **Codifica in C**:
   Supponiamo che `ellpack->values` e `ellpack->col_indices` siano gli array contenenti i valori e gli indici della matrice:

   ```c
   void transpose_ELLPack(ELLPackMatrix *A) {
       // Creazione di una nuova struttura ELLPack per la matrice trasposta
       ELLPackMatrix *transposed = (ELLPackMatrix*)malloc(sizeof(ELLPackMatrix));
       transposed->rows = A->cols;
       transposed->cols = A->rows;
       transposed->maxnz = A->maxnz;

       // Allocazione delle nuove matrici per i valori e gli indici
       transposed->values = (double**)malloc(A->cols * sizeof(double*));
       transposed->col_indices = (int**)malloc(A->cols * sizeof(int*));

       for (int i = 0; i < A->cols; i++) {
           transposed->values[i] = (double*)calloc(A->maxnz, sizeof(double));
           transposed->col_indices[i] = (int*)calloc(A->maxnz, sizeof(int));
       }

       // Trasposizione dei dati
       for (int i = 0; i < A->rows; i++) {
           for (int j = 0; j < A->maxnz; j++) {
               if (A->col_indices[i][j] != -1) {
                   int col = A->col_indices[i][j];  // Colonna originale
                   int row = i;                      // Riga originale
                   int idx = 0;                      // Indice per la colonna trasposta

                   // Trova la posizione giusta nella matrice trasposta
                   while (transposed->col_indices[col][idx] != -1) {
                       idx++;
                   }

                   // Memorizza i dati trasposti
                   transposed->values[col][idx] = A->values[i][j];
                   transposed->col_indices[col][idx] = row;
               }
           }
       }

       // Libera la memoria della matrice originale se non serve più
       free_ELL(A);

       // Copia la matrice trasposta nella struttura originale
       *A = *transposed;
   }
   ```

### Descrizione dei passaggi:

1. **Creazione della nuova struttura** `transposed`: 
   - Creiamo una nuova matrice con righe e colonne invertite.
   
2. **Allocazione della memoria**:
   - Allochiamo memoria per `values` e `col_indices` nelle dimensioni trasposte.

3. **Trasposizione**:
   - Per ogni elemento nella matrice originale (`A->col_indices[i][j]` e `A->values[i][j]`), trasferiamo i valori nella nuova struttura, invertendo i ruoli di riga e colonna.

4. **Liberazione e assegnazione**:
   - Una volta trasposta la matrice, liberiamo la memoria della matrice originale e assegniamo la matrice trasposta al posto dell'originale.

### **Conclusione**:
Trasporre una matrice ELLPack implica semplicemente riorganizzare i dati in modo che le colonne siano memorizzate come righe e viceversa. Una volta che la matrice è trasposta, possiamo utilizzarla per operazioni parallele in CUDA, dove questo layout migliora l'accesso alla memoria e aumenta le prestazioni, specialmente con l'accesso **coalizzato**.


Se stai utilizzando **CUDA**, la trasposizione della matrice ELLPack dovrebbe essere fatta prima di iniziare la parte di calcolo parallelo, perché il formato trasposto ottimizza l'accesso alla memoria per le operazioni parallele.

### In quale parte del codice chiamare la trasposizione

La trasposizione dovrebbe essere chiamata **prima** di qualsiasi operazione che coinvolga l'elaborazione parallela, come la moltiplicazione matrice-vettore, se decidi di utilizzare CUDA, poiché il formato trasposto migliorerebbe le prestazioni grazie alla disposizione ottimale in memoria per l'accesso ai dati.

Ad esempio, se hai un codice come questo:

```c
int main() {
    // Step 1: Caricamento della matrice da file o costruzione
    MatrixEntry *entries = load_matrix("cag4.mtx");
    ELLPackMatrix *ellpack = convert_to_ELL(M, N, NZ, entries);

    // Step 2: Trasposizione della matrice (se si sta usando CUDA)
    transpose_ELLPack(ellpack); // Chiamata alla funzione di trasposizione

    // Step 3: Creazione dei dati e memoria per CUDA (dopo la trasposizione)
    // Esegui la moltiplicazione matrice-vettore, ad esempio

    // Calcolo parallelizzato con CUDA
    cuda_multiplication(ellpack, x, y);

    // Step 4: Liberazione della memoria
    free_ELL(ellpack);
}
```

### Dettagli sulla posizione della chiamata:

1. **Subito dopo aver costruito la matrice ELLPack**:
   Dopo aver costruito la matrice ELLPack (`convert_to_ELL()`), puoi chiamare la funzione `transpose_ELLPack()` per ottenere il layout ottimizzato per CUDA. La matrice viene trasposta **prima** che venga usata nei calcoli paralleli.
   
2. **Dopo la trasposizione**:
   Dopo aver effettuato la trasposizione, puoi passare la matrice trasposta alle funzioni che eseguono la moltiplicazione matrice-vettore utilizzando CUDA. Questo garantirà che gli accessi ai dati siano efficienti grazie al layout trasposto.

### Perché trasporre prima dell'elaborazione parallela?

- **Accesso alla memoria coalizzato in CUDA**: Se la matrice è memorizzata in formato trasposto, le colonne della matrice sono adiacenti in memoria. Ciò significa che i thread CUDA che operano su una singola riga della matrice (utilizzando un thread per riga) possono accedere ai dati in modo coalizzato. Ciò migliora le prestazioni delle operazioni parallele.
  
- **Nessuna trasposizione necessaria durante il calcolo**: Trasporre la matrice una sola volta all'inizio (prima del calcolo parallelo) è molto più efficiente che cercare di trasporre continuamente i dati all'interno di ogni kernel CUDA. La matrice trasposta è pronta per essere utilizzata nel calcolo parallelo.

### Esempio di utilizzo in CUDA

Se hai una funzione che esegue la moltiplicazione matrice-vettore usando CUDA, dopo la trasposizione puoi semplicemente passare la matrice trasposta come input ai tuoi kernel CUDA. Ecco un esempio di come potrebbe apparire il flusso:

```c
__global__ void cuda_multiply_kernel(double *values, int *col_indices, double *x, double *y, int rows, int maxnz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < rows) {
        double result = 0.0;
        for (int j = 0; j < maxnz; j++) {
            int col = col_indices[idx * maxnz + j];
            if (col != -1) {
                result += values[idx * maxnz + j] * x[col];
            }
        }
        y[idx] = result;
    }
}

void cuda_multiplication(ELLPackMatrix *A, double *x, double *y) {
    // Allocate memory on device
    double *d_values, *d_x, *d_y;
    int *d_col_indices;

    cudaMalloc((void**)&d_values, A->rows * A->maxnz * sizeof(double));
    cudaMalloc((void**)&d_col_indices, A->rows * A->maxnz * sizeof(int));
    cudaMalloc((void**)&d_x, A->cols * sizeof(double));
    cudaMalloc((void**)&d_y, A->rows * sizeof(double));

    // Copy data to device
    cudaMemcpy(d_values, A->values, A->rows * A->maxnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, A->col_indices, A->rows * A->maxnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, A->cols * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel to multiply matrix by vector
    int threadsPerBlock = 256;
    int blocks = (A->rows + threadsPerBlock - 1) / threadsPerBlock;
    cuda_multiply_kernel<<<blocks, threadsPerBlock>>>(d_values, d_col_indices, d_x, d_y, A->rows, A->maxnz);

    // Copy result back to host
    cudaMemcpy(y, d_y, A->rows * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_values);
    cudaFree(d_col_indices);
    cudaFree(d_x);
    cudaFree(d_y);
}
```

### Sintesi dei passaggi:

1. **Creazione della matrice ELLPack**: Usa `convert_to_ELL` per costruire la matrice ELLPack.
2. **Trasposizione della matrice**: Usa `transpose_ELLPack` subito dopo aver costruito la matrice ELLPack.
3. **Esecuzione dei calcoli in CUDA**: Dopo la trasposizione, usa la matrice trasposta per eseguire le operazioni parallele in CUDA (ad esempio, la moltiplicazione matrice-vettore).
   
In questo modo, ottieni un layout della matrice che è ottimizzato per l'elaborazione parallela in CUDA, migliorando le prestazioni dei calcoli!