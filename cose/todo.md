# Roadmap

## Note Generiche
- cage4 generale
- cant simmetrica
- amazon0302 pattern
- can24 entrambi

## Attività

1. **Implementazione HLL**
    - Implementare la rappresentazione HLL (da fare da soli)

2. **Implementazione CSR**
    - Implementazione CSR si trova online

3. **CUDA**
    - Vedere bene come funzionano CMake e l'indicizzazione dei thread

## Note Tecniche

- La struct che contiene i campi che rappresentano la matrice in ciascun formato resta la stessa sia in CUDA che in OpenMP.
- L'unica differenza è la gestione della memoria:
  - In CUDA, dobbiamo allocare la memoria sulla GPU.
  - In OpenMP, la gestione rimane sulla CPU.

## TODO

- [ ] Implementare la rappresentazione HLL
- [ ] Trovare e studiare l'implementazione CSR online
- [ ] Studiare CMake e l'indicizzazione dei thread in CUDA
- [ ] Allocare memoria sulla GPU per CUDA
- [ ] Gestire la memoria sulla CPU per OpenMP



## HLL
Il formato HLL è una variante ottimizzata di ELLPack, che divide la matrice in blocchi di righe per migliorare l'accesso alla memoria e la parallelizzazione.
Le idee chiave sono:

1. Si sceglie un HackSize → Ad esempio, 32 righe per blocco.
2. Si suddivide la matrice in blocchi di HackSize righe ciascuno.
3. Ogni blocco viene memorizzato in formato ELLPack, quindi avrà il proprio maxnz.
4. La struttura dati deve gestire più blocchi, quindi avremo un array di blocchi.
