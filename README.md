# SCPA-project

## OpenMP

Per eseguire il nucleo di calcolo parallelo con OpenMP su tutte le matrici e ottenere un file di prestazioni e speedup, esegui:
```sh
./test_omp.sh run_openmp
```

### Cambiare il Numero di Thread

Per cambiare il numero di thread, modifica il parametro `THREADS` alla linea 39 in `test_omp.sh`.

## CUDA

Per eseguire il nucleo di calcolo parallelo con CUDA su tutte le matrici e ottenere un file di prestazioni e speedup, esegui:
```sh
./test_cuda.sh run_cuda
```
