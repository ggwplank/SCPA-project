#!/bin/bash

echo ">>> Opening openmp..."
cd openmp || exit 1  # Se fallisce, esce con errore

echo ">>> Cleaning..."
make clean

echo ">>> Building..."
if [[ "$1" == "run" ]]; then
    make run MAT="$2"
elif [[ "$1" == "run_openmp" ]]; then
    make run_openmp MAT="$2"
else
    echo "Usage: $0 [run|run_openmp] nome.mtx"
    exit 1
fi

cd ..

# Usage: (DALLA ROOT DEL PROGETTO, NON DALLA CARTELLA openmp)
# ./omp.sh run matrice.mtx
# ./omp.sh run_openmp matrice.mtx