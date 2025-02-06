#!/bin/bash

echo ">>> Opening openmp..."
cd openmp || exit 1  # Se fallisce, esce con errore

echo ">>> Cleaning..."
make clean

echo ">>> Building..."
if [[ "$1" == "run" ]]; then
    make run MAT="$2" MODE="$3"
elif [[ "$1" == "run_openmp" ]]; then
    make run_openmp MAT="$2" MODE="$3" THREADS="$4"
else
    echo "Usage: $0 [run|run_openmp] matrix.mtx mode {num_threads}"
    exit 1
fi

cd ..

# Usage: (DALLA ROOT DEL PROGETTO, NON DALLA CARTELLA openmp)
# ./omp.sh run matrix.mtx mode
# ./omp.sh run_openmp matrix.mtx mode num_threads