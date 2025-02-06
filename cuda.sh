#!/bin/bash

echo ">>> Opening cuda..."
cd cuda || exit 1  # Se fallisce, esce con errore

echo ">>> Cleaning..."
make clean

echo ">>> Building..."
if [[ "$1" == "run_cuda" ]]; then
    make run_cuda MAT="$2"
else
    echo "Usage: $0 [run_cuda] nome.mtx"
    exit 1
fi

cd ..

# Usage: (DALLA ROOT DEL PROGETTO, NON DALLA CARTELLA cuda)
# ./cuda.sh run_cuda matrice.mtx