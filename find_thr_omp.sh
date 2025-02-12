#!/bin/bash

matrices=(
    "Sandia/adder_dcop_32/adder_dcop_32.mtx"
    "Schenk_AFE/af_1_k101/af_1_k101.mtx"
    "Bai/af23560/af23560.mtx"
    "SNAP/amazon0302/amazon0302.mtx"
    "HB/bcsstk17/bcsstk17.mtx"
    "vanHeukelum/cage4/cage4.mtx"
    "Williams/cant/cant.mtx"
    "DRIVCAV/cavity10/cavity10.mtx"
    "Williams/cop20k_A/cop20k_A.mtx"
    "Janna/Cube_Coup_dt0/Cube_Coup_dt0.mtx"
    "IBM_EDA/dc1/dc1.mtx"
    "Botonakis/FEM_3D_thermal1/FEM_3D_thermal1.mtx"
    "Norris/lung2/lung2.mtx"
    "Williams/mac_econ_fwd500/mac_econ_fwd500.mtx"
    "HB/mcfe/mcfe.mtx"
    "Bai/mhd4800a/mhd4800a.mtx"
    "Bai/mhda416/mhda416.mtx"
    "Bai/mhda416/mhda416.mtx"
    "Janna/ML_Laplace/ML_Laplace.mtx"
    "Schenk/nlpkkt80/nlpkkt80.mtx"
    "Simon/olafu/olafu.mtx"
    "Bai/olm1000/olm1000.mtx"
    "Fluorem/PR02R/PR02R.mtx"
    "Simon/raefsky2/raefsky2.mtx"
    "Zitney/rdist2/rdist2.mtx"
    "SNAP/roadNet-PA/roadNet-PA.mtx"
    "Schmid/thermal1/thermal1.mtx"
    "Schmid/thermal2/thermal2.mtx"
    "Botonakis/thermomech_TK/thermomech_TK.mtx"
    "Williams/webbase-1M/webbase-1M.mtx"
    "HB/west2021/west2021.mtx"
)

modes=("-ompCSR" "-ompHLL")

THREADS_MAX=40
OUTPUT_FILE="threads.csv"

echo "Matrix,Mode,Threads,AvgTime(ms),AvgGFlops,BestTime(ms),BestGFlops" > "$OUTPUT_FILE"

echo ">>> Opening openmp..."
cd openmp || exit 1  # Se fallisce, esce con errore

echo ">>> Cleaning..."
make clean

echo ">>> Building..."
make all

for mat in "${matrices[@]}"; do
    for mode in "${modes[@]}"; do
        for threads in $(seq 1 $THREADS_MAX); do
            echo "Running with $threads threads on matrix $mat in mode $mode..."
            make run_openmp MAT="../../../matrici/MM/$mat" MODE="$mode" THREADS="$threads"
            
            awk -F',' -v mat="$mat" -v mode="$mode" -v threads="$threads" 'NR>1 { print mat "," mode "," threads "," $7 "," $10 "," $9 "," $12 }' performance.csv >> "$OUTPUT_FILE"
        done
    done
done

cd ..

# Usage: (DALLA ROOT DEL PROGETTO, NON DALLA CARTELLA openmp)
# ./threads.sh