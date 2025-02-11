#!/bin/bash

matrices=(
    "Bai/mhda416/mhda416.mtx"
    "Bai/olm1000/olm1000.mtx"
    "Bai/mhd4800a/mhd4800a.mtx"
    "Bai/af23560/af23560.mtx"
    "Bai/mhda416/mhda416.mtx"
    "Botonakis/thermomech_TK/thermomech_TK.mtx"
    "Botonakis/FEM_3D_thermal1/FEM_3D_thermal1.mtx"
    "DRIVCAV/cavity10/cavity10.mtx"
    "Fluorem/PR02R/PR02R.mtx"
    "HB/bcsstk17/bcsstk17.mtx"
    "HB/mcfe/mcfe.mtx"
    "HB/west2021/west2021.mtx"
    "IBM_EDA/dc1/dc1.mtx"
    "Janna/Cube_Coup_dt0/Cube_Coup_dt0.mtx"
    "Janna/ML_Laplace/ML_Laplace.mtx"
    "Norris/lung2/lung2.mtx"
    "Sandia/adder_dcop_32/adder_dcop_32.mtx"
    "Schenk/nlpkkt80/nlpkkt80.mtx"
    "Schenk_AFE/af_1_k101/af_1_k101.mtx"
    "Schmid/thermal1/thermal1.mtx"
    "Schmid/thermal2/thermal2.mtx"
    "Simon/raefsky2/raefsky2.mtx"
    "Simon/olafu/olafu.mtx"
    "SNAP/amazon0302/amazon0302.mtx"
    "SNAP/roadNet-PA/roadNet-PA.mtx"
    "vanHeukelum/cage4/cage4.mtx"
    "Williams/mac_econ_fwd500/mac_econ_fwd500.mtx"
    "Williams/cop20k_A/cop20k_A.mtx"
    "Williams/webbase-1M/webbase-1M.mtx"
    "Williams/cant/cant.mtx"
    "Zitney/rdist2/rdist2.mtx"
)
modes=("-serial" "-ompCSR" "-ompHLL")

THREADS=16

echo ">>> Opening cuda..."
cd openmp || exit 1  # Se fallisce, esce con errore

echo ">>> Cleaning..."
make clean

echo ">>> Building..."
for mat in "${matrices[@]}"; do
    for mode in "${modes[@]}"; do
        if [[ "$1" == "run" ]]; then
            make run MAT="../../../matrici/MM/$mat" MODE="$mode"
        elif [[ "$1" == "run_openmp" ]]; then
            make run_openmp MAT="../../../matrici/MM/$mat" MODE="$mode" THREADS="$THREADS"
        else
            echo "Usage: $0 [run|run_openmp]"
            exit 1
        fi
    done
done

cd ..

# Usage: (DALLA ROOT DEL PROGETTO, NON DALLA CARTELLA cuda)
# ./cuda.sh run_cuda