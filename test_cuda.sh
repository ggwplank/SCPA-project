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
modes=("-serial" "-cudaCSR" "-cudaHLL")

echo ">>> Opening cuda..."
cd cuda || exit 1

echo ">>> Cleaning..."
make clean

echo ">>> Building..."
make all

for mat in "${matrices[@]}"; do
    for mode in "${modes[@]}"; do
        if [[ "$1" == "run" ]]; then
            make run MAT="../../../matrici/MM/$mat" MODE="$mode"
        elif [[ "$1" == "run_cuda" ]]; then
            make run_cuda MAT="../../../matrici/MM/$mat" MODE="$mode"
        else
            echo "Usage: $0 [run|run_cuda]"
            exit 1
        fi
    done
done

cd ..

# Usage: (DALLA ROOT DEL PROGETTO, NON DALLA CARTELLA cuda)
# ./cuda.sh run_cuda