#!/bin/bash

INPUT_FILE="openmp/performance.csv"
OUTPUT_FILE="openmp/speedup.csv"

echo "Matrix,Mode,Threads,Speedup_Avg,Speedup_Median" > "$OUTPUT_FILE"

declare -A serial_times_avg
declare -A serial_times_median

# Legge tutte le righe del file tranne l'header
awk -F',' 'NR>1 { print }' "$INPUT_FILE" | while IFS=',' read -r matrix m n nz mode threads avg_time median_time best_time avg_gflops median_gflops best_gflops passed diff reldiff iterations; do
    
    if [[ "$mode" == "serial" ]]; then
        serial_times_avg[$matrix]=$avg_time
        serial_times_median[$matrix]=$median_time
    fi
    
    if [[ "$mode" == "ompCSR" || "$mode" == "ompHLL" ]]; then
        if [[ -n "${serial_times_avg[$matrix]}" && "${serial_times_avg[$matrix]}" != "0" && -n "${serial_times_median[$matrix]}" && "${serial_times_median[$matrix]}" != "0" ]]; then
            speedup_avg=$(echo "scale=6; ${serial_times_avg[$matrix]} / $avg_time" | bc)
            speedup_median=$(echo "scale=6; ${serial_times_median[$matrix]} / $median_time" | bc)
            echo "$matrix,$mode,$threads,$speedup_avg,$speedup_median" >> "$OUTPUT_FILE"
        fi
    fi

done
