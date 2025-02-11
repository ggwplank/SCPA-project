#!/bin/bash

INPUT_FILE="cuda/performance.csv"
OUTPUT_FILE="cuda/speedup.csv"

echo "Matrix,Mode,Speedup_Avg,Speedup_Median" > "$OUTPUT_FILE"

declare -A serial_times_avg
declare -A serial_times_median

# Legge tutte le righe del file tranne l'header
awk -F',' 'NR>1 { print }' "$INPUT_FILE" | while IFS=',' read -r matrix m n nz mode avg_time median_time best_time avg_gflops median_gflops best_gflops passed iterations; do
    
    if [[ "$mode" == "serial" ]]; then
        serial_times_avg[$matrix]=$avg_time
        serial_times_median[$matrix]=$median_time
    fi
    
    if [[ "$mode" == "cudaCSR" || "$mode" == "cudaHLL" ]]; then
        if [[ -n "${serial_times_avg[$matrix]}" && "${serial_times_avg[$matrix]}" != "0" && -n "${serial_times_median[$matrix]}" && "${serial_times_median[$matrix]}" != "0" ]]; then
            speedup_avg=$(echo "scale=6; ${serial_times_avg[$matrix]} / $avg_time" | bc)
            speedup_median=$(echo "scale=6; ${serial_times_median[$matrix]} / $median_time" | bc)
            echo "$matrix,$mode,$threads,$speedup_avg,$speedup_median" >> "$OUTPUT_FILE"
        fi
    fi

done
