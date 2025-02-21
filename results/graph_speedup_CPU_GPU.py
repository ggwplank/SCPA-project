import pandas as pd

# Carica i file CSV
cpu_file = "performance-OMP-17.csv" # Sostituisci con il percorso reale
gpu_file = "performance-CUDA-256.csv"  # Sostituisci con il percorso reale

# Legge i file CSV
cpu_df = pd.read_csv(cpu_file)
gpu_df = pd.read_csv(gpu_file)

# Rimuove l'estensione .mtx dai nomi delle matrici nel file GPU
gpu_df["Matrix"] = gpu_df["Matrix"].str.replace(".mtx", "", regex=False)

# Filtra solo le modalità CSR e HLL nei due DataFrame
cpu_df = cpu_df[cpu_df["Mode"].isin(["ompCSR", "ompHLL"])]
gpu_df = gpu_df[gpu_df["Mode"].isin(["cudaCSR", "cudaHLL"])]

# Seleziona solo le colonne necessarie e rinomina i campi
cpu_df = cpu_df[["Matrix", "Mode", "MedianTime(ms)"]].rename(columns={"MedianTime(ms)": "MedianTime_CPU"})
gpu_df = gpu_df[["Matrix", "Mode", "MedianTime(ms)"]].rename(columns={"MedianTime(ms)": "MedianTime_GPU"})

# Sostituiamo Mode per la CPU per far corrispondere CSR con CSR e HLL con HLL
cpu_df["Mode"] = cpu_df["Mode"].replace({"ompCSR": "cudaCSR", "ompHLL": "cudaHLL"})

# Merge dei dati su "Matrix" e "Mode" per confrontare correttamente
merged_df = pd.merge(cpu_df, gpu_df, on=["Matrix", "Mode"], how="inner")

# Calcola lo speedup
merged_df["Speedup"] = merged_df["MedianTime_CPU"] / merged_df["MedianTime_GPU"]

# Mostra i risultati
print(merged_df)

# Salva il risultato in un file CSV
merged_df.to_csv("speedup_results.csv", index=False)


