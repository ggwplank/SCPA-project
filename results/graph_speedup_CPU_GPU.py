import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

# Carica il file CSV con i risultati
result_file = "speedup_results.csv"  # Sostituisci con il percorso corretto
results_df = pd.read_csv(result_file)

# Filtra solo le modalità CUDA CSR e CUDA HLL
results_df = results_df[results_df["Mode"].isin(["cudaCSR", "cudaHLL"])]

# Rinomina le modalità per la legenda
results_df["Mode"] = results_df["Mode"].replace({"cudaCSR": "CSR", "cudaHLL": "HLL"})

# Ordina i dati per nome matrice ignorando maiuscole/minuscole
results_df = results_df.sort_values(by="Matrix", key=lambda x: x.str.lower())

# Creazione del grafico con Seaborn
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x="Matrix", y="Speedup", hue="Mode", palette={"CSR": "red", "HLL": "blue"})

# Personalizzazioni
plt.xticks(rotation=90, ha="center", fontsize=10)  # Nomi matrici verticali
plt.ylabel("Speedup (CPU Time / GPU Time)")
plt.legend(title="Mode")

# Layout e salvataggio
plt.tight_layout()
plt.savefig("speedup_comparison_sns.png", dpi=300)
plt.show()
