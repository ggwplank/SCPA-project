#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

# Imposta lo stile ggplot per un aspetto semplice e pulito
plt.style.use('ggplot')

# Carica il file CSV
df = pd.read_csv('performance_per_thread.csv')

# Escludi la modalità "serial"
df = df[df['Mode'] != 'serial']

plt.figure(figsize=(10, 6))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Per ogni modalità (es. ompCSR, ompHLL, ecc.)
for i, mode in enumerate(df['Mode'].unique()):
    mode_data = df[df['Mode'] == mode]
    
    # Raggruppa per Threads e calcola la media dei GFlops
    grouped = mode_data.groupby('Threads', as_index=False).agg({'MedianGFlops': 'mean'})
    
    # Trova il thread che massimizza i GFlops
    best_idx = grouped['MedianGFlops'].idxmax()
    best_thread = grouped.loc[best_idx, 'Threads']
    best_gflops = grouped.loc[best_idx, 'MedianGFlops']
    
    # Stampa i risultati per la modalità corrente
    print(f"Modalità: {mode}")
    print(grouped)
    print(f"  Best thread: {best_thread} (MedianGFlops = {best_gflops:.3f})\n")
    
    # Seleziona un colore e traccia la curva
    col = colors[i % len(colors)]
    plt.plot(grouped['Threads'], grouped['MedianGFlops'], marker='o', color=col, label=mode)
    plt.scatter(best_thread, best_gflops, color=col, edgecolor='k', s=100, marker='D',
                label=f"{mode} Best")

plt.xlabel("Numero di Threads")
plt.ylabel("GFlops Mediani")
plt.title("MedianGFlops al variare del numero dei threads")
plt.legend()
plt.tight_layout()
plt.show()
