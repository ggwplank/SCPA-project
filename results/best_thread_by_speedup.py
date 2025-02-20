#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

# Imposta lo stile ggplot per un aspetto semplice e pulito
plt.style.use('ggplot')

# Carica il file CSV relativo allo speedup (modifica il nome se necessario)
df = pd.read_csv('speedup_per_thread.csv')

# Escludi la modalità "serial" se presente
df = df[df['Mode'] != 'serial']

plt.figure(figsize=(10, 6))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Analizza i dati per ogni modalità (ad es. ompCSR, ompHLL, etc.)
for i, mode in enumerate(df['Mode'].unique()):
    mode_data = df[df['Mode'] == mode]
    
    # Raggruppa per numero di thread e calcola la media dello Speedup
    grouped = mode_data.groupby('Threads', as_index=False).agg({'Speedup_Median': 'mean'})
    
    # Trova il thread che massimizza lo Speedup
    best_idx = grouped['Speedup_Median'].idxmax()
    best_thread = grouped.loc[best_idx, 'Threads']
    best_speedup = grouped.loc[best_idx, 'Speedup_Median']
    
    # Stampa i risultati per la modalità corrente
    print(f"Modalità: {mode}")
    print(grouped)
    print(f"  Best thread: {best_thread} (Speedup_Median = {best_speedup:.3f})\n")
    
    # Seleziona un colore e traccia la curva
    col = colors[i % len(colors)]
    plt.plot(grouped['Threads'], grouped['Speedup_Median'], marker='o', color=col, label=mode)
    plt.scatter(best_thread, best_speedup, color=col, edgecolor='k', s=100, marker='D',
                label=f"{mode} Best")

plt.xlabel("Numero di Threads")
plt.ylabel("Speedup_Median")
plt.title("Speedup_Median vs Threads per Modalità")
plt.legend()
plt.tight_layout()
plt.show()
