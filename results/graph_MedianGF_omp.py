#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Imposta lo stile
plt.style.use('ggplot')

# Carica il file CSV
df = pd.read_csv('performance-CUDA-128.csv')

# Rimuove l'estensione .mtx dalla colonna "Matrix"
df["Matrix"] = df["Matrix"].str.replace(r'\.mtx$', '', regex=True)

# Definisce l'ordine delle modalità
mode_order = ["serial", "cudaCSR", "cudaHLL"]
df["Mode"] = pd.Categorical(df["Mode"], categories=mode_order, ordered=True)

# Ordina le matrici alfabeticamente PRIMA di estrarre l'ordine
df = df.sort_values(by="Matrix", ascending=True)

# Ora otteniamo l'ordine alfabetico corretto
matrix_order = sorted(df["Matrix"], key=str.lower)

# Imposta il grafico
plt.figure(figsize=(12, 6))

# Plotta il grafico con l'ordine alfabetico forzato
sns.barplot(x="Matrix", y="MedianGFlops", hue="Mode", data=df, 
            palette="tab10", hue_order=mode_order, order=matrix_order)

# Etichette e titolo
plt.xlabel("Matrice")
plt.ylabel("GFlops Mediani")
plt.title("GFlops mediani per matrice e formato")
plt.xticks(rotation=90)  # Ruota i nomi delle matrici per leggibilità

# Mostra la legenda con ordine corretto
plt.legend(title="Formato")

# Migliora la disposizione e mostra il grafico
plt.tight_layout()
plt.show()
