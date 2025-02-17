import pandas as pd
import matplotlib.pyplot as plt
import os

def compare_matrices(file_path1, file_path2, output_path):
    # Carica il primo file CSV
    df1 = pd.read_csv(file_path1)

    # Calcola le metriche richieste
    df1['avg_nnz_per_row'] = df1['NZ'] / df1['M']
    df1['density'] = df1['NZ'] / (df1['M'] * df1['N'])

    # Filtra le matrici secondo la condizione specificata e fai una copia esplicita
    df1_filtered = df1[(df1['avg_nnz_per_row'] < 25) | (df1['density'] > 1.5e-3)].copy()

    # Calcola la grandezza della matrice
    df1_filtered.loc[:, 'Size'] = df1_filtered['NZ'] / (df1_filtered['M'] * df1_filtered['N'])

    # Carica il secondo file CSV
    df2 = pd.read_csv(file_path2)

    # Calcola le metriche richieste
    df2['avg_nnz_per_row'] = df2['NZ'] / df2['M']
    df2['density'] = df2['NZ'] / (df2['M'] * df2['N'])

    # Filtra le matrici secondo la condizione specificata e fai una copia esplicita
    df2_filtered = df2[(df2['avg_nnz_per_row'] < 25) | (df2['density'] > 1.5e-3)].copy()

    # Calcola la grandezza della matrice
    df2_filtered.loc[:, 'Size'] = df2_filtered['NZ'] / (df2_filtered['M'] * df2_filtered['N'])

    # Ordina i dati per grandezza della matrice
    df1_sorted = df1_filtered.sort_values(by='Size')
    df2_sorted = df2_filtered.sort_values(by='Size')

    # Seleziona solo le matrici comuni tra i due file
    common_matrices = set(df1_sorted['Matrix']).intersection(set(df2_sorted['Matrix']))

    # Filtra entrambi i DataFrame per contenere solo le matrici comuni
    df1_common = df1_sorted[df1_sorted['Matrix'].isin(common_matrices)]
    df2_common = df2_sorted[df2_sorted['Matrix'].isin(common_matrices)]

    # Ordina per grandezza della matrice
    df1_common = df1_common.sort_values(by='Size')
    df2_common = df2_common.sort_values(by='Size')

    # Traccia il grafico a barre per confrontare le matrici
    plt.figure(figsize=(12, 6))

    # Indici per posizionare le barre
    x = range(len(df1_common))

    # Crea le barre per il primo file
    plt.bar([i - 0.2 for i in x], df1_common['MedianGFlops'], width=0.4, label='Row based', color='skyblue')

    # Crea le barre per il secondo file
    plt.bar([i + 0.2 for i in x], df2_common['MedianGFlops'], width=0.4, label='Warp', color='lightgreen')

    # Etichetta gli assi
    plt.xlabel("Matrice")
    plt.ylabel("Median GFlops")

    # Rotazione delle etichette sull'asse X per una migliore leggibilità
    plt.xticks(x, df1_common['Matrix'].str.replace('.mtx', '', regex=False), rotation=45, ha='right')

    # Aggiungi la griglia per migliorare la leggibilità
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Aggiungi la legenda
    plt.legend()

    # Crea la cartella 'graphs' se non esiste
    if not os.path.exists('graphs'):
        os.makedirs('graphs')

    # Salva il grafico nella cartella 'graphs'
    plt.savefig(output_path, bbox_inches='tight')

    # Mostra il grafico
    plt.show()

# Esempio di utilizzo
file_path1 = "performance-CSR-warp.csv"
file_path2 = "performance-CSR.csv"
output_path = "graphs/littleMAT-CSR-comaprison.png"
compare_matrices(file_path1, file_path2, output_path)
