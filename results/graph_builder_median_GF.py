import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

# Funzione per caricare e filtrare i dati da un file CSV utilizzando gli indici
def load_and_filter_data(filename, matrix_name):
    df = pd.read_csv(filename)
    # Filtra i dati usando gli indici delle colonne: 'Matrix' è la colonna 0 e 'Mode' è la colonna 4
    df_filtered = df[(df.iloc[:, 0] == matrix_name) & (df.iloc[:, 4] == 'cudaCSR')]
    return df_filtered

# Nomi dei file CSV
files = [
    'performance-CUDA-128.csv',
    'performance-CUDA-256.csv',
    'performance-CUDA-512.csv',
    'performance-CUDA-1024.csv'
]

# Dimensioni dei blocchi come indici (correspondono ai nomi dei file)
block_sizes = [128, 256, 512, 1024]

# Carica il primo file per ottenere tutti i nomi delle matrici
df = pd.read_csv(files[0])

# Ottieni una lista di tutte le matrici uniche (eliminando i duplicati) usando l'indice (colonna 0)
matrix_names = df.iloc[:, 0].unique()

# Crea una mappa di colori con 30 colori distinti
colors = cm.get_cmap('tab20c', 30)  # Usa una colormap con 30 colori distinti

# Inizializza la figura per il grafico
plt.figure(figsize=(10, 6))

# Per ogni matrice, ripeti il processo e aggiungi una linea al grafico
for idx, matrix_name in enumerate(matrix_names):
    avg_gflops = []
    
    # Carica e filtra i dati per ogni dimensione di blocco
    for file in files:
        df_filtered = load_and_filter_data(file, matrix_name)
        if not df_filtered.empty:
            avg_gflops.append(df_filtered.iloc[0, 9])  # Usa l'indice per ottenere AvgGFlops (colonna 8)
        else:
            avg_gflops.append(None)  # Aggiungi None se non ci sono dati per questo file

    # Rimuovi i valori None da avg_gflops e i corrispondenti block_sizes
    block_sizes_filtered = [size for size, gflop in zip(block_sizes, avg_gflops) if gflop is not None]
    avg_gflops_filtered = [gflop for gflop in avg_gflops if gflop is not None]

    # Traccia la linea per questa matrice con un colore dalla colormap
    plt.plot(block_sizes_filtered, avg_gflops_filtered, marker='o', linestyle='-', color=colors(idx), label=matrix_name)

# Aggiungi titolo e etichette
plt.title('AvgGFlops per tutte le matrici')
plt.xlabel('Block Size')
plt.ylabel('Avg GFlops')
plt.grid(True)
plt.xticks(block_sizes)

# Aggiungi la legenda, spostata a lato
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=8)

# Usa tight_layout per migliorare la disposizione
plt.tight_layout()

# Crea la cartella 'graphs' se non esiste
if not os.path.exists('graphs'):
    os.makedirs('graphs')

# Salva il grafico nella cartella 'graphs'
plt.savefig('graphs/median_gflops_plot-CSR.png')

# Mostra il grafico
plt.show()
