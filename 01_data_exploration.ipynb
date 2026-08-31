# ============================================================
# Carga de librerías
# ============================================================

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import ListedColormap, BoundaryNorm
from IPython.display import HTML, display, Image
from base64 import b64encode
import os
import scipy.stats as stats
import seaborn as sns
from scipy.stats import expon, f_oneway, spearmanr
from scipy.signal import convolve
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import glob

from google.colab import drive
drive.mount('/content/drive')

# Ruta a tus archivos base
h5_path = "/content/drive/MyDrive/TESIS 2025/MR-0688-t1_results_curated.hdf5"
cluster_path = "/content/drive/MyDrive/TESIS 2025/MR-0688-t1.clusters.hdf5"
coord_path = "/content/drive/MyDrive/TESIS 2025/Coordinates_2.xlsx"

# ============================================================
# Lectura del archivo HDF5
# ============================================================

nombres = []
ruta = '/content/drive/MyDrive/TESIS 2025/MR-0688-t1_results_curated.hdf5'

with h5py.File(ruta, 'r') as f:
    def guardar_nombres(name, obj):
      if name != 'spiketimes':
        nombres.append(name)
    f.visititems(guardar_nombres)

# ============================================================
# Exploración de spiketimes
# ============================================================

len(nombres)

with h5py.File('/content/drive/MyDrive/TESIS 2025/MR-0688-t1_results_curated.hdf5', 'r') as f:
    def print_attrs(name, obj):
        print(name)
    f.visititems(print_attrs)

with h5py.File(ruta, 'r') as f:
    data18 = f['spiketimes/temp_18'][:]

# ============================================================
# Relación entre células y electrodos
# ============================================================

with h5py.File('/content/drive/MyDrive/TESIS 2025/MR-0688-t1.clusters.hdf5', 'r') as f:
    data18 = f['electrodes'][:]
df = pd.DataFrame(data18.T)
conteo_electrodos = df[0].value_counts().sort_index()

print(f"Número máximo de celulas: {conteo_electrodos.max()}")
print(f"Número mínimo de celulas: {conteo_electrodos.min()}")

# ============================================================
# Lectura de los archivos Excel.
# ============================================================

archivo_eventos = pd.read_excel('/content/drive/MyDrive/TESIS 2025/event_list_MR-0688-t1_bars_.xlsx')
#archivo_eventos = pd.read_excel('/content/drive/MyDrive/TESIS 2025/Coordinates_2.xlsx')
#archivo_eventos.iloc[0:65]
archivo_eventos

# Seleccionar las columnas deseadas
#columnas_deseadas = ['n_frames', 'start_event', 'end_event', 'start_next_event', 'extra_description']
#columnas_deseadas = ['end_event']
#archivo_eventos_seleccionado = archivo_eventos.loc[0:21, columnas_deseadas]
#print(archivo_eventos_seleccionado)

