# ===================================================================================
# Construcción de la matriz de electrodos / Grafo 16 × 16 / Matrices de activación
# ===================================================================================

#!pip install networkx matplotlib pandas openpyxl h5py

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import ast
import h5py

def generar_diccionario_coord_indices(df_coords, columna='Coordinates'):
    diccionario = {}
    for i, coord_str in enumerate(df_coords[columna]):
        coord = ast.literal_eval(coord_str)
        x = int(coord[0] // 100)
        y = int(coord[1] // 100)
        diccionario[(x, y)] = i
    return diccionario

def visualizar_diccionario(ax, diccionario, size=16):
    for (x, y), valor in diccionario.items():
        ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor='black'))
        ax.text(x + 0.5, y + 0.5, str(valor), va='center', ha='center', fontsize=8)

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect('equal')
    tick_positions = [i + 0.5 for i in range(size)]
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(range(size))
    ax.set_yticklabels(range(size))
    ax.grid(False)

def crear_grafo_16x16_sin_esquinas():
    N = 15
    G = nx.Graph()
    V = [(x, y) for x in range(N+1) for y in range(N+1)
         if not ((x == 0 or x == N) and (y == 0 or y == N))]
    G.add_nodes_from(V)
    for (x, y) in V:
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor = (x+dx, y+dy)
            if neighbor in V:
                G.add_edge((x, y), neighbor)
    return G

def visualizar_grafo(ax, G, size=16):
    pos = {(x, y): (x + 0.5, y + 0.5) for (x, y) in G.nodes()}
    nx.draw(G, pos, node_size=20, with_labels=False, edge_color='gray', ax=ax)
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

def visualizar_celulas_por_electrodo(ax, conteo, coord_dict, size=16):
    matriz_celulas = np.zeros((size, size))
    for (x, y), index in coord_dict.items():
        cantidad = conteo.get(index, 0)  # Si el electrodo no tiene células, poner 0
        matriz_celulas[y, x] = cantidad  # y fila, x columna

    im = ax.imshow(matriz_celulas, origin='lower', cmap='YlOrRd')
    for y in range(size):
        for x in range(size):
            valor = int(matriz_celulas[y, x])
            if valor > 0:
                ax.text(x, y, str(valor), va='center', ha='center', fontsize=8)

    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_title("Células por electrodo")
    ax.set_aspect('equal')
    #plt.colorbar(im, ax=ax, shrink=0.7)

# --- CARGA DE DATOS ---
df_coords = pd.read_excel('/content/drive/MyDrive/TESIS 2025/Coordinates_2.xlsx')
dicc_coords = generar_diccionario_coord_indices(df_coords)

with h5py.File('/content/drive/MyDrive/TESIS 2025/MR-0688-t1.clusters.hdf5', 'r') as f:
    data18 = f['electrodes'][:]
df_electrodos = pd.DataFrame(data18.T)
conteo = df_electrodos[0].value_counts().sort_index()

# Crear grafo
grafo = crear_grafo_16x16_sin_esquinas()

# Visualizar los tres gráficos
fig, axs = plt.subplots(1, 3, figsize=(21, 7))

# 1. Visualizar la matriz de IDs de electrodos
visualizar_diccionario(axs[0], dicc_coords, size=16)
axs[0].set_title("Matriz de índices")

# 2. Visualizar el grafo
visualizar_grafo(axs[1], grafo, size=16)
axs[1].set_title("Grafo sin esquinas")

# 3. Visualizar la cantidad de células por electrodo
visualizar_celulas_por_electrodo(axs[2], conteo, dicc_coords, size=16)

plt.tight_layout()
plt.show()

# ============================================================
# Matrices únicas
# ============================================================

# Ángulo y bloques asociados
#angulo, bloque_inicio, bloque_fin =   0, 0, 9
#angulo, bloque_inicio, bloque_fin =  45, 0, 12
#angulo, bloque_inicio, bloque_fin =  90, 0, 9
#angulo, bloque_inicio, bloque_fin = 135, 0, 12
#angulo, bloque_inicio, bloque_fin = 180, 0, 9
#angulo, bloque_inicio, bloque_fin = 225, 0, 12
#angulo, bloque_inicio, bloque_fin = 270, 0, 9
angulo, bloque_inicio, bloque_fin = 315, 0, 12

# Inicialización
clave_to_index = {}
claves_unicas = []
transiciones = None
prev_clave = None  # para almacenar el estado anterior

def matriz_a_clave(m):
    return tuple(m.flatten())

# Procesamiento por bloque
for i in range(bloque_inicio, bloque_fin + 1):
    ruta = f"/content/drive/MyDrive/TESIS 2025/matrices_por_bloques/matrices_reales_{angulo}_bloque_{i}.npz"
    print(f"Procesando bloque {i}...")
    data = np.load(ruta, allow_pickle=True)
    matrices = data["matrices"]

    for m in matrices:
        clave = matriz_a_clave(m)

        # Si es una nueva configuración, registrarla
        if clave not in clave_to_index:
            idx = len(claves_unicas)
            clave_to_index[clave] = idx
            claves_unicas.append(clave)

            # Expandir la matriz de transiciones si es necesario
            if transiciones is None:
                transiciones = np.zeros((1, 1), dtype=int)
            else:
                n = len(claves_unicas)
                nueva = np.zeros((n, n), dtype=int)
                nueva[:transiciones.shape[0], :transiciones.shape[1]] = transiciones
                transiciones = nueva

        actual = clave_to_index[clave]

        # Si hay una transición válida (desde prev_clave → actual)
        if prev_clave is not None:
            transiciones[prev_clave, actual] += 1

        prev_clave = actual

print(f"\n✅ Total de configuraciones únicas: {len(claves_unicas)}")

# Crear DataFrame ordenado
labels = [f"M{i}" for i in range(len(claves_unicas))]
df = pd.DataFrame(transiciones, index=labels, columns=labels)

# Guardar resultado
ruta_salida = f"/content/drive/MyDrive/TESIS 2025/matriz_transiciones/matriz_transiciones_reales_angulo_{angulo}.csv"
df.to_csv(ruta_salida)

# Convertir claves_unicas a arrays de 16x16 y guardar
matrices_unicas = np.array([np.array(clave).reshape(16, 16) for clave in claves_unicas], dtype=np.uint8)

# Guardar como .npz comprimido
np.savez_compressed(
    f"/content/drive/MyDrive/TESIS 2025/matrices_unicas/matrices_unicas_angulo_{angulo}.npz",
    matrices=matrices_unicas
)

archivo_transiciones = pd.read_csv('/content/drive/MyDrive/TESIS 2025/matriz_transiciones/matriz_transiciones_reales_angulo_0.csv')
archivo_transiciones

#graficar_matriz_unica(angulo, indice)
graficar_matriz_unica(0,0)

# ============================================================
# Matrices Markovianas
# ============================================================

angulo = 0

# [0, 45, 90, 135, 180, 225, 270, 315]

ruta_entrada = f"/content/drive/MyDrive/TESIS 2025/matriz_transiciones/matriz_transiciones_reales_angulo_{angulo}.csv"
df = pd.read_csv(ruta_entrada, index_col=0)
df.columns = df.columns.astype(str)
df.index = df.index.astype(str)

df = df.loc[df.index.intersection(df.columns), df.columns.intersection(df.index)]
df_markov = df.div(df.sum(axis=1), axis=0).fillna(0)

ruta_salida = f"/content/drive/MyDrive/TESIS 2025/matriz_markov_angulo_{angulo}.csv"
df_markov.to_csv(ruta_salida)

df_markov

# Obtener coordenadas (x, y) activas en el tiempo i
ii=1
electrodos_activos_t0 = [coord for coord, tiempos in extended_coord_spikes.items() if ii in tiempos]

# Mostrar resultados
print(f"Electrodos activos en t={ii}: {electrodos_activos_t0}")
print(f"Total de electrodos activos: {len(electrodos_activos_t0)}")

# Crear matriz vacía
grid_size = 16
matriz_t0 = np.zeros((grid_size, grid_size))

# Activar coordenadas donde t = 0 aparece
for (x, y), tiempos in extended_coord_spikes.items():
    if ii in tiempos:
        matriz_t0[y, x] = 1  # Recordar: [fila, columna] = [y, x]

# Visualización
plt.figure(figsize=(6, 6))
plt.imshow(matriz_t0, cmap='Reds', vmin=0, vmax=1)
plt.title(f"Electrodos activos en t = {ii}")
plt.xticks(np.arange(grid_size))
plt.yticks(np.arange(grid_size))
plt.gca().invert_yaxis()
plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
plt.show()
