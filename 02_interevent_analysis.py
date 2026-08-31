# ============================================================
# Segmentación de los datos por ángulo
# ============================================================

h5_path = '/content/drive/MyDrive/TESIS 2025/MR-0688-t1_results_curated.hdf5'

# Rango de tiempos
start_time, end_time, grado = 41128708, 42122545, 0 # 0
#start_time, end_time, grado = 1019020, 2369283, 45 # 45
#start_time, end_time, grado = 2369283, 3321542, 90 # 90
#start_time, end_time, grado = 3321542, 4624527, 135 # 135
#start_time, end_time, grado = 4624527, 5624399, 180 # 180
#start_time, end_time, grado = 5624399, 6927049, 225 # 225
#start_time, end_time, grado = 6927049, 7926921, 270 # 270
#start_time, end_time, grado = 7926921, 9316000, 315 # 315


# Inicializa el diccionario con claves de 0 a 254
spike_dict = {i: None for i in range(255)}

with h5py.File(h5_path, 'r') as f:
    spiketimes_group = f['spiketimes']

    for i in range(255):
        key = f'temp_{i}'
        if key in spiketimes_group:
            # Carga y filtra los tiempos
            data = spiketimes_group[key][:]
            filtered = data[(data >= start_time) & (data <= end_time)]
            spike_dict[i] = filtered.tolist()  # Guarda como lista

# ======================================================================
# Cálculo de tiempos inter-evento / Promedios / Desviaciones estándar
# ======================================================================

tiempos_por_angulo = {
    0: (41128708, 42122545),
    45: (42128580, 43425195),
    90: (43431231, 44425067),
    135: (44431103, 45728052),
    180: (45734088, 46727924),
    225: (46733960, 48030574),
    270: (48036610, 49030446),
    315: (49036482, 50333767),
}

for angulo, (start_time, end_time) in tiempos_por_angulo.items():
    intervalo_dict = tiempos_de_activacion_spikes(angulo, start_time, end_time)
    inter_eventos = unir_intereventos(intervalo_dict)

    inter_eventos = np.sort(inter_eventos)
    corte_80 = int(len(inter_eventos) * 1)
    inter_eventos_filtrado = inter_eventos[:corte_80]

    mean_interval = np.mean(inter_eventos_filtrado)
    std_interval = np.std(inter_eventos_filtrado)

    print(f"Ángulo {angulo}°:")
    print(f"  Promedio = {mean_interval:.4f} ms")
    print(f"  Desviación estándar = {std_interval:.4f} ms")
    print(f"{mean_interval:.4f} & {std_interval:.4f}\n")

# ============================================================
# Histogramas / Q-Q plots
# ============================================================

tiempos_por_angulo = {
    0: (41128708, 42122545),
    45: (42128580, 43425195),
    90: (43431231, 44425067),
    135: (44431103, 45728052),
    180: (45734088, 46727924),
    225: (46733960, 48030574),
    270: (48036610, 49030446),
    315: (49036482, 50333767),
}

for angulo, (start_time, end_time) in tiempos_por_angulo.items():
    intervalos_por_celula = tiempos_de_activacion_spikes(angulo, start_time, end_time)
    inter_eventos = unir_intereventos(intervalos_por_celula)
    inter_eventos = np.sort(inter_eventos)
    corte = int(len(inter_eventos) * 0.95)
    inter_eventos_filtrados = inter_eventos[:corte]

    # Ajuste exponencial
    lambda_hat = 1 / np.mean(inter_eventos_filtrados)
    x_vals = np.linspace(0, max(inter_eventos_filtrados), 200)
    pdf_expon = expon.pdf(x_vals, scale=1/lambda_hat)

    # Graficar histograma
    plt.figure(figsize=(8, 5))
    #plt.hist(inter_eventos_filtrados, bins=30, density=True, alpha=0.6, label="Datos (95% inferior)")
    plt.hist(inter_eventos_filtrados, bins=30, density=True, alpha=0.6, label="Datos")
    plt.plot(x_vals, pdf_expon, 'r-', label=f"Ajuste Exponencial\nλ = {lambda_hat:.5f}", linewidth=2)
    #plt.title(f"Histograma de intervalos inter-evento\nÁngulo {angulo}°")
    plt.xlabel("Tiempo entre eventos (sample)")
    plt.ylabel("Densidad")
    #plt.legend()
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # QQ-Plot contra distribución exponencial
    plt.figure(figsize=(8, 5))
    stats.probplot(inter_eventos_filtrados, dist="expon", sparams=(0, 1/lambda_hat), plot=plt)
    plt.title(f"")
    plt.xlabel("Cuantiles teóricos")
    plt.ylabel("Cuantiles observados")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
