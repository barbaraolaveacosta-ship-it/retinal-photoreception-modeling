# ===================================================================================
# Simulación del modelo no Markoviano / AUC / Precision / Sensibilidad
# Ángulo 0
# ===================================================================================

angulo, bloque_inicio, bloque_fin = 0, 41128708, 42122545

tiempos_activacion = tiempos_de_activacion_spikes(
    grado=angulo,
    start_time=bloque_inicio,
    end_time=bloque_fin
)

coord_spikes = generar_extended_coord_spikes(
    start_time=bloque_inicio,
    end_time=bloque_fin
)

tiempos_barra = intervalos_barra(angulo)
ini, fin = tiempos_barra[0]
duracion_total = fin-ini
duracion_pixel = duracion_total / 16

intervalos_por_electrodo = generar_diccionario_intervalos_barras(
    generar_extended_coord_spikes=coord_spikes,
    tiempos_barra=tiempos_barra,
    angulo=angulo,
    duracion=duracion_total,
    start_time=bloque_inicio
)

celula_a_coordenada = obtener_celula_a_coordenada(
    cluster_path,
    coord_path)

# --- EJECUTAR EL PROCESO ---
resultados_finales = optimizar_alpha_para_todas_las_celulas(
    tiempos_activacion,
    celula_a_coordenada,
    intervalos_por_electrodo
)

# Mostrar resultados
#for celula, datos in resultados_finales.items():
#    print(f"Célula {celula}: α={datos['alpha']:.6f}, AUC={datos['auc']:.4f}")

# Convertir a DataFrame
df_resultados = pd.DataFrame.from_dict(resultados_finales, orient='index')
df_resultados.index.name = 'celula'

# Montar Drive (solo necesitas hacerlo una vez por sesión)
drive.mount('/content/drive')

# Guardar en Drive como CSV
ruta_csv = f'/content/drive/MyDrive/resultados_modelo_{angulo}_grados.csv'
df_resultados.to_csv(ruta_csv)

# 5. Confirmación visual
print(f"\n✅ Resultados guardados en: {ruta_csv}")

# ===================================================================================
# Comparación entre simulación y datos
# Ángulo 0
# ===================================================================================

ruta_csv = '/content/drive/MyDrive/resultados_modelo_0_grados.csv'
df_resultados = pd.read_csv(ruta_csv, index_col='celula')

for celula_str, fila in df_resultados.iterrows():
    celula = int(celula_str)
    alpha = fila['alpha']
    auc = fila['auc']

    if celula not in tiempos_activacion:
        continue

    diferencias = tiempos_activacion[celula]
    tiempos_activacion_real = np.cumsum(diferencias)
    conteo = np.arange(len(tiempos_activacion_real))

    tiempos_step = np.insert(tiempos_activacion_real, 0, 0)
    conteo_step = np.insert(conteo, 0, 0)

    # Obtener estímulo U_t
    coord = celula_a_coordenada.get(celula)
    intervalos = list(intervalos_por_electrodo.get(coord, {}).values())
    T = max(tiempos_step[-1], max((fin for _, fin in intervalos), default=0)) + 1000
    T = int(T)

    U_t = np.zeros(T)
    for inicio, fin in intervalos:
        U_t[inicio:fin] = 1

    # Simular señal original
    V_simulado = simular_modelo_no_markoviano(U_t, alpha)

    # Interpolación del conteo real sobre la escala completa
    conteo_interp_real = np.interp(np.arange(T), tiempos_step, conteo_step)

    # Escalamiento de simulación al rango del conteo real
    p98_conteo = np.percentile(conteo_interp_real, 98)
    p98_V = np.percentile(V_simulado, 98)
    escala = p98_conteo / p98_V if p98_V > 0 else 1.0
    V_simulado_escalado = V_simulado * escala

    # Calcular diferencia con misma escala
    diferencia = V_simulado_escalado - conteo_interp_real

    # Crear gráfico con 3 paneles
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.5), sharex=False)

    # Panel izquierdo: spikes reales (escala real)
    axs[0].step(tiempos_step, conteo_step, where='post', label='Spikes reales', linewidth=1.5)
    for t_inicio, t_fin in intervalos:
        axs[0].axvspan(t_inicio, t_fin, color='cyan', alpha=0.5)
    axs[0].set_title(f'Célula {celula} - Spikes')
    axs[0].set_xlabel('Tiempo (samples)')
    axs[0].set_ylabel('Conteo acumulado')
    axs[0].grid(True)
    axs[0].legend()

    # Panel central: simulación escalada
    axs[1].plot(V_simulado_escalado, label='Señal simulada', color='purple', linewidth=1)
    for t_inicio, t_fin in intervalos:
        axs[1].axvspan(t_inicio, t_fin, color='cyan', alpha=0.5)
    axs[1].set_title('Simulación $V_t$')
    axs[1].set_xlabel('Tiempo (samples)')
    axs[1].set_ylabel('$V_t$')
    axs[1].grid(True)
    axs[1].legend()

    # Panel derecho: diferencia con misma escala
    axs[2].plot(diferencia, label='Diferencia', color='orange', linewidth=1)
    axs[2].axhline(0, color='red', linestyle='--', linewidth=0.8)
    for t_inicio, t_fin in intervalos:
        axs[2].axvspan(t_inicio, t_fin, color='cyan', alpha=0.5)
    axs[2].set_title('Diferencia ($V_t$ - spikes)')
    axs[2].set_xlabel('Tiempo (samples)')
    axs[2].set_ylabel('Diferencia')
    axs[2].set_ylim(-axs[0].get_ylim()[1]/8,axs[0].get_ylim()[1]/8)
    #axs[2].grid(True)
    axs[2].legend()

    # Anotar parámetros
    texto = f"α = {alpha:.4e}\nAUC = {auc:.3f}"
    axs[1].text(0.02, 0.98, texto, transform=axs[1].transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

# ===================================================================================
# ANOVA / Correlación
# Ángulo 0
# ===================================================================================

# Cargar datos
df = pd.read_csv('/content/drive/MyDrive/resultados_modelo_0_grados.csv')

# --- ANOVA: AUC según alpha ---
grupos = [grupo['auc'].values for _, grupo in df.groupby('alpha')]
f_stat, p_val = f_oneway(*grupos)

print(f"ANOVA entre valores de α: F = {f_stat:.3f}, p = {p_val}")
if p_val < 0.05:
    print("✅ Hay diferencias significativas en AUC según el valor de α.")
else:
    print("❌ No se detectaron diferencias significativas.")

# --- Correlación AUC vs alpha (no lineal) ---
rho, p_corr = spearmanr(df['alpha'], df['auc'])
print(f"Correlación de Spearman: rho = {rho:.3f}, p = {p_corr}")

# ===================================================================================
# Análisis de estabilidad
# Ángulo 0
# ===================================================================================

# Calcular media y desviación estándar por alpha
stability_df = df.groupby('alpha')['auc'].agg(['mean', 'std']).reset_index()

# Gráfico de errorbar con barras de error más transparentes
plt.figure(figsize=(10, 6))
plt.errorbar(
    stability_df['alpha'],
    stability_df['mean'],
    yerr=stability_df['std'],
    fmt='-o',
    ecolor='gray',
    alpha=0.5
)

#plt.title('Estabilidad del modelo por valor de alpha')
plt.xlabel('Alpha (memoria)')
plt.ylabel('AUC promedio ± desv. estándar')
plt.show()

# ===================================================================================
# Análisis de varianza
# Ángulo 0
# ===================================================================================

angulo, bloque_inicio, bloque_fin = 0, 41128708, 42122545

tiempos_activacion = tiempos_de_activacion_spikes(
    grado=angulo,
    start_time=bloque_inicio,
    end_time=bloque_fin
)

coord_spikes = generar_extended_coord_spikes(
    start_time=bloque_inicio,
    end_time=bloque_fin
)

tiempos_barra = intervalos_barra(angulo)
ini, fin = tiempos_barra[0]
duracion_total = fin-ini
duracion_pixel = duracion_total / 16

intervalos_por_electrodo = generar_diccionario_intervalos_barras(
    generar_extended_coord_spikes=coord_spikes,
    tiempos_barra=tiempos_barra,
    angulo=angulo,
    duracion=duracion_total,
    start_time=bloque_inicio
)

celula_a_coordenada = obtener_celula_a_coordenada(
    cluster_path,
    coord_path)

# Ruta al archivo CSV
dataframe_path = '/content/drive/MyDrive/resultados_modelo_0_grados.csv'

varianza_0 = dic_varianza_desviacion(dataframe_path)

var_0_reales = combinar_tiempos_activacion(tiempos_activacion)
var_0 = extraer_varianzas(varianza_0)

print("✅ OK")

var_0 = extraer_varianzas(varianza_0)
boxplot_var(var_0)

#####################################################################################

# ===================================================================================
# Simulación del modelo no Markoviano / AUC / Precision / Sensibilidad
# Ángulo 45
# ===================================================================================

angulo, bloque_inicio, bloque_fin = 45, 42128580, 43425195

tiempos_activacion = tiempos_de_activacion_spikes(
    grado=angulo,
    start_time=bloque_inicio,
    end_time=bloque_fin
)

coord_spikes = generar_extended_coord_spikes(
    start_time=bloque_inicio,
    end_time=bloque_fin
)

tiempos_barra = intervalos_barra(angulo)
ini, fin = tiempos_barra[0]
duracion_total = fin-ini
duracion_pixel = duracion_total / 16

intervalos_por_electrodo = generar_diccionario_intervalos_barras(
    generar_extended_coord_spikes=coord_spikes,
    tiempos_barra=tiempos_barra,
    angulo=angulo,
    duracion=duracion_total,
    start_time=bloque_inicio
)

celula_a_coordenada = obtener_celula_a_coordenada(
    cluster_path,
    coord_path)

# --- EJECUTAR EL PROCESO ---
resultados_finales = optimizar_alpha_para_todas_las_celulas(
    tiempos_activacion,
    celula_a_coordenada,
    intervalos_por_electrodo
)

# Mostrar resultados
#for celula, datos in resultados_finales.items():
#    print(f"Célula {celula}: α={datos['alpha']:.6f}, AUC={datos['auc']:.4f}")

# Convertir a DataFrame
df_resultados = pd.DataFrame.from_dict(resultados_finales, orient='index')
df_resultados.index.name = 'celula'

# Montar Drive (solo necesitas hacerlo una vez por sesión)
drive.mount('/content/drive')

# Guardar en Drive como CSV
ruta_csv = f'/content/drive/MyDrive/resultados_modelo_{angulo}_grados.csv'
df_resultados.to_csv(ruta_csv)

# Confirmación visual
print(f"\n✅ Resultados guardados en: {ruta_csv}")

# ===================================================================================
# Comparación entre simulación y datos
# Ángulo 45
# ===================================================================================
ruta_csv = '/content/drive/MyDrive/resultados_modelo_45_grados.csv'
df_resultados = pd.read_csv(ruta_csv, index_col='celula')

for celula_str, fila in df_resultados.iterrows():
    celula = int(celula_str)
    alpha = fila['alpha']
    auc = fila['auc']

    if celula not in tiempos_activacion:
        continue

    diferencias = tiempos_activacion[celula]
    tiempos_activacion_real = np.cumsum(diferencias)
    conteo = np.arange(len(tiempos_activacion_real))

    tiempos_step = np.insert(tiempos_activacion_real, 0, 0)
    conteo_step = np.insert(conteo, 0, 0)

    # Obtener estímulo U_t
    coord = celula_a_coordenada.get(celula)
    intervalos = list(intervalos_por_electrodo.get(coord, {}).values())
    T = max(tiempos_step[-1], max((fin for _, fin in intervalos), default=0)) + 1000
    T = int(T)

    U_t = np.zeros(T)
    for inicio, fin in intervalos:
        U_t[inicio:fin] = 1

    # Simular señal original
    V_simulado = simular_modelo_no_markoviano(U_t, alpha)

    # Interpolación del conteo real sobre la escala completa
    conteo_interp_real = np.interp(np.arange(T), tiempos_step, conteo_step)

    # Escalamiento de simulación al rango del conteo real
    p98_conteo = np.percentile(conteo_interp_real, 98)
    p98_V = np.percentile(V_simulado, 98)
    escala = p98_conteo / p98_V if p98_V > 0 else 1.0
    V_simulado_escalado = V_simulado * escala

    # Calcular diferencia con misma escala
    diferencia = V_simulado_escalado - conteo_interp_real

    # Crear gráfico con 3 paneles
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.5), sharex=False)

    # Panel izquierdo: spikes reales (escala real)
    axs[0].step(tiempos_step, conteo_step, where='post', label='Spikes reales', linewidth=1.5)
    for t_inicio, t_fin in intervalos:
        axs[0].axvspan(t_inicio, t_fin, color='cyan', alpha=0.5)
    axs[0].set_title(f'Célula {celula} - Spikes')
    axs[0].set_xlabel('Tiempo (samples)')
    axs[0].set_ylabel('Conteo acumulado')
    axs[0].grid(True)
    axs[0].legend()

    # Panel central: simulación escalada
    axs[1].plot(V_simulado_escalado, label='Señal simulada', color='purple', linewidth=1)
    for t_inicio, t_fin in intervalos:
        axs[1].axvspan(t_inicio, t_fin, color='cyan', alpha=0.5)
    axs[1].set_title('Simulación $V_t$')
    axs[1].set_xlabel('Tiempo (samples)')
    axs[1].set_ylabel('$V_t$')
    axs[1].grid(True)
    axs[1].legend()

    # Panel derecho: diferencia con misma escala
    axs[2].plot(diferencia, label='Diferencia', color='orange', linewidth=1)
    axs[2].axhline(0, color='red', linestyle='--', linewidth=0.8)
    for t_inicio, t_fin in intervalos:
        axs[2].axvspan(t_inicio, t_fin, color='cyan', alpha=0.5)
    axs[2].set_title('Diferencia ($V_t$ - spikes)')
    axs[2].set_xlabel('Tiempo (samples)')
    axs[2].set_ylabel('Diferencia')
    axs[2].set_ylim(-axs[0].get_ylim()[1]/8,axs[0].get_ylim()[1]/8)
    #axs[2].grid(True)
    axs[2].legend()

    # Anotar parámetros
    texto = f"α = {alpha:.4e}\nAUC = {auc:.3f}"
    axs[1].text(0.02, 0.98, texto, transform=axs[1].transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

# ===================================================================================
# ANOVA / Correlación
# Ángulo 45
# ===================================================================================

# Cargar datos
df = pd.read_csv('/content/drive/MyDrive/resultados_modelo_45_grados.csv')

# --- ANOVA: AUC según alpha ---
grupos = [grupo['auc'].values for _, grupo in df.groupby('alpha')]
f_stat, p_val = f_oneway(*grupos)

print(f"ANOVA entre valores de α: F = {f_stat:.3f}, p = {p_val}")
if p_val < 0.05:
    print("✅ Hay diferencias significativas en AUC según el valor de α.")
else:
    print("❌ No se detectaron diferencias significativas.")

# --- Correlación AUC vs alpha (no lineal) ---
rho, p_corr = spearmanr(df['alpha'], df['auc'])
print(f"Correlación de Spearman: rho = {rho:.3f}, p = {p_corr}")

# ===================================================================================
# Análisis de estabilidad
# Ángulo 45
# ===================================================================================

# Calcular media y desviación estándar por alpha
stability_df = df.groupby('alpha')['auc'].agg(['mean', 'std']).reset_index()

# Gráfico de errorbar con barras de error más transparentes
plt.figure(figsize=(10, 6))
plt.errorbar(
    stability_df['alpha'],
    stability_df['mean'],
    yerr=stability_df['std'],
    fmt='-o',
    ecolor='gray',
    alpha=0.5
)

#plt.title('Estabilidad del modelo por valor de alpha')
plt.xlabel('Alpha (memoria)')
plt.ylabel('AUC promedio ± desv. estándar')
plt.show()

# ===================================================================================
# Análisis de varianza
# Ángulo 45
# ===================================================================================

angulo, bloque_inicio, bloque_fin = 45, 42128580, 43425195

tiempos_activacion = tiempos_de_activacion_spikes(
    grado=angulo,
    start_time=bloque_inicio,
    end_time=bloque_fin
)

coord_spikes = generar_extended_coord_spikes(
    start_time=bloque_inicio,
    end_time=bloque_fin
)

tiempos_barra = intervalos_barra(angulo)
ini, fin = tiempos_barra[0]
duracion_total = fin-ini
duracion_pixel = duracion_total / 16

intervalos_por_electrodo = generar_diccionario_intervalos_barras(
    generar_extended_coord_spikes=coord_spikes,
    tiempos_barra=tiempos_barra,
    angulo=angulo,
    duracion=duracion_total,
    start_time=bloque_inicio
)

celula_a_coordenada = obtener_celula_a_coordenada(
    cluster_path,
    coord_path)

# Ruta al archivo CSV
dataframe_path = '/content/drive/MyDrive/resultados_modelo_45_grados.csv'

varianza_45 = dic_varianza_desviacion(dataframe_path)
var_45_reales = combinar_tiempos_activacion(tiempos_activacion)
var_45 = extraer_varianzas(varianza_45)

print("✅ OK")

var_45 = extraer_varianzas(varianza_45)
boxplot_var(var_45)

#####################################################################################

# ===================================================================================
# Simulación del modelo no Markoviano / AUC / Precision / Sensibilidad
# Ángulo 90
# ===================================================================================

angulo, bloque_inicio, bloque_fin = 90, 43431231, 44425067

tiempos_activacion = tiempos_de_activacion_spikes(
    grado=angulo,
    start_time=bloque_inicio,
    end_time=bloque_fin
)

coord_spikes = generar_extended_coord_spikes(
    start_time=bloque_inicio,
    end_time=bloque_fin
)

tiempos_barra = intervalos_barra(angulo)
ini, fin = tiempos_barra[0]
duracion_total = fin-ini
duracion_pixel = duracion_total / 16

intervalos_por_electrodo = generar_diccionario_intervalos_barras(
    generar_extended_coord_spikes=coord_spikes,
    tiempos_barra=tiempos_barra,
    angulo=angulo,
    duracion=duracion_total,
    start_time=bloque_inicio
)

celula_a_coordenada = obtener_celula_a_coordenada(
    cluster_path,
    coord_path)

# --- EJECUTAR EL PROCESO ---
resultados_finales = optimizar_alpha_para_todas_las_celulas(
    tiempos_activacion,
    celula_a_coordenada,
    intervalos_por_electrodo
)

# Mostrar resultados
#for celula, datos in resultados_finales.items():
#    print(f"Célula {celula}: α={datos['alpha']:.6f}, AUC={datos['auc']:.4f}")

# Convertir a DataFrame
df_resultados = pd.DataFrame.from_dict(resultados_finales, orient='index')
df_resultados.index.name = 'celula'

# Montar Drive (solo necesitas hacerlo una vez por sesión)
drive.mount('/content/drive')

# Guardar en Drive como CSV
ruta_csv = f'/content/drive/MyDrive/resultados_modelo_{angulo}_grados.csv'
df_resultados.to_csv(ruta_csv)

# Confirmación visual
print(f"\n✅ Resultados guardados en: {ruta_csv}")

# ===================================================================================
# Comparación entre simulación y datos
# Ángulo 90
# ===================================================================================
ruta_csv = '/content/drive/MyDrive/resultados_modelo_90_grados.csv'
df_resultados = pd.read_csv(ruta_csv, index_col='celula')

for celula_str, fila in df_resultados.iterrows():
    celula = int(celula_str)
    alpha = fila['alpha']
    auc = fila['auc']

    if celula not in tiempos_activacion:
        continue

    diferencias = tiempos_activacion[celula]
    tiempos_activacion_real = np.cumsum(diferencias)
    conteo = np.arange(len(tiempos_activacion_real))

    tiempos_step = np.insert(tiempos_activacion_real, 0, 0)
    conteo_step = np.insert(conteo, 0, 0)

    # Obtener estímulo U_t
    coord = celula_a_coordenada.get(celula)
    intervalos = list(intervalos_por_electrodo.get(coord, {}).values())
    T = max(tiempos_step[-1], max((fin for _, fin in intervalos), default=0)) + 1000
    T = int(T)

    U_t = np.zeros(T)
    for inicio, fin in intervalos:
        U_t[inicio:fin] = 1

    # Simular señal original
    V_simulado = simular_modelo_no_markoviano(U_t, alpha)

    # Interpolación del conteo real sobre la escala completa
    conteo_interp_real = np.interp(np.arange(T), tiempos_step, conteo_step)

    # Escalamiento de simulación al rango del conteo real
    p98_conteo = np.percentile(conteo_interp_real, 98)
    p98_V = np.percentile(V_simulado, 98)
    escala = p98_conteo / p98_V if p98_V > 0 else 1.0
    V_simulado_escalado = V_simulado * escala

    # Calcular diferencia con misma escala
    diferencia = V_simulado_escalado - conteo_interp_real

    # Crear gráfico con 3 paneles
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.5), sharex=False)

    # Panel izquierdo: spikes reales (escala real)
    axs[0].step(tiempos_step, conteo_step, where='post', label='Spikes reales', linewidth=1.5)
    for t_inicio, t_fin in intervalos:
        axs[0].axvspan(t_inicio, t_fin, color='cyan', alpha=0.5)
    axs[0].set_title(f'Célula {celula} - Spikes')
    axs[0].set_xlabel('Tiempo (samples)')
    axs[0].set_ylabel('Conteo acumulado')
    axs[0].grid(True)
    axs[0].legend()

    # Panel central: simulación escalada
    axs[1].plot(V_simulado_escalado, label='Señal simulada', color='purple', linewidth=1)
    for t_inicio, t_fin in intervalos:
        axs[1].axvspan(t_inicio, t_fin, color='cyan', alpha=0.5)
    axs[1].set_title('Simulación $V_t$')
    axs[1].set_xlabel('Tiempo (samples)')
    axs[1].set_ylabel('$V_t$')
    axs[1].grid(True)
    axs[1].legend()

    # Panel derecho: diferencia con misma escala
    axs[2].plot(diferencia, label='Diferencia', color='orange', linewidth=1)
    axs[2].axhline(0, color='red', linestyle='--', linewidth=0.8)
    for t_inicio, t_fin in intervalos:
        axs[2].axvspan(t_inicio, t_fin, color='cyan', alpha=0.5)
    axs[2].set_title('Diferencia ($V_t$ - spikes)')
    axs[2].set_xlabel('Tiempo (samples)')
    axs[2].set_ylabel('Diferencia')
    axs[2].set_ylim(-axs[0].get_ylim()[1]/8,axs[0].get_ylim()[1]/8)
    #axs[2].grid(True)
    axs[2].legend()

    # Anotar parámetros
    texto = f"α = {alpha:.4e}\nAUC = {auc:.3f}"
    axs[1].text(0.02, 0.98, texto, transform=axs[1].transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

# ===================================================================================
# ANOVA / Correlación
# Ángulo 90
# ===================================================================================

# Cargar datos
df = pd.read_csv('/content/drive/MyDrive/resultados_modelo_90_grados.csv')

# --- ANOVA: AUC según alpha ---
grupos = [grupo['auc'].values for _, grupo in df.groupby('alpha')]
f_stat, p_val = f_oneway(*grupos)

print(f"ANOVA entre valores de α: F = {f_stat:.3f}, p = {p_val}")
if p_val < 0.05:
    print("✅ Hay diferencias significativas en AUC según el valor de α.")
else:
    print("❌ No se detectaron diferencias significativas.")

# --- Correlación AUC vs alpha (no lineal) ---
rho, p_corr = spearmanr(df['alpha'], df['auc'])
print(f"Correlación de Spearman: rho = {rho:.3f}, p = {p_corr}")

# ===================================================================================
# Análisis de estabilidad
# Ángulo 90
# ===================================================================================

# Calcular media y desviación estándar por alpha
stability_df = df.groupby('alpha')['auc'].agg(['mean', 'std']).reset_index()

# Gráfico de errorbar con barras de error más transparentes
plt.figure(figsize=(10, 6))
plt.errorbar(
    stability_df['alpha'],
    stability_df['mean'],
    yerr=stability_df['std'],
    fmt='-o',
    ecolor='gray',
    alpha=0.5
)

#plt.title('Estabilidad del modelo por valor de alpha')
plt.xlabel('Alpha (memoria)')
plt.ylabel('AUC promedio ± desv. estándar')
plt.show()

# ===================================================================================
# Análisis de varianza
# Ángulo 90
# ===================================================================================

angulo, bloque_inicio, bloque_fin = 90, 43431231, 44425067

tiempos_activacion = tiempos_de_activacion_spikes(
    grado=angulo,
    start_time=bloque_inicio,
    end_time=bloque_fin
)

coord_spikes = generar_extended_coord_spikes(
    start_time=bloque_inicio,
    end_time=bloque_fin
)

tiempos_barra = intervalos_barra(angulo)
ini, fin = tiempos_barra[0]
duracion_total = fin-ini
duracion_pixel = duracion_total / 16

intervalos_por_electrodo = generar_diccionario_intervalos_barras(
    generar_extended_coord_spikes=coord_spikes,
    tiempos_barra=tiempos_barra,
    angulo=angulo,
    duracion=duracion_total,
    start_time=bloque_inicio
)

celula_a_coordenada = obtener_celula_a_coordenada(
    cluster_path,
    coord_path)

# Ruta al archivo CSV
dataframe_path = '/content/drive/MyDrive/resultados_modelo_90_grados.csv'

varianza_90 = dic_varianza_desviacion(dataframe_path)
var_90_reales = combinar_tiempos_activacion(tiempos_activacion)
var_90 = extraer_varianzas(varianza_90)

print("✅ OK")

var_90 = extraer_varianzas(varianza_90)
boxplot_var(var_90)

#####################################################################################

# ===================================================================================
# Simulación del modelo no Markoviano / AUC / Precision / Sensibilidad
# Ángulo 135
# ===================================================================================

angulo, bloque_inicio, bloque_fin = 135, 44431103, 45728052

tiempos_activacion = tiempos_de_activacion_spikes(
    grado=angulo,
    start_time=bloque_inicio,
    end_time=bloque_fin
)

coord_spikes = generar_extended_coord_spikes(
    start_time=bloque_inicio,
    end_time=bloque_fin
)

tiempos_barra = intervalos_barra(angulo)
ini, fin = tiempos_barra[0]
duracion_total = fin-ini
duracion_pixel = duracion_total / 16

intervalos_por_electrodo = generar_diccionario_intervalos_barras(
    generar_extended_coord_spikes=coord_spikes,
    tiempos_barra=tiempos_barra,
    angulo=angulo,
    duracion=duracion_total,
    start_time=bloque_inicio
)

celula_a_coordenada = obtener_celula_a_coordenada(
    cluster_path,
    coord_path)

# --- EJECUTAR EL PROCESO ---
resultados_finales = optimizar_alpha_para_todas_las_celulas(
    tiempos_activacion,
    celula_a_coordenada,
    intervalos_por_electrodo
)

# Convertir a DataFrame
df_resultados = pd.DataFrame.from_dict(resultados_finales, orient='index')
df_resultados.index.name = 'celula'

# Montar Drive (solo necesitas hacerlo una vez por sesión)
drive.mount('/content/drive')

# Guardar en Drive como CSV
ruta_csv = f'/content/drive/MyDrive/resultados_modelo_{angulo}_grados.csv'
df_resultados.to_csv(ruta_csv)

# Confirmación visual
print(f"\n✅ Resultados guardados en: {ruta_csv}")

# ===================================================================================
# Comparación entre simulación y datos
# Ángulo 135
# ===================================================================================

ruta_csv = '/content/drive/MyDrive/resultados_modelo_135_grados.csv'
df_resultados = pd.read_csv(ruta_csv, index_col='celula')

for celula_str, fila in df_resultados.iterrows():
    celula = int(celula_str)
    alpha = fila['alpha']
    auc = fila['auc']

    if celula not in tiempos_activacion:
        continue

    diferencias = tiempos_activacion[celula]
    tiempos_activacion_real = np.cumsum(diferencias)
    conteo = np.arange(len(tiempos_activacion_real))

    tiempos_step = np.insert(tiempos_activacion_real, 0, 0)
    conteo_step = np.insert(conteo, 0, 0)

    # Obtener estímulo U_t
    coord = celula_a_coordenada.get(celula)
    intervalos = list(intervalos_por_electrodo.get(coord, {}).values())
    T = max(tiempos_step[-1], max((fin for _, fin in intervalos), default=0)) + 1000
    T = int(T)

    U_t = np.zeros(T)
    for inicio, fin in intervalos:
        U_t[inicio:fin] = 1

    # Simular señal original
    V_simulado = simular_modelo_no_markoviano(U_t, alpha)

    # Interpolación del conteo real sobre la escala completa
    conteo_interp_real = np.interp(np.arange(T), tiempos_step, conteo_step)

    # Escalamiento de simulación al rango del conteo real
    p98_conteo = np.percentile(conteo_interp_real, 98)
    p98_V = np.percentile(V_simulado, 98)
    escala = p98_conteo / p98_V if p98_V > 0 else 1.0
    V_simulado_escalado = V_simulado * escala

    # Calcular diferencia con misma escala
    diferencia = V_simulado_escalado - conteo_interp_real

    # Crear gráfico con 3 paneles
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.5), sharex=False)

    # Panel izquierdo: spikes reales (escala real)
    axs[0].step(tiempos_step, conteo_step, where='post', label='Spikes reales', linewidth=1.5)
    for t_inicio, t_fin in intervalos:
        axs[0].axvspan(t_inicio, t_fin, color='cyan', alpha=0.5)
    axs[0].set_title(f'Célula {celula} - Spikes')
    axs[0].set_xlabel('Tiempo (samples)')
    axs[0].set_ylabel('Conteo acumulado')
    axs[0].grid(True)
    axs[0].legend()

    # Panel central: simulación escalada
    axs[1].plot(V_simulado_escalado, label='Señal simulada', color='purple', linewidth=1)
    for t_inicio, t_fin in intervalos:
        axs[1].axvspan(t_inicio, t_fin, color='cyan', alpha=0.5)
    axs[1].set_title('Simulación $V_t$')
    axs[1].set_xlabel('Tiempo (samples)')
    axs[1].set_ylabel('$V_t$')
    axs[1].grid(True)
    axs[1].legend()

    # Panel derecho: diferencia con misma escala
    axs[2].plot(diferencia, label='Diferencia', color='orange', linewidth=1)
    axs[2].axhline(0, color='red', linestyle='--', linewidth=0.8)
    for t_inicio, t_fin in intervalos:
        axs[2].axvspan(t_inicio, t_fin, color='cyan', alpha=0.5)
    axs[2].set_title('Diferencia ($V_t$ - spikes)')
    axs[2].set_xlabel('Tiempo (samples)')
    axs[2].set_ylabel('Diferencia')
    axs[2].set_ylim(-axs[0].get_ylim()[1]/8,axs[0].get_ylim()[1]/8)
    #axs[2].grid(True)
    axs[2].legend()

    # Anotar parámetros
    texto = f"α = {alpha:.4e}\nAUC = {auc:.3f}"
    axs[1].text(0.02, 0.98, texto, transform=axs[1].transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

# ===================================================================================
# ANOVA / Correlación
# Ángulo 135
# ===================================================================================

# Cargar datos
df = pd.read_csv('/content/drive/MyDrive/resultados_modelo_135_grados.csv')

# --- ANOVA: AUC según alpha ---
grupos = [grupo['auc'].values for _, grupo in df.groupby('alpha')]
f_stat, p_val = f_oneway(*grupos)

print(f"ANOVA entre valores de α: F = {f_stat:.3f}, p = {p_val}")
if p_val < 0.05:
    print("✅ Hay diferencias significativas en AUC según el valor de α.")
else:
    print("❌ No se detectaron diferencias significativas.")

# --- Correlación AUC vs alpha (no lineal) ---
rho, p_corr = spearmanr(df['alpha'], df['auc'])
print(f"Correlación de Spearman: rho = {rho:.3f}, p = {p_corr}")

# ===================================================================================
# Análisis de estabilidad
# Ángulo 135
# ===================================================================================

# Calcular media y desviación estándar por alpha
stability_df = df.groupby('alpha')['auc'].agg(['mean', 'std']).reset_index()

# Gráfico de errorbar con barras de error más transparentes
plt.figure(figsize=(10, 6))
plt.errorbar(
    stability_df['alpha'],
    stability_df['mean'],
    yerr=stability_df['std'],
    fmt='-o',
    ecolor='gray',
    alpha=0.5
)

#plt.title('Estabilidad del modelo por valor de alpha')
plt.xlabel('Alpha (memoria)')
plt.ylabel('AUC promedio ± desv. estándar')
plt.show()

# ===================================================================================
# Análisis de varianza
# Ángulo 135
# ===================================================================================

angulo, bloque_inicio, bloque_fin = 135, 44431103, 45728052

tiempos_activacion = tiempos_de_activacion_spikes(
    grado=angulo,
    start_time=bloque_inicio,
    end_time=bloque_fin
)

coord_spikes = generar_extended_coord_spikes(
    start_time=bloque_inicio,
    end_time=bloque_fin
)

tiempos_barra = intervalos_barra(angulo)
ini, fin = tiempos_barra[0]
duracion_total = fin-ini
duracion_pixel = duracion_total / 16

intervalos_por_electrodo = generar_diccionario_intervalos_barras(
    generar_extended_coord_spikes=coord_spikes,
    tiempos_barra=tiempos_barra,
    angulo=angulo,
    duracion=duracion_total,
    start_time=bloque_inicio
)

celula_a_coordenada = obtener_celula_a_coordenada(
    cluster_path,
    coord_path)

# Ruta al archivo CSV
dataframe_path = '/content/drive/MyDrive/resultados_modelo_135_grados.csv'

varianza_135 = dic_varianza_desviacion(dataframe_path)
var_135_reales = combinar_tiempos_activacion(tiempos_activacion)
var_135 = extraer_varianzas(varianza_135)

print("✅ OK")

var_135 = extraer_varianzas(varianza_135)
boxplot_var(var_135)

#####################################################################################

# ===================================================================================
# Simulación del modelo no Markoviano / AUC / Precision / Sensibilidad
# Ángulo 180
# ===================================================================================

angulo, bloque_inicio, bloque_fin = 180, 45734088, 46727924

tiempos_activacion = tiempos_de_activacion_spikes(
    grado=angulo,
    start_time=bloque_inicio,
    end_time=bloque_fin
)

coord_spikes = generar_extended_coord_spikes(
    start_time=bloque_inicio,
    end_time=bloque_fin
)

tiempos_barra = intervalos_barra(angulo)
ini, fin = tiempos_barra[0]
duracion_total = fin-ini
duracion_pixel = duracion_total / 16

intervalos_por_electrodo = generar_diccionario_intervalos_barras(
    generar_extended_coord_spikes=coord_spikes,
    tiempos_barra=tiempos_barra,
    angulo=angulo,
    duracion=duracion_total,
    start_time=bloque_inicio
)

celula_a_coordenada = obtener_celula_a_coordenada(
    cluster_path,
    coord_path)

# --- EJECUTAR EL PROCESO ---
resultados_finales = optimizar_alpha_para_todas_las_celulas(
    tiempos_activacion,
    celula_a_coordenada,
    intervalos_por_electrodo
)

# Convertir a DataFrame
df_resultados = pd.DataFrame.from_dict(resultados_finales, orient='index')
df_resultados.index.name = 'celula'

# Montar Drive (solo necesitas hacerlo una vez por sesión)
drive.mount('/content/drive')

# Guardar en Drive como CSV
ruta_csv = f'/content/drive/MyDrive/resultados_modelo_{angulo}_grados.csv'
df_resultados.to_csv(ruta_csv)

# Confirmación visual
print(f"\n✅ Resultados guardados en: {ruta_csv}")

# ===================================================================================
# Comparación entre simulación y datos
# Ángulo 180
# ===================================================================================

ruta_csv = '/content/drive/MyDrive/resultados_modelo_180_grados.csv'
df_resultados = pd.read_csv(ruta_csv, index_col='celula')

for celula_str, fila in df_resultados.iterrows():
    celula = int(celula_str)
    alpha = fila['alpha']
    auc = fila['auc']

    if celula not in tiempos_activacion:
        continue

    diferencias = tiempos_activacion[celula]
    tiempos_activacion_real = np.cumsum(diferencias)
    conteo = np.arange(len(tiempos_activacion_real))

    tiempos_step = np.insert(tiempos_activacion_real, 0, 0)
    conteo_step = np.insert(conteo, 0, 0)

    # Obtener estímulo U_t
    coord = celula_a_coordenada.get(celula)
    intervalos = list(intervalos_por_electrodo.get(coord, {}).values())
    T = max(tiempos_step[-1], max((fin for _, fin in intervalos), default=0)) + 1000
    T = int(T)

    U_t = np.zeros(T)
    for inicio, fin in intervalos:
        U_t[inicio:fin] = 1

    # Simular señal original
    V_simulado = simular_modelo_no_markoviano(U_t, alpha)

    # Interpolación del conteo real sobre la escala completa
    conteo_interp_real = np.interp(np.arange(T), tiempos_step, conteo_step)

    # Escalamiento de simulación al rango del conteo real
    p98_conteo = np.percentile(conteo_interp_real, 98)
    p98_V = np.percentile(V_simulado, 98)
    escala = p98_conteo / p98_V if p98_V > 0 else 1.0
    V_simulado_escalado = V_simulado * escala

    # Calcular diferencia con misma escala
    diferencia = V_simulado_escalado - conteo_interp_real

    # Crear gráfico con 3 paneles
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.5), sharex=False)

    # Panel izquierdo: spikes reales (escala real)
    axs[0].step(tiempos_step, conteo_step, where='post', label='Spikes reales', linewidth=1.5)
    for t_inicio, t_fin in intervalos:
        axs[0].axvspan(t_inicio, t_fin, color='cyan', alpha=0.5)
    axs[0].set_title(f'Célula {celula} - Spikes')
    axs[0].set_xlabel('Tiempo (samples)')
    axs[0].set_ylabel('Conteo acumulado')
    axs[0].grid(True)
    axs[0].legend()

    # Panel central: simulación escalada
    axs[1].plot(V_simulado_escalado, label='Señal simulada', color='purple', linewidth=1)
    for t_inicio, t_fin in intervalos:
        axs[1].axvspan(t_inicio, t_fin, color='cyan', alpha=0.5)
    axs[1].set_title('Simulación $V_t$')
    axs[1].set_xlabel('Tiempo (samples)')
    axs[1].set_ylabel('$V_t$')
    axs[1].grid(True)
    axs[1].legend()

    # Panel derecho: diferencia con misma escala
    axs[2].plot(diferencia, label='Diferencia', color='orange', linewidth=1)
    axs[2].axhline(0, color='red', linestyle='--', linewidth=0.8)
    for t_inicio, t_fin in intervalos:
        axs[2].axvspan(t_inicio, t_fin, color='cyan', alpha=0.5)
    axs[2].set_title('Diferencia ($V_t$ - spikes)')
    axs[2].set_xlabel('Tiempo (samples)')
    axs[2].set_ylabel('Diferencia')
    axs[2].set_ylim(-axs[0].get_ylim()[1]/8,axs[0].get_ylim()[1]/8)
    #axs[2].grid(True)
    axs[2].legend()

    # Anotar parámetros
    texto = f"α = {alpha:.4e}\nAUC = {auc:.3f}"
    axs[1].text(0.02, 0.98, texto, transform=axs[1].transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

# ===================================================================================
# ANOVA / Correlación
# Ángulo 180
# ===================================================================================

# Cargar datos
df = pd.read_csv('/content/drive/MyDrive/resultados_modelo_180_grados.csv')

# --- ANOVA: AUC según alpha ---
grupos = [grupo['auc'].values for _, grupo in df.groupby('alpha')]
f_stat, p_val = f_oneway(*grupos)

print(f"ANOVA entre valores de α: F = {f_stat:.3f}, p = {p_val}")
if p_val < 0.05:
    print("✅ Hay diferencias significativas en AUC según el valor de α.")
else:
    print("❌ No se detectaron diferencias significativas.")

# --- Correlación AUC vs alpha (no lineal) ---
rho, p_corr = spearmanr(df['alpha'], df['auc'])
print(f"Correlación de Spearman: rho = {rho:.3f}, p = {p_corr}")

# ===================================================================================
# Análisis de estabilidad
# Ángulo 180
# ===================================================================================

# Calcular media y desviación estándar por alpha
stability_df = df.groupby('alpha')['auc'].agg(['mean', 'std']).reset_index()

# Gráfico de errorbar con barras de error más transparentes
plt.figure(figsize=(10, 6))
plt.errorbar(
    stability_df['alpha'],
    stability_df['mean'],
    yerr=stability_df['std'],
    fmt='-o',
    ecolor='gray',
    alpha=0.5
)

#plt.title('Estabilidad del modelo por valor de alpha')
plt.xlabel('Alpha (memoria)')
plt.ylabel('AUC promedio ± desv. estándar')
plt.show()

# ===================================================================================
# Análisis de varianza
# Ángulo 180
# ===================================================================================

angulo, bloque_inicio, bloque_fin = 180, 45734088, 46727924

tiempos_activacion = tiempos_de_activacion_spikes(
    grado=angulo,
    start_time=bloque_inicio,
    end_time=bloque_fin
)

coord_spikes = generar_extended_coord_spikes(
    start_time=bloque_inicio,
    end_time=bloque_fin
)

tiempos_barra = intervalos_barra(angulo)
ini, fin = tiempos_barra[0]
duracion_total = fin-ini
duracion_pixel = duracion_total / 16

intervalos_por_electrodo = generar_diccionario_intervalos_barras(
    generar_extended_coord_spikes=coord_spikes,
    tiempos_barra=tiempos_barra,
    angulo=angulo,
    duracion=duracion_total,
    start_time=bloque_inicio
)

celula_a_coordenada = obtener_celula_a_coordenada(
    cluster_path,
    coord_path)

# Ruta al archivo CSV
dataframe_path = '/content/drive/MyDrive/resultados_modelo_180_grados.csv'

varianza_180 = dic_varianza_desviacion(dataframe_path)

print("✅ OK")

var_180 = extraer_varianzas(varianza_180)
boxplot_var(var_180)

#####################################################################################

# ===================================================================================
# Simulación del modelo no Markoviano / AUC / Precision / Sensibilidad
# Ángulo 225
# ===================================================================================

angulo, bloque_inicio, bloque_fin = 225, 46733960, 48030574

tiempos_activacion = tiempos_de_activacion_spikes(
    grado=angulo,
    start_time=bloque_inicio,
    end_time=bloque_fin
)

coord_spikes = generar_extended_coord_spikes(
    start_time=bloque_inicio,
    end_time=bloque_fin
)

tiempos_barra = intervalos_barra(angulo)
ini, fin = tiempos_barra[0]
duracion_total = fin-ini
duracion_pixel = duracion_total / 16

intervalos_por_electrodo = generar_diccionario_intervalos_barras(
    generar_extended_coord_spikes=coord_spikes,
    tiempos_barra=tiempos_barra,
    angulo=angulo,
    duracion=duracion_total,
    start_time=bloque_inicio
)

celula_a_coordenada = obtener_celula_a_coordenada(
    cluster_path,
    coord_path)

# --- EJECUTAR EL PROCESO ---
resultados_finales = optimizar_alpha_para_todas_las_celulas(
    tiempos_activacion,
    celula_a_coordenada,
    intervalos_por_electrodo
)

# Convertir a DataFrame
df_resultados = pd.DataFrame.from_dict(resultados_finales, orient='index')
df_resultados.index.name = 'celula'

# Montar Drive (solo necesitas hacerlo una vez por sesión)
drive.mount('/content/drive')

# Guardar en Drive como CSV
ruta_csv = f'/content/drive/MyDrive/resultados_modelo_{angulo}_grados.csv'
df_resultados.to_csv(ruta_csv)

# Confirmación visual
print(f"\n✅ Resultados guardados en: {ruta_csv}")

# ===================================================================================
# Comparación entre simulación y datos
# Ángulo 225
# ===================================================================================

ruta_csv = '/content/drive/MyDrive/resultados_modelo_225_grados.csv'
df_resultados = pd.read_csv(ruta_csv, index_col='celula')

for celula_str, fila in df_resultados.iterrows():
    celula = int(celula_str)
    alpha = fila['alpha']
    auc = fila['auc']

    if celula not in tiempos_activacion:
        continue

    diferencias = tiempos_activacion[celula]
    tiempos_activacion_real = np.cumsum(diferencias)
    conteo = np.arange(len(tiempos_activacion_real))

    tiempos_step = np.insert(tiempos_activacion_real, 0, 0)
    conteo_step = np.insert(conteo, 0, 0)

    # Obtener estímulo U_t
    coord = celula_a_coordenada.get(celula)
    intervalos = list(intervalos_por_electrodo.get(coord, {}).values())
    T = max(tiempos_step[-1], max((fin for _, fin in intervalos), default=0)) + 1000
    T = int(T)

    U_t = np.zeros(T)
    for inicio, fin in intervalos:
        U_t[inicio:fin] = 1

    # Simular señal original
    V_simulado = simular_modelo_no_markoviano(U_t, alpha)

    # Interpolación del conteo real sobre la escala completa
    conteo_interp_real = np.interp(np.arange(T), tiempos_step, conteo_step)

    # Escalamiento de simulación al rango del conteo real
    p98_conteo = np.percentile(conteo_interp_real, 98)
    p98_V = np.percentile(V_simulado, 98)
    escala = p98_conteo / p98_V if p98_V > 0 else 1.0
    V_simulado_escalado = V_simulado * escala

    # Calcular diferencia con misma escala
    diferencia = V_simulado_escalado - conteo_interp_real

    # Crear gráfico con 3 paneles
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.5), sharex=False)

    # Panel izquierdo: spikes reales (escala real)
    axs[0].step(tiempos_step, conteo_step, where='post', label='Spikes reales', linewidth=1.5)
    for t_inicio, t_fin in intervalos:
        axs[0].axvspan(t_inicio, t_fin, color='cyan', alpha=0.5)
    axs[0].set_title(f'Célula {celula} - Spikes')
    axs[0].set_xlabel('Tiempo (samples)')
    axs[0].set_ylabel('Conteo acumulado')
    axs[0].grid(True)
    axs[0].legend()

    # Panel central: simulación escalada
    axs[1].plot(V_simulado_escalado, label='Señal simulada', color='purple', linewidth=1)
    for t_inicio, t_fin in intervalos:
        axs[1].axvspan(t_inicio, t_fin, color='cyan', alpha=0.5)
    axs[1].set_title('Simulación $V_t$')
    axs[1].set_xlabel('Tiempo (samples)')
    axs[1].set_ylabel('$V_t$')
    axs[1].grid(True)
    axs[1].legend()

    # Panel derecho: diferencia con misma escala
    axs[2].plot(diferencia, label='Diferencia', color='orange', linewidth=1)
    axs[2].axhline(0, color='red', linestyle='--', linewidth=0.8)
    for t_inicio, t_fin in intervalos:
        axs[2].axvspan(t_inicio, t_fin, color='cyan', alpha=0.5)
    axs[2].set_title('Diferencia ($V_t$ - spikes)')
    axs[2].set_xlabel('Tiempo (samples)')
    axs[2].set_ylabel('Diferencia')
    axs[2].set_ylim(-axs[0].get_ylim()[1]/8,axs[0].get_ylim()[1]/8)
    #axs[2].grid(True)
    axs[2].legend()

    # Anotar parámetros
    texto = f"α = {alpha:.4e}\nAUC = {auc:.3f}"
    axs[1].text(0.02, 0.98, texto, transform=axs[1].transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

# ===================================================================================
# ANOVA / Correlación
# Ángulo 225
# ===================================================================================

# Cargar datos
df = pd.read_csv('/content/drive/MyDrive/resultados_modelo_225_grados.csv')

# --- ANOVA: AUC según alpha ---
grupos = [grupo['auc'].values for _, grupo in df.groupby('alpha')]
f_stat, p_val = f_oneway(*grupos)

print(f"ANOVA entre valores de α: F = {f_stat:.3f}, p = {p_val}")
if p_val < 0.05:
    print("✅ Hay diferencias significativas en AUC según el valor de α.")
else:
    print("❌ No se detectaron diferencias significativas.")

# --- Correlación AUC vs alpha (no lineal) ---
rho, p_corr = spearmanr(df['alpha'], df['auc'])
print(f"Correlación de Spearman: rho = {rho:.3f}, p = {p_corr}")

# ===================================================================================
# Análisis de estabilidad
# Ángulo 225
# ===================================================================================

# Calcular media y desviación estándar por alpha
stability_df = df.groupby('alpha')['auc'].agg(['mean', 'std']).reset_index()

# Gráfico de errorbar con barras de error más transparentes
plt.figure(figsize=(10, 6))
plt.errorbar(
    stability_df['alpha'],
    stability_df['mean'],
    yerr=stability_df['std'],
    fmt='-o',
    ecolor='gray',
    alpha=0.5
)

#plt.title('Estabilidad del modelo por valor de alpha')
plt.xlabel('Alpha (memoria)')
plt.ylabel('AUC promedio ± desv. estándar')
plt.show()

# ===================================================================================
# Análisis de varianza
# Ángulo 225
# ===================================================================================

angulo, bloque_inicio, bloque_fin = 225, 46733960, 48030574

tiempos_activacion = tiempos_de_activacion_spikes(
    grado=angulo,
    start_time=bloque_inicio,
    end_time=bloque_fin
)

coord_spikes = generar_extended_coord_spikes(
    start_time=bloque_inicio,
    end_time=bloque_fin
)

tiempos_barra = intervalos_barra(angulo)
ini, fin = tiempos_barra[0]
duracion_total = fin-ini
duracion_pixel = duracion_total / 16

intervalos_por_electrodo = generar_diccionario_intervalos_barras(
    generar_extended_coord_spikes=coord_spikes,
    tiempos_barra=tiempos_barra,
    angulo=angulo,
    duracion=duracion_total,
    start_time=bloque_inicio
)

celula_a_coordenada = obtener_celula_a_coordenada(
    cluster_path,
    coord_path)

# Ruta al archivo CSV
dataframe_path = '/content/drive/MyDrive/resultados_modelo_225_grados.csv'

varianza_225 = dic_varianza_desviacion(dataframe_path)

print("✅ OK")

var_225 = extraer_varianzas(varianza_225)
boxplot_var(var_225)

#####################################################################################

# ===================================================================================
# Simulación del modelo no Markoviano / AUC / Precision / Sensibilidad 
# Ángulo 270
# ===================================================================================

angulo, bloque_inicio, bloque_fin = 270, 48036610, 49030446

tiempos_activacion = tiempos_de_activacion_spikes(
    grado=angulo,
    start_time=bloque_inicio,
    end_time=bloque_fin
)

coord_spikes = generar_extended_coord_spikes(
    start_time=bloque_inicio,
    end_time=bloque_fin
)

tiempos_barra = intervalos_barra(angulo)
ini, fin = tiempos_barra[0]
duracion_total = fin-ini
duracion_pixel = duracion_total / 16

intervalos_por_electrodo = generar_diccionario_intervalos_barras(
    generar_extended_coord_spikes=coord_spikes,
    tiempos_barra=tiempos_barra,
    angulo=angulo,
    duracion=duracion_total,
    start_time=bloque_inicio
)

celula_a_coordenada = obtener_celula_a_coordenada(
    cluster_path,
    coord_path)

# --- EJECUTAR EL PROCESO ---
resultados_finales = optimizar_alpha_para_todas_las_celulas(
    tiempos_activacion,
    celula_a_coordenada,
    intervalos_por_electrodo
)

# Convertir a DataFrame
df_resultados = pd.DataFrame.from_dict(resultados_finales, orient='index')
df_resultados.index.name = 'celula'

# Montar Drive (solo necesitas hacerlo una vez por sesión)
drive.mount('/content/drive')

# Guardar en Drive como CSV
ruta_csv = f'/content/drive/MyDrive/resultados_modelo_{angulo}_grados.csv'
df_resultados.to_csv(ruta_csv)

# Confirmación visual
print(f"\n✅ Resultados guardados en: {ruta_csv}")

# ===================================================================================
# Comparación entre simulación y datos
# Ángulo 270
# ===================================================================================

ruta_csv = '/content/drive/MyDrive/resultados_modelo_270_grados.csv'
df_resultados = pd.read_csv(ruta_csv, index_col='celula')

for celula_str, fila in df_resultados.iterrows():
    celula = int(celula_str)
    alpha = fila['alpha']
    auc = fila['auc']

    if celula not in tiempos_activacion:
        continue

    diferencias = tiempos_activacion[celula]
    tiempos_activacion_real = np.cumsum(diferencias)
    conteo = np.arange(len(tiempos_activacion_real))

    tiempos_step = np.insert(tiempos_activacion_real, 0, 0)
    conteo_step = np.insert(conteo, 0, 0)

    # Obtener estímulo U_t
    coord = celula_a_coordenada.get(celula)
    intervalos = list(intervalos_por_electrodo.get(coord, {}).values())
    T = max(tiempos_step[-1], max((fin for _, fin in intervalos), default=0)) + 1000
    T = int(T)

    U_t = np.zeros(T)
    for inicio, fin in intervalos:
        U_t[inicio:fin] = 1

    # Simular señal original
    V_simulado = simular_modelo_no_markoviano(U_t, alpha)

    # Interpolación del conteo real sobre la escala completa
    conteo_interp_real = np.interp(np.arange(T), tiempos_step, conteo_step)

    # Escalamiento de simulación al rango del conteo real
    p98_conteo = np.percentile(conteo_interp_real, 98)
    p98_V = np.percentile(V_simulado, 98)
    escala = p98_conteo / p98_V if p98_V > 0 else 1.0
    V_simulado_escalado = V_simulado * escala

    # Calcular diferencia con misma escala
    diferencia = V_simulado_escalado - conteo_interp_real

    # Crear gráfico con 3 paneles
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.5), sharex=False)

    # Panel izquierdo: spikes reales (escala real)
    axs[0].step(tiempos_step, conteo_step, where='post', label='Spikes reales', linewidth=1.5)
    for t_inicio, t_fin in intervalos:
        axs[0].axvspan(t_inicio, t_fin, color='cyan', alpha=0.5)
    axs[0].set_title(f'Célula {celula} - Spikes')
    axs[0].set_xlabel('Tiempo (samples)')
    axs[0].set_ylabel('Conteo acumulado')
    axs[0].grid(True)
    axs[0].legend()

    # Panel central: simulación escalada
    axs[1].plot(V_simulado_escalado, label='Señal simulada', color='purple', linewidth=1)
    for t_inicio, t_fin in intervalos:
        axs[1].axvspan(t_inicio, t_fin, color='cyan', alpha=0.5)
    axs[1].set_title('Simulación $V_t$')
    axs[1].set_xlabel('Tiempo (samples)')
    axs[1].set_ylabel('$V_t$')
    axs[1].grid(True)
    axs[1].legend()

    # Panel derecho: diferencia con misma escala
    axs[2].plot(diferencia, label='Diferencia', color='orange', linewidth=1)
    axs[2].axhline(0, color='red', linestyle='--', linewidth=0.8)
    for t_inicio, t_fin in intervalos:
        axs[2].axvspan(t_inicio, t_fin, color='cyan', alpha=0.5)
    axs[2].set_title('Diferencia ($V_t$ - spikes)')
    axs[2].set_xlabel('Tiempo (samples)')
    axs[2].set_ylabel('Diferencia')
    axs[2].set_ylim(-axs[0].get_ylim()[1]/8,axs[0].get_ylim()[1]/8)
    #axs[2].grid(True)
    axs[2].legend()

    # Anotar parámetros
    texto = f"α = {alpha:.4e}\nAUC = {auc:.3f}"
    axs[1].text(0.02, 0.98, texto, transform=axs[1].transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


# ===================================================================================
# ANOVA / Correlación
# Ángulo 270
# ===================================================================================

# Cargar datos
df = pd.read_csv('/content/drive/MyDrive/resultados_modelo_270_grados.csv')

# --- ANOVA: AUC según alpha ---
grupos = [grupo['auc'].values for _, grupo in df.groupby('alpha')]
f_stat, p_val = f_oneway(*grupos)

print(f"ANOVA entre valores de α: F = {f_stat:.3f}, p = {p_val}")
if p_val < 0.05:
    print("✅ Hay diferencias significativas en AUC según el valor de α.")
else:
    print("❌ No se detectaron diferencias significativas.")

# --- Correlación AUC vs alpha (no lineal) ---
rho, p_corr = spearmanr(df['alpha'], df['auc'])
print(f"Correlación de Spearman: rho = {rho:.3f}, p = {p_corr}")

# ===================================================================================
# Análisis de estabilidad
# Ángulo 270
# ===================================================================================

# Calcular media y desviación estándar por alpha
stability_df = df.groupby('alpha')['auc'].agg(['mean', 'std']).reset_index()

# Gráfico de errorbar con barras de error más transparentes
plt.figure(figsize=(10, 6))
plt.errorbar(
    stability_df['alpha'],
    stability_df['mean'],
    yerr=stability_df['std'],
    fmt='-o',
    ecolor='gray',
    alpha=0.5
)

#plt.title('Estabilidad del modelo por valor de alpha')
plt.xlabel('Alpha (memoria)')
plt.ylabel('AUC promedio ± desv. estándar')
plt.show()

# ===================================================================================
# Análisis de varianza
# Ángulo 270
# ===================================================================================

angulo, bloque_inicio, bloque_fin = 270, 48036610, 49030446

tiempos_activacion = tiempos_de_activacion_spikes(
    grado=angulo,
    start_time=bloque_inicio,
    end_time=bloque_fin
)

coord_spikes = generar_extended_coord_spikes(
    start_time=bloque_inicio,
    end_time=bloque_fin
)

tiempos_barra = intervalos_barra(angulo)
ini, fin = tiempos_barra[0]
duracion_total = fin-ini
duracion_pixel = duracion_total / 16

intervalos_por_electrodo = generar_diccionario_intervalos_barras(
    generar_extended_coord_spikes=coord_spikes,
    tiempos_barra=tiempos_barra,
    angulo=angulo,
    duracion=duracion_total,
    start_time=bloque_inicio
)

celula_a_coordenada = obtener_celula_a_coordenada(
    cluster_path,
    coord_path)

# Ruta al archivo CSV
dataframe_path = '/content/drive/MyDrive/resultados_modelo_270_grados.csv'

varianza_270 = dic_varianza_desviacion(dataframe_path)

print("✅ OK")

var_270=extraer_varianzas(varianza_270)
boxplot_var(var_270)

#####################################################################################

# ===================================================================================
# Simulación del modelo no Markoviano / AUC / Precision / Sensibilidad
# Ángulo 315
# ===================================================================================

angulo, bloque_inicio, bloque_fin = 315, 49036482, 50333767

tiempos_activacion = tiempos_de_activacion_spikes(
    grado=angulo,
    start_time=bloque_inicio,
    end_time=bloque_fin
)

coord_spikes = generar_extended_coord_spikes(
    start_time=bloque_inicio,
    end_time=bloque_fin
)

tiempos_barra = intervalos_barra(angulo)
ini, fin = tiempos_barra[0]
duracion_total = fin-ini
duracion_pixel = duracion_total / 16

intervalos_por_electrodo = generar_diccionario_intervalos_barras(
    generar_extended_coord_spikes=coord_spikes,
    tiempos_barra=tiempos_barra,
    angulo=angulo,
    duracion=duracion_total,
    start_time=bloque_inicio
)

celula_a_coordenada = obtener_celula_a_coordenada(
    cluster_path,
    coord_path)

# --- EJECUTAR EL PROCESO ---
resultados_finales = optimizar_alpha_para_todas_las_celulas(
    tiempos_activacion,
    celula_a_coordenada,
    intervalos_por_electrodo
)

# Convertir a DataFrame
df_resultados = pd.DataFrame.from_dict(resultados_finales, orient='index')
df_resultados.index.name = 'celula'

# Montar Drive (solo necesitas hacerlo una vez por sesión)
drive.mount('/content/drive')

# Guardar en Drive como CSV
ruta_csv = f'/content/drive/MyDrive/resultados_modelo_{angulo}_grados.csv'
df_resultados.to_csv(ruta_csv)

# Confirmación visual
print(f"\n✅ Resultados guardados en: {ruta_csv}")

# ===================================================================================
# Comparación entre simulación y datos
# Ángulo 315
# ===================================================================================

ruta_csv = '/content/drive/MyDrive/resultados_modelo_315_grados.csv'
df_resultados = pd.read_csv(ruta_csv, index_col='celula')

for celula_str, fila in df_resultados.iterrows():
    celula = int(celula_str)
    alpha = fila['alpha']
    auc = fila['auc']

    if celula not in tiempos_activacion:
        continue

    diferencias = tiempos_activacion[celula]
    tiempos_activacion_real = np.cumsum(diferencias)
    conteo = np.arange(len(tiempos_activacion_real))

    tiempos_step = np.insert(tiempos_activacion_real, 0, 0)
    conteo_step = np.insert(conteo, 0, 0)

    # Obtener estímulo U_t
    coord = celula_a_coordenada.get(celula)
    intervalos = list(intervalos_por_electrodo.get(coord, {}).values())
    T = max(tiempos_step[-1], max((fin for _, fin in intervalos), default=0)) + 1000
    T = int(T)

    U_t = np.zeros(T)
    for inicio, fin in intervalos:
        U_t[inicio:fin] = 1

    # Simular señal original
    V_simulado = simular_modelo_no_markoviano(U_t, alpha)

    # Interpolación del conteo real sobre la escala completa
    conteo_interp_real = np.interp(np.arange(T), tiempos_step, conteo_step)

    # Escalamiento de simulación al rango del conteo real
    p98_conteo = np.percentile(conteo_interp_real, 98)
    p98_V = np.percentile(V_simulado, 98)
    escala = p98_conteo / p98_V if p98_V > 0 else 1.0
    V_simulado_escalado = V_simulado * escala

    # Calcular diferencia con misma escala
    diferencia = V_simulado_escalado - conteo_interp_real

    # Crear gráfico con 3 paneles
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.5), sharex=False)

    # Panel izquierdo: spikes reales (escala real)
    axs[0].step(tiempos_step, conteo_step, where='post', label='Spikes reales', linewidth=1.5)
    for t_inicio, t_fin in intervalos:
        axs[0].axvspan(t_inicio, t_fin, color='cyan', alpha=0.5)
    axs[0].set_title(f'Célula {celula} - Spikes')
    axs[0].set_xlabel('Tiempo (samples)')
    axs[0].set_ylabel('Conteo acumulado')
    axs[0].grid(True)
    axs[0].legend()

    # Panel central: simulación escalada
    axs[1].plot(V_simulado_escalado, label='Señal simulada', color='purple', linewidth=1)
    for t_inicio, t_fin in intervalos:
        axs[1].axvspan(t_inicio, t_fin, color='cyan', alpha=0.5)
    axs[1].set_title('Simulación $V_t$')
    axs[1].set_xlabel('Tiempo (samples)')
    axs[1].set_ylabel('$V_t$')
    axs[1].grid(True)
    axs[1].legend()

    # Panel derecho: diferencia con misma escala
    axs[2].plot(diferencia, label='Diferencia', color='orange', linewidth=1)
    axs[2].axhline(0, color='red', linestyle='--', linewidth=0.8)
    for t_inicio, t_fin in intervalos:
        axs[2].axvspan(t_inicio, t_fin, color='cyan', alpha=0.5)
    axs[2].set_title('Diferencia ($V_t$ - spikes)')
    axs[2].set_xlabel('Tiempo (samples)')
    axs[2].set_ylabel('Diferencia')
    axs[2].set_ylim(-axs[0].get_ylim()[1]/8,axs[0].get_ylim()[1]/8)
    #axs[2].grid(True)
    axs[2].legend()

    # Anotar parámetros
    texto = f"α = {alpha:.4e}\nAUC = {auc:.3f}"
    axs[1].text(0.02, 0.98, texto, transform=axs[1].transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

# ===================================================================================
# ANOVA / Correlación
# Ángulo 315
# ===================================================================================

# Cargar datos
df = pd.read_csv('/content/drive/MyDrive/resultados_modelo_315_grados.csv')

# --- ANOVA: AUC según alpha ---
grupos = [grupo['auc'].values for _, grupo in df.groupby('alpha')]
f_stat, p_val = f_oneway(*grupos)

print(f"ANOVA entre valores de α: F = {f_stat:.3f}, p = {p_val}")
if p_val < 0.05:
    print("✅ Hay diferencias significativas en AUC según el valor de α.")
else:
    print("❌ No se detectaron diferencias significativas.")

# --- Correlación AUC vs alpha (no lineal) ---
rho, p_corr = spearmanr(df['alpha'], df['auc'])
print(f"Correlación de Spearman: rho = {rho:.3f}, p = {p_corr}")

# ===================================================================================
# Análisis de estabilidad
# Ángulo 315
# ===================================================================================

# Calcular media y desviación estándar por alpha
stability_df = df.groupby('alpha')['auc'].agg(['mean', 'std']).reset_index()

# Gráfico de errorbar con barras de error más transparentes
plt.figure(figsize=(10, 6))
plt.errorbar(
    stability_df['alpha'],
    stability_df['mean'],
    yerr=stability_df['std'],
    fmt='-o',
    ecolor='gray',
    alpha=0.5
)

#plt.title('Estabilidad del modelo por valor de alpha')
plt.xlabel('Alpha (memoria)')
plt.ylabel('AUC promedio ± desv. estándar')
plt.show()

# ===================================================================================
# Análisis de varianza
# Ángulo 315
# ===================================================================================

angulo, bloque_inicio, bloque_fin = 315, 49036482, 50333767

tiempos_activacion = tiempos_de_activacion_spikes(
    grado=angulo,
    start_time=bloque_inicio,
    end_time=bloque_fin
)

coord_spikes = generar_extended_coord_spikes(
    start_time=bloque_inicio,
    end_time=bloque_fin
)

tiempos_barra = intervalos_barra(angulo)
ini, fin = tiempos_barra[0]
duracion_total = fin-ini
duracion_pixel = duracion_total / 16

intervalos_por_electrodo = generar_diccionario_intervalos_barras(
    generar_extended_coord_spikes=coord_spikes,
    tiempos_barra=tiempos_barra,
    angulo=angulo,
    duracion=duracion_total,
    start_time=bloque_inicio
)

celula_a_coordenada = obtener_celula_a_coordenada(
    cluster_path,
    coord_path)

# Ruta al archivo CSV
dataframe_path = '/content/drive/MyDrive/resultados_modelo_315_grados.csv'

varianza_315 = dic_varianza_desviacion(dataframe_path)

var_315 = extraer_varianzas(varianza_315)
boxplot_var(var_315)

#####################################################################################
#####################################################################################

# ============================================================
# Boxplots
# ============================================================

# Cargar todos los resultados por ángulo
rutas = glob.glob('/content/drive/MyDrive/resultados_modelo_*_grados.csv')
lista_df = []

for ruta in rutas:
    angulo = int(ruta.split('_modelo_')[1].split('_')[0])
    df_temp = pd.read_csv(ruta)
    df_temp['angulo'] = angulo
    lista_df.append(df_temp)

df_todos = pd.concat(lista_df)

# Boxplot de AUC por ángulo
plt.figure(figsize=(12,6))
sns.boxplot(x='angulo', y='auc', data=df_todos, palette='Set2')
#plt.title('Distribución de AUC por ángulo de estimulación')
plt.xlabel('Ángulo de estimulación (grados)')
plt.ylabel('AUC')
plt.grid(True)
plt.show()

# ANOVA entre ángulos
grupos_ang = [grupo['auc'].values for _, grupo in df_todos.groupby('angulo')]
f_stat_ang, p_val_ang = f_oneway(*grupos_ang)

print(f"ANOVA entre ángulos: F = {f_stat_ang:.3f}, p = {p_val_ang:.4e}")
if p_val_ang < 0.05:
    print("✅ Hay diferencias significativas en el rendimiento según el ángulo.")
else:
    print("❌ No se detectaron diferencias significativas entre ángulos.")

all_variances = [var_0, var_45, var_90, var_135, var_180, var_225, var_270, var_315]
angle_labels = ['0', '45', '90', '135', '180', '225', '270', '315']
colores = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']

graficar_box_violin_superpuestos(all_variances, colores, angle_labels, titulo="")

# ============================================================
# Comparación entre ángulos
# ============================================================

# Crear un diccionario con las varianzas para cada ángulo
varianzas_por_angulo = {
    0: varianza_0,
    45: varianza_45,
    90: varianza_90,
    135: varianza_135,
    180: varianza_180,
    225: varianza_225,
    270: varianza_270,
    315: varianza_315,
}

# Lista de ángulos a procesar
angulos_a_procesar = [0, 45, 90, 135, 180, 225, 270, 315]

# Llamar a la función para generar las tablas
generar_tablas_outliers(angulos_a_procesar, varianzas_por_angulo)
