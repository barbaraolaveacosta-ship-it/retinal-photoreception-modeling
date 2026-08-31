# ============================================================
# Test de Kolmogorov-Smirnov
# ============================================================

from scipy.stats import kstest

for angulo, (start_time, end_time) in tiempos_por_angulo.items():
    intervalos_por_celula = tiempos_de_activacion_spikes(angulo, start_time, end_time)
    inter_eventos = unir_intereventos(intervalos_por_celula)
    inter_eventos = np.sort(inter_eventos)
    corte = int(len(inter_eventos) * 0.95)
    inter_eventos_filtrados = inter_eventos[:corte]

    lambda_hat = 1 / np.mean(inter_eventos_filtrados)

    ks_stat, ks_pvalue = kstest(inter_eventos_filtrados, 'expon', args=(0, 1/lambda_hat))

    print(f"Ángulo {angulo}°:")
    print(f"  λ estimado     = {lambda_hat:.5f}")
    print(f"  Estadístico D  = {ks_stat:.5f}")
    print(f"  Valor-p        = {ks_pvalue}")
    print(f"{ks_stat:.5f} & {ks_pvalue} &  {lambda_hat:.5f} & {'Aceptada' if ks_pvalue > 0.05 else 'Rechazada'}")
    print(f"  {'✅ Aceptado' if ks_pvalue > 0.05 else '❌ Rechazado'}\n")

resultados_test = []

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

    # Opción de recorte (por ejemplo 95%)
    corte = int(len(inter_eventos) * 1.0)  # usa todo o ajusta a 0.95 si quieres el percentil inferior
    inter_eventos_filtrados = inter_eventos[:corte]

    # Estimación de parámetro lambda de la exponencial
    lambda_hat = 1 / np.mean(inter_eventos_filtrados)

    # Test de Kolmogorov-Smirnov
    D, p_value = kstest(inter_eventos_filtrados, 'expon', args=(0, 1/lambda_hat))

    # Determinar aceptación/rechazo
    rechaza_005 = "Sí" if p_value < 0.05 else "No"
    rechaza_001 = "Sí" if p_value < 0.01 else "No"

    resultados_test.append({
        "Ángulo (°)": angulo,
        "λ estimado": round(lambda_hat, 6),
        "D (KS)": round(D, 4),
        "Valor-p": round(p_value, 5),
        "Rechaza H₀ (α=0.05)": rechaza_005,
        "Rechaza H₀ (α=0.01)": rechaza_001
    })

# Crear tabla resumen con Pandas
df_resultados = pd.DataFrame(resultados_test)
print("Tabla resumen del test de Kolmogorov-Smirnov contra distribución exponencial:\n")
display(df_resultados)

# ============================================================
# Test de Anderson-Darling
# ============================================================

from scipy.stats import anderson

for angulo, (start_time, end_time) in tiempos_por_angulo.items():
    intervalos_por_celula = tiempos_de_activacion_spikes(angulo, start_time, end_time)
    inter_eventos = unir_intereventos(intervalos_por_celula)
    inter_eventos = np.sort(inter_eventos)
    corte = int(len(inter_eventos) * 0.95)
    inter_eventos_filtrados = inter_eventos[:corte]

    ad_result = anderson(inter_eventos_filtrados, dist='expon')

    # Buscar valor crítico al 5%
    niveles = ad_result.significance_level.tolist()
    criticos = ad_result.critical_values.tolist()
    idx_5 = niveles.index(5.0)
    critico_5 = criticos[idx_5]

    print(f"Ángulo {angulo}°:")
    print(f"  Estadístico A² = {ad_result.statistic:.5f}")
    print(f"  Valor crítico (5%) = {critico_5:.5f}")
    print(f"{ad_result.statistic:.5f} & {critico_5:.5f} & {'Aceptada' if ad_result.statistic < critico_5 else 'Rechazada'}")
    print(f"  {'✅ Aceptado' if ad_result.statistic < critico_5 else '❌ Rechazado'}\n")

# ============================================================
# Test de distribución temporal
# ============================================================

# ============================================================
# Búsqueda de mejores distribuciones
# ============================================================

# ============================================================
# Análisis de censura
# ============================================================

# ============================================================
# Evaluación de las hipótesis
# ============================================================

