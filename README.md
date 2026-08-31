# Dinámica de Fotorecepción en Retina Animal

## Mathematical Modeling and Stochastic Analysis of Retinal Ganglion Cell Activity

Este repositorio contiene el trabajo computacional, matemático y metodológico asociado a la tesis de título **“Dinámica de Fotorecepción en Retina Animal”**, desarrollada para optar al título de **Ingeniera Civil Matemática**.

El proyecto estudia la dinámica del potencial de acción generado por células ganglionares de la retina de ratones en respuesta a estímulos visuales específicos. La investigación combina análisis de datos experimentales, procesos estocásticos, pruebas estadísticas y simulación computacional.

---

## Objetivo

El objetivo principal de esta investigación es realizar un análisis inferencial y desarrollar simulaciones que permitan comprender la dinámica de la actividad neuronal de las células ganglionares de la retina en respuesta a estímulos visuales.

En particular, se busca:

- Analizar los tiempos entre eventos de activación neuronal.
- Evaluar la hipótesis de una dinámica sin memoria.
- Implementar pruebas estadísticas para evaluar el ajuste de modelos estocásticos.
- Analizar procesos puntuales mediante censura selectiva de datos.
- Desarrollar un prototipo de modelo con memoria.
- Comparar simulaciones con los patrones observados experimentalmente.

---

## Hipótesis de estudio

La investigación se desarrolló mediante una evaluación secuencial de tres hipótesis.

### H1 — Sistema sin memoria

Se plantea que la dinámica de la fotorecepción puede describirse mediante un proceso Markoviano de saltos.

Bajo esta hipótesis:

- El sistema no posee memoria.
- Las transiciones dependen únicamente del estado actual.
- Los tiempos entre eventos deberían seguir una distribución exponencial.

---

### H2 — Procesos puntuales censurados

Ante el rechazo de la primera hipótesis, se evaluó si una censura selectiva de ciertos eventos permitía aproximar parcialmente la dinámica a un proceso sin memoria.

Para ello, se eliminaron valores extremos de los tiempos entre eventos y se repitieron las pruebas estadísticas de ajuste.

---

### H3 — Modelo con memoria

Debido al rechazo de las hipótesis anteriores, se desarrolló un modelo que incorpora explícitamente memoria en la dinámica del sistema.

El modelo utiliza un kernel convolucional para incorporar la influencia de estados anteriores y representar una dinámica no Markoviana.

---

# Datos experimentales

Los datos analizados corresponden a registros de actividad neuronal obtenidos mediante una matriz de multielectrodos de 16 × 16.

Cada electrodo representa una posición espacial dentro de una matriz bidimensional utilizada para registrar la actividad eléctrica de las células ganglionares de la retina.

Los datos incluyen información sobre:

- Tiempos de spikes.
- Asociación entre células y electrodos.
- Coordenadas espaciales.
- Estímulos visuales.
- Intervalos temporales.
- Ángulos de estimulación.

---

## Estímulo analizado

El análisis principal se realizó utilizando el estímulo visual de barras en movimiento.

La estimulación se organizó en ocho direcciones:

| Ángulo |
|---|
| 0° |
| 45° |
| 90° |
| 135° |
| 180° |
| 225° |
| 270° |
| 315° |

Cada dirección fue analizada como un subconjunto independiente de datos.

---

# Metodología

El flujo general del proyecto fue el siguiente:

```text
Datos experimentales
        ↓
Carga de archivos HDF5 y Excel
        ↓
Organización de células y electrodos
        ↓
Segmentación por ángulo de estimulación
        ↓
Extracción de tiempos de spikes
        ↓
Cálculo de tiempos inter-evento
        ↓
Análisis exploratorio
        ↓
Evaluación de modelos estocásticos
        ↓
Tests de bondad de ajuste
        ↓
Modelo con memoria
        ↓
Simulación y comparación
```

---

## Metodología


El proyecto se desarrolló utilizando:

-Python
-Jupyter Notebook
-Google Colab
-HDF5
-Excel


Las principales bibliotecas incluyen:

-NumPy
-Pandas
-Matplotlib
-SciPy
-h5py
-scikit-learn
-NetworkX
