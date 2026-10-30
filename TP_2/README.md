## Aprendizaje Automático 1
## Facultad de Ciencias Exactas, Ingeniería y Agrimensura

## Trabajo Práctico 2

## Predicción de lluvia en Australia (clasificación)

**Integrantes:** Aldana Desiré Sánchez · Lisandro Odisio Martinelli · Marisa Silvina Zappalá

## Objetivo
Predecir si al día siguiente va a llover (`RainTomorrow`) a partir de las observaciones meteorológicas diarias de estaciones de Australia. El objetivo final del trabajo es construir un ejecutable de Docker que haga esa predicción a las 23:59 hs de cada día, comparando antes distintos modelos de clasificación (regresión logística, modelo base, redes neuronales, AutoML). El enunciado completo está en [2026C2 - TrabajoPrácticoClasificacionAA1.pdf](2026C2%20-%20TrabajoPr%C3%A1cticoClasificacionAA1.pdf).

## Contenido del repositorio

| Ruta | Descripción |
|---|---|
| [TP-clasificacion-AA1.ipynb](TP-clasificacion-AA1.ipynb) | Notebook con todo el trabajo: análisis descriptivo, limpieza, preprocesamiento, regresión logística y modelo base |
| [weatherAUS_2026C2.csv](weatherAUS_2026C2.csv) | Dataset (145.412 filas, 25 columnas) |
| [requirements.txt](requirements.txt) | Dependencias de Python |
| [2026C2 - TrabajoPrácticoClasificacionAA1.pdf](2026C2%20-%20TrabajoPr%C3%A1cticoClasificacionAA1.pdf) | Enunciado del TP |
| `docker/` *(pendiente, ítem 9)* | Script `inferencia.py`, `requirements.txt`, binarios del modelo/imputers/scalers, `Dockerfile` y `readme.md` |

## Dataset

Observaciones diarias de **49 estaciones** entre el **01/11/2007 y el 24/06/2017**. Cada fila es un día en una ubicación y reúne temperatura, lluvia, evaporación, horas de sol, viento, humedad, presión y nubosidad. La variable objetivo es `RainTomorrow` (`Yes`/`No`).

- Está **desbalanceado**: 75,8 % `No`, 21,9 % `Yes` y 2,2 % sin dato. Un modelo que siempre responda "No llueve" acierta ~78 %, por eso no alcanza con la accuracy y se usan precision, recall, F1, balanced accuracy, ROC-AUC y matriz de confusión.
- Faltantes: ~10 % en `WindGustDir`, `WindGustSpeed`, `WindDir9am`, `Pressure9am` y `Pressure3pm`; entre 38 % y 50 % en `Evaporation`, `Sunshine`, `Cloud9am` y `Cloud3pm`; menos de 5 % en el resto.
- `RainfallTomorrow` se descarta de los predictores porque informa el futuro (*data leakage*).

## Estado del trabajo

Según las entregas parciales del enunciado:

| Entrega | Ítems | Fecha límite | Estado |
|---|---|---|---|
| 1 | 1, 2, 3 (análisis descriptivo, regresión logística, modelo base) | 30/10 | Realizada en el notebook |
| 2 | 4, 5, 6 (hiperparámetros, explicabilidad con SHAP, AutoML con PyCaret) | 13/11 | Pendiente |
| 3 | 7, 8, 9, 10 (red neuronal, comparación de modelos, Docker, conclusión) | 27/11 | Pendiente |

Defensas: jueves 03/12 y 10/12. No se aceptan entregas finales posteriores al 29/11.

## Lo realizado hasta el momento

### 1. Análisis descriptivo
- Tamaño, estadísticas, tipos de datos y porcentaje de nulos por columna.
- Tabla con el significado, unidad, tipo y rango observado de cada variable.
- Gráficos: boxplots e histogramas de las numéricas continuas, barras de `Cloud9am`/`Cloud3pm` (discretas), barras de las categóricas, correlaciones de Pearson y Spearman, gráfico de pares y un **mapa de Australia** con la lluvia por ciudad (coordenadas obtenidas por geocodificación).
- Hallazgos: `Rainfall`, `Evaporation` y `RainfallTomorrow` muy sesgadas a la derecha; `Temp3pm`–`MaxTemp` y `Pressure9am`–`Pressure3pm` correlacionadas por encima de 0,96; la lluvia se concentra en la costa este y sudeste, lo que sugiere que la ubicación aporta información.

### 2. Limpieza y preprocesamiento
1. Se elimina `Unnamed: 0` (índice con valores erróneos).
2. `Location` se reemplaza por `Latitud` y `Longitud`.
3. **Codificación cíclica** (seno/coseno) del mes y de las direcciones del viento (`WindGustDir`, `WindDir9am`, `WindDir3pm`); el año y el día se mantienen numéricos.
4. `RainToday` con One Hot Encoding.
5. Se eliminan las filas sin `RainTomorrow` (2,24 %).
6. **Split estratificado 70 / 15 / 15** (train / validación / test, `random_state=42`) hecho *antes* de imputar y escalar para evitar data leakage.
7. Se eliminan en train las filas con más nulos hasta completar un 5 % de filas descartadas sobre el total.
8. **Imputación**: mediana (`SimpleImputer`) para `Evaporation`, `Sunshine`, `Cloud9am` y `Cloud3pm` (38–50 % de nulos, donde KNN no aporta información y es muy costoso); `KNNImputer` sobre datos estandarizados para el resto, ajustado con una muestra de 10.000 filas de train por costo computacional. Todo se ajusta solo con train.
9. **Multicolinealidad (VIF)**: se eliminan iterativamente variables con VIF > 10 (`Temp3pm`, `Pressure9am`, `Temp9am`). Quedan 26 predictoras, todas con VIF < 10.

### 3. Regresión logística
- `Pipeline` con `StandardScaler` + `LogisticRegression`.
- Métricas: accuracy, precision, recall, F1, balanced accuracy y ROC-AUC; matrices de confusión (absoluta y normalizada) y análisis de falsos positivos y falsos negativos.
- Curva ROC y búsqueda del **umbral que maximiza el F1** en validación: 0,30 (el 0,5 por defecto no es el mejor).
- Evaluación final en test, usado una única vez:

| Umbral | Accuracy | Precision | Recall | F1 | Balanced acc. | ROC-AUC |
|---|---|---|---|---|---|---|
| 0,50 (validación) | 0,847 | 0,726 | 0,510 | 0,599 | 0,727 | – |
| 0,30 (test) | 0,819 | 0,580 | 0,695 | 0,632 | 0,775 | 0,869 |

El F1 en validación (0,638) y en test (0,632) son casi idénticos, por lo que el modelo generaliza bien.

### 4. Modelo base
`DummyClassifier(strategy="most_frequent")`: siempre predice "No". Obtiene accuracy 0,776 pero precision, recall y F1 iguales a 0 y balanced accuracy de 0,5; su curva ROC coincide con la diagonal. La regresión logística aporta valor claro frente a esta línea base.

## Cómo ejecutarlo

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (en Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
jupyter notebook TP-clasificacion-AA1.ipynb
```

Notas:
- El notebook lee `weatherAUS_2026C2.csv` desde esta carpeta.
- Además de lo listado en `requirements.txt`, el notebook importa `scikit-learn`, `seaborn`, `matplotlib`, `geopandas`, `adjustText`, `geopy` y `statsmodels`, que conviene agregar al archivo.
- La geocodificación de las ciudades (ArcGIS) y el mapa descargan datos de internet.
- La imputación con KNN puede tardar algunos minutos.
