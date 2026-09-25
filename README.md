# TP Aprendizaje Automático 1 — Predicción de precios de casas (Boston)

Trabajo práctico de regresión de la Tecnicatura en Inteligencia Artificial (Facultad de Ciencias Exactas, Ingeniería y Agrimensura), materia **Aprendizaje Automático 1**, 2026 C2.

**Integrantes:** Aldana Desiré Sánchez · Lisandro Odisio Martinelli · Marisa Silvina Zappalá

El objetivo es predecir `MEDV` (valor mediano de las viviendas, en miles de USD) a partir de las características de cada barrio de Boston, comparando distintos modelos de regresión lineal con scikit-learn. El enunciado completo está en [TrabajoPrácticoRegresiónAA1_2026C2.pdf](TrabajoPrácticoRegresiónAA1_2026C2.pdf).

## Contenido del repositorio

| Ruta | Descripción |
|---|---|
| [codigo.ipynb](codigo.ipynb) | Notebook con todo el trabajo: análisis descriptivo, preprocesamiento, modelos, optimización de hiperparámetros, comparación y conclusión |
| [house-prices-tp.csv](house-prices-tp.csv) | Dataset (556 filas, 13 predictoras + target `MEDV`, con valores faltantes) |
| [requirements.txt](requirements.txt) | Dependencias de Python (versiones fijadas) |
| [graficos_dataset_original/](graficos_dataset_original/) | Histogramas, boxplots, countplots y matrices de correlación (Pearson y Spearman) del dataset original |
| [graficos_dataset_estandarizado/](graficos_dataset_estandarizado/) | Los mismos gráficos sobre las variables ya estandarizadas |
| [graficos_modelo/](graficos_modelo/) | Curvas de error del gradiente descendente y barridos de hiperparámetros (`alpha`, `learning_rate`) |

## Dataset

Variables predictoras: `CRIM`, `ZN`, `INDUS`, `CHAS`, `NOX`, `RM`, `AGE`, `DIS`, `RAD`, `TAX`, `PTRATIO`, `B`, `LSTAT`. Variable objetivo: `MEDV`.

## Metodología

1. **Análisis descriptivo.** Estadísticas, nulos por columna, histogramas/boxplots por variable y matrices de correlación de Pearson y Spearman.
2. **Decisiones sobre variables.**
   - Outliers de `CRIM`: se conservan (son barrios reales con alta criminalidad, no errores de carga).
   - `ZN` se elimina (poco aporte) y `RAD` también (muy correlacionada con `TAX`).
   - `CHAS` se trata como categórica (ya viene codificada 0/1).
3. **Train / test.** Se descartan las filas sin `MEDV` y se separa 80 % / 20 % (`random_state=42`) *antes* de imputar y estandarizar, para evitar data leakage.
4. **Valores faltantes.** Se descartan las filas con demasiados nulos simultáneos; `CHAS` se imputa por la moda y el resto con `KNNImputer` (k=5) sobre datos estandarizados. Imputer y scaler se ajustan solo con train.
5. **Multicolinealidad.** VIF calculado sobre train: ninguna variable supera 5.
6. **Modelos.** Regresión lineal (`LinearRegression`), gradiente descendente implementado a mano (batch, estocástico y mini-batch), Ridge, Lasso y Elastic Net.
7. **Hiperparámetros.** `learning_rate` y épocas del gradiente descendente; `alpha` de Ridge, Lasso y Elastic Net elegido por validación cruzada de 5 folds **solo sobre train** (test se usa una única vez al final).
8. **Comparación.** Métrica principal: RMSE de test (queda en las mismas unidades que `MEDV`). También se reportan R², MAE y la brecha train-test.

## Cómo ejecutarlo

Requiere Python 3.13 (el entorno usado en el desarrollo).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (en Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
jupyter notebook codigo.ipynb
```

El notebook lee `house-prices-tp.csv` desde la carpeta del proyecto y escribe los `.png` en las carpetas `graficos_*`. Usa el backend `Agg` de matplotlib, así que los gráficos se guardan como archivos y no se muestran en pantalla.