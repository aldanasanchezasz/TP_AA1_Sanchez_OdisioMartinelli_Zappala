# Graficos con informacion original del dataset.


import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from funciones_basicas import BASE_DIR
from calculos import df

matplotlib.use("Agg")



"""
Genera un set de gráficos por cada columna del dataset.

Lógica:
  - Variable OBJETIVO (MEDV): histograma + boxplot.
  - Variable "categórica" (pocos valores distintos, ej. CHAS):
        countplot (frecuencias).
  - Variable "continua" (muchos valores distintos):
        histograma + boxplot (outliers).
  - Extra: mapa de calor de correlaciones.

  No se grafica la relación de cada variable contra MEDV, salvo en las
  matrices de correlación (Pearson y Spearman) que incluyen a MEDV.

Todos los .png se guardan en la carpeta 'graficos_dataset_original/'.
"""

# ----------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------
CARPETA = BASE_DIR / "graficos_dataset_original"   # se crea al lado de este .py
OBJETIVO = "MEDV"           # variable a predecir
UMBRAL_CATEGORICA = 10      # columnas con <= 10 valores distintos -> categóricas

sns.set_theme(style="whitegrid")
os.makedirs(CARPETA, exist_ok=True)


# Guardamos cada grafico generado en una carpeta llamada 'Graficos'
def guardar(fig, nombre):
    """Guarda la figura y la cierra para liberar memoria."""
    ruta = os.path.join(CARPETA, nombre)
    fig.tight_layout()
    fig.savefig(ruta, dpi=100, bbox_inches="tight")
    plt.close(fig)

# ----------------------------------------------------------------------
# 1) Funciones de graficado
# ----------------------------------------------------------------------
def graficar_objetivo():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[OBJETIVO].dropna(), kde=True, ax=axes[0])
    axes[0].set_title(f"Distribución de {OBJETIVO}")
    sns.boxplot(x=df[OBJETIVO], ax=axes[1])
    axes[1].set_title(f"Boxplot de {OBJETIVO}")
    fig.suptitle(f"{OBJETIVO} — variable objetivo", fontweight="bold")
    guardar(fig, f"00_{OBJETIVO}_objetivo.png")


def graficar_continua(col, idx):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    sns.histplot(df[col].dropna(), kde=True, ax=axes[0])
    axes[0].set_title(f"Distribución de {col}")

    sns.boxplot(x=df[col], ax=axes[1])
    axes[1].set_title(f"Boxplot de {col} (outliers)")

    fig.suptitle(f"{col}", fontweight="bold")
    guardar(fig, f"{idx:02d}_{col}_continua.png")


def graficar_categorica(col, idx):
    fig, ax = plt.subplots(figsize=(6, 4))

    sns.countplot(x=df[col], ax=ax)
    ax.set_title(f"Frecuencia de {col}")

    fig.suptitle(f"{col} (categórica)", fontweight="bold")
    guardar(fig, f"{idx:02d}_{col}_categorica.png")


# ----------------------------------------------------------------------
# 2) Recorrer todas las columnas
# ----------------------------------------------------------------------
graficar_objetivo()

for idx, col in enumerate(df.columns, start=1):
    if col == OBJETIVO:
        continue

    if df[col].nunique() <= UMBRAL_CATEGORICA:
        graficar_categorica(col, idx)
    else:
        graficar_continua(col, idx)


# ----------------------------------------------------------------------
# 3) Gráficos globales
# ----------------------------------------------------------------------
corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Matriz de correlaciones")
guardar(fig, "zz_1_correlaciones.png")


# ----------------------------------------------------------------------
# Relaciones no lineales: matriz de correlación de Spearman
# ----------------------------------------------------------------------

''' Pearson sólo capta relación lineal. Spearman (basado en rangos) capta
    cualquier relación monótona, sea lineal o no. '''

# Gráfico: matriz de correlación de Spearman entre TODAS las variables,
# incluyendo MEDV — análoga a la matriz de Pearson de arriba, pero usando el
# método que capta relación monótona, sea lineal o no. Comparando ambas
# matrices se ve qué pares de variables tienen una relación no lineal entre
# sí (o con MEDV) que Pearson no refleja.
corr_spearman = df.corr(method="spearman", numeric_only=True)
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr_spearman, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Matriz de correlaciones de Spearman (relaciones no lineales)")
guardar(fig, "zz_2_correlaciones_spearman.png")