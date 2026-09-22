# Mismos graficos que graficos_iniciales.py, pero sobre las variables ya
# estandarizadas (media 0, desvio 1) que arma calculos.py (df_train_std).
# Sirve para observar como cambian las distribuciones una vez estandarizadas.
#
# OJO: las matrices de correlacion (Pearson y Spearman) dan EXACTAMENTE
# iguales a las de graficos_iniciales.py -> ambos coeficientes son
# invariantes a la escala (estandarizar no cambia la correlacion entre dos
# variables, solo su media y su desvio). Lo que SI cambia es cada grafico
# individual (histograma/boxplot): ahora todas las variables quedan
# centradas en 0 y con la misma dispersion, comparables entre si.


import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from funciones_basicas import BASE_DIR
from calculos import df_train_std as df

matplotlib.use("Agg")


# ----------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------
CARPETA = BASE_DIR / "graficos_dataset_estandarizado"
OBJETIVO = "MEDV"           # no se estandariza (es el target)
UMBRAL_CATEGORICA = 10

sns.set_theme(style="whitegrid")
os.makedirs(CARPETA, exist_ok=True)


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
    fig.suptitle(f"{OBJETIVO} — variable objetivo (sin estandarizar)", fontweight="bold")
    guardar(fig, f"00_{OBJETIVO}_objetivo.png")


def graficar_continua(col, idx):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    sns.histplot(df[col].dropna(), kde=True, ax=axes[0])
    axes[0].set_title(f"Distribución de {col} (estandarizada)")

    sns.boxplot(x=df[col], ax=axes[1])
    axes[1].set_title(f"Boxplot de {col} (estandarizada)")

    fig.suptitle(f"{col}", fontweight="bold")
    guardar(fig, f"{idx:02d}_{col}_continua.png")


def graficar_categorica(col, idx):
    fig, ax = plt.subplots(figsize=(6, 4))

    sns.countplot(x=df[col], ax=ax)
    ax.set_title(f"Frecuencia de {col}")

    fig.suptitle(f"{col} (categórica, no se estandariza)", fontweight="bold")
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
ax.set_title("Matriz de correlaciones (Pearson) — datos estandarizados")
guardar(fig, "zz_1_correlaciones.png")

corr_spearman = df.corr(method="spearman", numeric_only=True)
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr_spearman, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Matriz de correlaciones de Spearman — datos estandarizados")
guardar(fig, "zz_2_correlaciones_spearman.png")
