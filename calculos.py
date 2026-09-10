# Librerías
import os
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # para gráficos: backend sin ventana, solo escribe archivos PNG
import matplotlib.pyplot as plt
import seaborn as sns


# Carpeta donde está ESTE archivo .py -> todas las rutas cuelgan de acá,
# así funciona sin importar desde qué directorio ejecutes el script.
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:            # por si se corre en celdas/consola sin __file__
    BASE_DIR = Path.cwd()
print("Carpeta base:", BASE_DIR)


# Para mostrar en pantalla todas las columnas y que no corte la salida
pd.set_option("display.max_columns", None)   # que no corte columnas
pd.set_option("display.max_rows", None)      # que no corte filas
pd.set_option("display.width", 120)





# Función para imprimir un separador y un título en la salida
def paso(titulo):
    """Imprime un separador para ubicarte en la salida."""
    print("\n" + "=" * 60)
    print(titulo)
    print("=" * 60)





# ANALISIS DESCRIPTIVO

# Métricas descriptivas del dataset
paso("1) Tamaño dataset")
df = pd.read_csv(BASE_DIR / "house-prices-tp.csv")
print("Filas y columnas:", df.shape)

paso("2) Primeras filas")
print(df.head())

paso("3) Tipos de datos y nulos")
print(df.info())

paso("4) Estadísticas descriptivas")
print(df.describe())

paso("5) Nulos por columna")
print(df.isna().sum())





# VARIABLES CATEGORICAS
df['CHAS'] = df['CHAS'].astype('category') # ya está codificada.





# ANALSIS DE VARIABLES Y GRAFICOS


##### CRIM
''' Como parte del analisis queremos corroborar si los outliers del boxplot son valores realistas o errores.'''
Q1 = df['CRIM'].quantile(0.25)
Q3 = df['CRIM'].quantile(0.75)
IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

print(f"Q1={Q1}, Q3={Q3}, IQR={IQR}")
print(f"Límite superior (outliers por encima de esto): {limite_superior}")

# Filtramos las filas con CRIM por encima del límite superior (~12.05)
# y las ordenamos de menor a mayor para observarlas.

outliers_crim = df[df['CRIM'] > limite_superior]
print(f"Cantidad de outliers de CRIM (> {limite_superior:.2f}): {len(outliers_crim)} de {len(df)} filas")
print(outliers_crim.head())

df_crim_filtrado = df[df['CRIM'] <= limite_superior]    
print(df_crim_filtrado.head())

print(outliers_crim[['CRIM']].describe())
print(df_crim_filtrado[['CRIM']].describe())


# Gráficos estadísticos de los outliers de CRIM
CARPETA_CRIM = BASE_DIR / "graficos_outliers_crim"
os.makedirs(CARPETA_CRIM, exist_ok=True)
sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1) Histograma de CRIM en todo el dataset, con el límite superior marcado
sns.histplot(df['CRIM'], bins=40, kde=True, ax=axes[0, 0])
axes[0, 0].axvline(limite_superior, color="red", linestyle="--",
                   label=f"Límite superior = {limite_superior:.2f}")
axes[0, 0].set_title("Distribución de CRIM (dataset completo)")
axes[0, 0].legend()

# 2) Histograma sólo de los outliers (los 69 valores por encima del límite)
sns.histplot(outliers_crim['CRIM'], bins=30, kde=False, color="tab:orange", ax=axes[0, 1])
axes[0, 1].set_title(f"Distribución de CRIM en los outliers (n={len(outliers_crim)})")

# 3) Boxplot comparativo: CRIM con y sin outliers
comp = pd.concat([
    df_crim_filtrado[['CRIM']].assign(grupo="Sin outliers"),
    outliers_crim[['CRIM']].assign(grupo="Outliers"),
])
sns.boxplot(data=comp, x="grupo", y="CRIM", ax=axes[1, 0])
axes[1, 0].set_title("Boxplot de CRIM: sin outliers vs outliers")

# 4) Dispersión CRIM vs MEDV marcando los outliers
axes[1, 1].scatter(df_crim_filtrado['CRIM'], df_crim_filtrado['MEDV'],
                   alpha=0.4, s=15, label="Resto")
axes[1, 1].scatter(outliers_crim['CRIM'], outliers_crim['MEDV'],
                   alpha=0.7, s=20, color="tab:red", label="Outliers CRIM")
axes[1, 1].set_xlabel("CRIM")
axes[1, 1].set_ylabel("MEDV")
axes[1, 1].set_title("CRIM vs MEDV")
axes[1, 1].legend()

fig.suptitle("Análisis estadístico de outliers de CRIM", fontweight="bold")
fig.tight_layout()
ruta_crim = CARPETA_CRIM / "outliers_crim.png"
fig.savefig(ruta_crim, dpi=100, bbox_inches="tight")
plt.close(fig)
print("Gráfico guardado en:", ruta_crim)

# agregar conclusion


### ZN
# No aporta información para el modelo por l oque decidimos eliminarla.
df = df.drop(columns='ZN')


### INDUS

























'''
RECORDAR SACAR ESTO DE COMENTARIOS (SON LOS GRAFICOS)






# GRAFICOS DE ANALISIS PREVIO

"""
Genera un set de gráficos por cada columna del dataset.

Lógica:
  - Variable OBJETIVO (MEDV): histograma + boxplot.
  - Variable "categórica" (pocos valores distintos, ej. CHAS):
        countplot (frecuencias) + boxplot de MEDV según cada categoría.
  - Variable "continua" (muchos valores distintos):
        histograma + boxplot (outliers) + scatter vs MEDV con recta de tendencia.
  - Extra: mapa de calor de correlaciones y ranking de correlación con MEDV.

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
print("Guardando gráficos en:", CARPETA)


# Guardamos cada grafico generado en una carpeta llamada 'Graficos'
def guardar(fig, nombre):
    """Guarda la figura y la cierra para liberar memoria."""
    ruta = os.path.join(CARPETA, nombre)
    fig.tight_layout()
    fig.savefig(ruta, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("  guardado:", ruta)

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
    datos = df[[col, OBJETIVO]].dropna()          # scatter necesita pares sin NaN
    r = datos[col].corr(datos[OBJETIVO])          # correlación de Pearson

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    sns.histplot(df[col].dropna(), kde=True, ax=axes[0])
    axes[0].set_title(f"Distribución de {col}")

    sns.boxplot(x=df[col], ax=axes[1])
    axes[1].set_title(f"Boxplot de {col} (outliers)")

    sns.regplot(
        data=datos, x=col, y=OBJETIVO, ax=axes[2],
        scatter_kws={"alpha": 0.4, "s": 15},
        line_kws={"color": "red"},
    )
    axes[2].set_title(f"{col} vs {OBJETIVO}   (r = {r:.2f})")

    fig.suptitle(f"{col}", fontweight="bold")
    guardar(fig, f"{idx:02d}_{col}_continua.png")


def graficar_categorica(col, idx):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.countplot(x=df[col], ax=axes[0])
    axes[0].set_title(f"Frecuencia de {col}")

    sns.boxplot(data=df, x=col, y=OBJETIVO, ax=axes[1])
    axes[1].set_title(f"{OBJETIVO} según {col}")

    fig.suptitle(f"{col} (categórica)", fontweight="bold")
    guardar(fig, f"{idx:02d}_{col}_categorica.png")


# ----------------------------------------------------------------------
# 2) Recorrer todas las columnas
# ----------------------------------------------------------------------
paso("2) Gráfico de la variable objetivo")
graficar_objetivo()

for idx, col in enumerate(df.columns, start=1):
    if col == OBJETIVO:
        continue

    if df[col].nunique() <= UMBRAL_CATEGORICA:
        paso(f"{col}: categórica  ->  countplot + boxplot vs {OBJETIVO}")
        graficar_categorica(col, idx)
    else:
        paso(f"{col}: continua  ->  histograma + boxplot + scatter vs {OBJETIVO}")
        graficar_continua(col, idx)


# ----------------------------------------------------------------------
# 3) Gráficos globales
# ----------------------------------------------------------------------
paso("3) Mapa de calor de correlaciones")
corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Matriz de correlaciones")
guardar(fig, "zz_1_correlaciones.png")

paso("4) Correlación de cada variable con el objetivo")
corr_obj = corr[OBJETIVO].drop(OBJETIVO).sort_values()
print(corr_obj)
fig, ax = plt.subplots(figsize=(8, 6))
colores = ["tab:red" if v < 0 else "tab:blue" for v in corr_obj]
corr_obj.plot.barh(ax=ax, color=colores)
ax.set_title(f"Correlación de Pearson con {OBJETIVO}")
ax.axvline(0, color="black", linewidth=0.8)
guardar(fig, "zz_2_correlacion_con_objetivo.png")


# ----------------------------------------------------------------------
# 4) Resumen final
# ----------------------------------------------------------------------
paso("Listo")
archivos = sorted(os.listdir(CARPETA))
print(f"{len(archivos)} archivos en '{CARPETA}/':")
for f in archivos:
    print("  -", f)
'''