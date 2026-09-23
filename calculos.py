# Analisis descriptivo y tratamiento de variables del dataset





# Librerías
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from funciones_basicas import BASE_DIR, paso
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
matplotlib.use("Agg")  # para gráficos: backend sin ventana, solo escribe archivos PNG





# ---------------------------------------------------------------------------------------------------
# ANALISIS DESCRIPTIVO
# ---------------------------------------------------------------------------------------------------

# Métricas descriptivas del dataset
paso("1) Tamaño dataset")
df = pd.read_csv(BASE_DIR / "house-prices-tp.csv")
df_original = df.copy()   # snapshot crudo, tal cual el CSV, para graficos_iniciales.py
print("Filas y columnas:", df.shape)

paso("2) Primeras filas")
print(df.head())

paso("3) Tipos de datos y nulos")
print(df.info())

paso("4) Estadísticas descriptivas")
print(df.describe())

paso("5) Nulos por columna")
print(df.isna().sum())


# ----------------------------------------------------------------------
# Análisis de variables y datos faltantes
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("ANÁLISIS DE VARIABLES")
print("=" * 70)

# Cantidad y porcentaje de valores faltantes por variable
faltantes = pd.DataFrame({
    "faltantes": df_original.isna().sum(),
    "porcentaje": df_original.isna().mean() * 100
})

print("\nValores faltantes por variable:")
print(faltantes[faltantes["faltantes"] > 0].sort_values("faltantes", ascending=False))


# Cantidad de valores distintos por variable
print("\nCantidad de valores distintos:")
print(df_original.nunique().sort_values())


# Estadísticos descriptivos de las variables numéricas
print("\nEstadísticos descriptivos:")
print(df_original.describe().T)

# ----------------------------------------------------------------------
# Análisis de valores faltantes
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("ANÁLISIS DE VALORES FALTANTES")
print("=" * 70)

for col in df_original.columns:
    cantidad = df_original[col].isna().sum()

    if cantidad > 0:
        porcentaje = cantidad / len(df_original) * 100

        print(f"\nVariable: {col}")
        print(f"Valores faltantes: {cantidad}")
        print(f"Porcentaje: {porcentaje:.2f}%")

# ----------------------------------------------------------------------
# Decisión sobre el tratamiento de valores faltantes
# ----------------------------------------------------------------------

# Se eliminan las filas sin valor en MEDV, ya que es la variable objetivo.
# También se eliminan las filas con 6 o más valores faltantes por presentar
# una cantidad elevada de información ausente.
# CHAS se imputa mediante la moda calculada sobre el conjunto de entrenamiento.
# Las variables numéricas se imputan mediante KNNImputer.
# La estandarización y la imputación se ajustan utilizando únicamente train
# para evitar utilizar información del conjunto de prueba.



# graficos originales






# --------------------------------------------------------------------------------------------------
# TRATAMIENTO DE VARIABLES
# --------------------------------------------------------------------------------------------------

# VARIABLES CATEGORICAS
df['CHAS'] = df['CHAS'].astype('category') # ya está codificada.





# --------------------------------------------------------------------------------------------------
# ANALSIS DE VARIABLES
# --------------------------------------------------------------------------------------------------


##### CRIM ####

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

'''Estos 'outliers' son valores reales, no errores de carga de datos.
Son barrios con tasas de criminalidad muy altas, pero son casos reales. 
Por lo tanto, decidimos mantenerlos en el dataset.'''


# ----------------------------------------------------------------------
# Análisis de variables descartadas
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("ANÁLISIS DE VARIABLES DESCARTADAS")
print("=" * 70)

# ZN
print("\n--- ZN ---")
print(df_original["ZN"].describe())
print(f"Valores faltantes: {df_original['ZN'].isna().sum()}")
print(f"Correlación de ZN con MEDV: {df_original['ZN'].corr(df_original['MEDV']):.4f}")
print(f"Cantidad de ceros en ZN: {(df_original['ZN'] == 0).sum()}")
print(f"Porcentaje de ceros en ZN: {(df_original['ZN'] == 0).mean() * 100:.2f}%")

# RAD
print("\n--- RAD ---")
print(df_original["RAD"].describe())
print(f"Valores faltantes: {df_original['RAD'].isna().sum()}")
print(f"Correlación de RAD con MEDV: {df_original['RAD'].corr(df_original['MEDV']):.4f}")
print(f"Correlación de RAD con TAX: {df_original['RAD'].corr(df_original['TAX']):.4f}")





# ----------------------------------------------------------------------
# Decisión sobre variables descartadas
# ----------------------------------------------------------------------


#### ZN ####
# ZN se descarta debido a su distribución altamente concentrada en cero:
# el 66.91% de sus valores son cero, lo que limita su variabilidad efectiva.
#
df = df.drop(columns='ZN')

#### RAD ####
# RAD se descarta principalmente por su alta correlación con TAX (0.8731),
# indicando una fuerte redundancia entre ambas variables predictoras.
# Además, RAD presenta 28 valores faltantes.

df = df.drop(columns='RAD')


# -------------------------------------------------------------------------------------------------
# VALIDACION CRUZADA
# -------------------------------------------------------------------------------------------------

paso("Train / test split")

antes = len(df)
df = df.dropna(subset=['MEDV'])
print(f"Filas descartadas por no tener MEDV: {antes - len(df)}")

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train = df_train.copy()
df_test = df_test.copy()
print(f"Train: {len(df_train)} filas  |  Test: {len(df_test)} filas")






# ----------------------------------------------------------------------
# TRATAMIENTO DE VALORES FALTANTES
# ----------------------------------------------------------------------

''' Imputación con KNN (KNNImputer): para cada fila con un nulo, busca las
    filas más parecidas (distancia euclídea, usando las columnas que sí
    tienen dato en ambas) y completa el faltante con el promedio de esos
    vecinos.

    Como KNN usa distancia euclídea, hay que estandarizar antes: si una
    variable tiene rango grande (ej. TAX, en cientos) y otra rango chico
    (ej. NOX, entre 0 y 1), la de rango grande domina la distancia y las
    demás casi no pesan. Estandarizando (media 0, desvío 1) todas pesan
    por igual.

    MEDV (target) no se estandariza ni se usa en el KNN: no se debe
    "inventar" el valor que el modelo tiene que aprender a predecir.
    CHAS (categórica 0/1) tampoco se estandariza: se imputa aparte, por la
    moda, porque tiene relación casi nula con el resto de las variables.

    Los datos de test pasan por el mismo tratamiento que los de train, pero
    SIN volver a "aprender" nada de test: el SimpleImputer, el StandardScaler
    y el KNNImputer se ajustan (fit) solo con train, y a test se le aplica
    ese mismo ajuste (transform). Así, el KNNImputer busca vecinos de una
    fila de test únicamente entre las filas de train. '''

paso("Tratamiento de valores faltantes")

# 1) Filas con demasiados nulos simultáneos -> se descartan en train y test
#    por separado (no queda información real en la fila para que KNN
#    encuentre vecinos parecidos)
umbral_nulos = df.shape[1] // 2   # la mitad o más de las columnas vacías

antes = len(df_train)
df_train = df_train[df_train.isna().sum(axis=1) < umbral_nulos]
print(f"Train: filas descartadas por tener {umbral_nulos}+ columnas vacías: {antes - len(df_train)}")

antes = len(df_test)
df_test = df_test[df_test.isna().sum(axis=1) < umbral_nulos]
print(f"Test: filas descartadas por tener {umbral_nulos}+ columnas vacías: {antes - len(df_test)}")

# 2) CHAS: imputación simple por la moda. Se ajusta (fit) solo con train.
chas_imputer = SimpleImputer(strategy='most_frequent')
df_train['CHAS'] = chas_imputer.fit_transform(df_train[['CHAS']].astype('float'))
df_test['CHAS'] = chas_imputer.transform(df_test[['CHAS']].astype('float'))
df_train['CHAS'] = df_train['CHAS'].astype('category')
df_test['CHAS'] = df_test['CHAS'].astype('category')

# 3) Estandarizamos el resto de las variables numéricas (sin MEDV) y
#    aplicamos KNNImputer sobre los datos ya estandarizados. Scaler y
#    KNNImputer se ajustan (fit) solo con train.
columnas_knn = df_train.select_dtypes('number').columns.drop('MEDV')

scaler = StandardScaler()
train_estandarizado = scaler.fit_transform(df_train[columnas_knn])   # ignora los NaN al calcular media/desvío
test_estandarizado = scaler.transform(df_test[columnas_knn])

knn_imputer = KNNImputer(n_neighbors=5)
train_imputado_std = knn_imputer.fit_transform(train_estandarizado)
test_imputado_std = knn_imputer.transform(test_estandarizado)   # vecinos buscados solo entre filas de train

# 4) Revertimos la estandarización para volver a la escala original
df_train[columnas_knn] = scaler.inverse_transform(train_imputado_std)
df_test[columnas_knn] = scaler.inverse_transform(test_imputado_std)

print("Nulos restantes en train:")
print(df_train.isna().sum())
print("Nulos restantes en test:")
print(df_test.isna().sum())

# Además de df_train (escala original), guardamos una versión con los
# predictores ya estandarizados (media 0, desvío 1) — la misma que usó
# KNNImputer internamente — para poder graficar cómo se ven las variables
# una vez estandarizadas. MEDV y CHAS quedan igual (no se estandarizan).
df_train_std = df_train.copy()
df_train_std[columnas_knn] = scaler.transform(df_train[columnas_knn])





# ----------------------------------------------------------------------
# MULTICOLINEALIDAD (VIF)
# ----------------------------------------------------------------------

''' El VIF (Variance Inflation Factor) mide cuánta varianza de cada
    predictora es explicada por TODAS las demás predictoras juntas — a
    diferencia de la matriz de correlación, que solo mira pares de
    variables a la vez. VIF > 5 (o 10, según el criterio) suele tomarse
    como señal de multicolinealidad problemática para un modelo lineal.

    Se calcula solo con TRAIN: es un diagnóstico que informa decisiones
    sobre qué variables usar en el modelo, y esas decisiones no deben
    apoyarse en información de test. '''

paso("Multicolinealidad (VIF)")

predictoras = df_train.select_dtypes('number').drop(columns='MEDV')
vif = pd.DataFrame({
    'variable': predictoras.columns,
    'VIF': [variance_inflation_factor(predictoras.values, i) for i in range(predictoras.shape[1])],
}).sort_values('VIF', ascending=False)

print(vif.to_string(index=False))

# Como ninguna variable supera el umbral de 5, continuamos con nuestro análisis.





# ----------------------------------------------------------------------
# Recomponemos train + test en un único `df`, ya limpio, para el resto del
# análisis descriptivo/gráficos. El modelo y su evaluación deben usar
# df_train / df_test por separado, nunca este df combinado.
# ----------------------------------------------------------------------
df = pd.concat([df_train, df_test]).sort_index()










# aca corremos la carpeta de graficos nuevos










# ---------------------------------------------------------------------------------------------------
# sigue: MODELO
# probar metodos, graficos. etc. Seguir enunciado
# optimizacion de hiperparametros


# ---------------------------------------------------------------------------------------------------
# PREPARAMOS X e y PARA LOS MODELOS
# ---------------------------------------------------------------------------------------------------

''' Usamos los predictores YA estandarizados (df_train_std, que arma
    calculos.py con el mismo StandardScaler que se ajustó solo con train) +
    CHAS como número. A test lo estandarizamos con ESE MISMO scaler, nunca
    ajustando uno nuevo con test (mismo criterio que en la imputación). '''

import numpy as np

X_cols = list(columnas_knn) + ['CHAS']   # todas las predictoras numéricas + CHAS

X_train = df_train_std[X_cols].astype(float).values
y_train = df_train['MEDV'].astype(float).values   # target sin estandarizar

X_test_std = df_test.copy()
X_test_std[columnas_knn] = scaler.transform(df_test[columnas_knn])   # transform, no fit
X_test = X_test_std[X_cols].astype(float).values
y_test = df_test['MEDV'].astype(float).values


# ---------------------------------------------------------------------------------------------------
# MODELO: REGRESION LINEAL (OLS) — baseline
# ---------------------------------------------------------------------------------------------------

''' Regresión lineal múltiple "clásica": mínimos cuadrados ordinarios,
    resuelta de forma cerrada (sin iterar) con sklearn. Es el punto de
    referencia contra el que comparamos el gradiente descendente: si el
    gradiente descendente convergió bien, sus pesos finales deberían
    parecerse mucho a los coeficientes de este modelo. '''

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

paso("Modelo: Regresión lineal (OLS)")

modelo_ols = LinearRegression()
modelo_ols.fit(X_train, y_train)

pred_train_ols = modelo_ols.predict(X_train)
pred_test_ols = modelo_ols.predict(X_test)

print("Coeficientes (OLS):")
print(pd.Series(modelo_ols.coef_, index=X_cols).sort_values())
print(f"Intercepto: {modelo_ols.intercept_:.3f}")

rmse_train_ols = np.sqrt(mean_squared_error(y_train, pred_train_ols))
rmse_test_ols = np.sqrt(mean_squared_error(y_test, pred_test_ols))
print(f"R² train: {r2_score(y_train, pred_train_ols):.3f}  |  R² test: {r2_score(y_test, pred_test_ols):.3f}")
print(f"RMSE train: {rmse_train_ols:.3f}  |  RMSE test: {rmse_test_ols:.3f}")


# ---------------------------------------------------------------------------------------------------
# MODELO: GRADIENTE DESCENDENTE (batch / estocástico / mini-batch)
# ---------------------------------------------------------------------------------------------------

''' Implementamos el gradiente descendente "a mano" (en vez de usar
    SGDRegressor) para poder elegir libremente el tamaño del batch y así
    comparar las 3 variantes con el mismo código:

      - batch_size = n (todo el train): BATCH gradient descent. En cada
        paso calcula el gradiente promediando TODAS las filas de train.
        Convergencia suave y estable, pero cada paso "mira" todo el
        dataset.
      - batch_size = 1: ESTOCÁSTICO (SGD). En cada paso usa una sola fila
        (elegida al azar). Cada paso es más rápido, pero el camino hacia
        el mínimo es ruidoso (la curva de error zigzaguea).
      - 1 < batch_size < n: MINI-BATCH. Punto intermedio entre los dos
        anteriores; es lo más usado en la práctica con datasets grandes.

    Con un dataset tan chico como el nuestro (~424 filas de train) no hay
    ninguna ventaja de cómputo en usar estocástico o mini-batch -> las
    probamos igual para poder comparar y responder la consigna, pero para
    quedarnos con un modelo final usaríamos "batch". '''


def gradiente_descendente(X, y, learning_rate=0.01, epocas=200, batch_size=None, random_state=0):
    """Ajusta una regresión lineal por descenso de gradiente sobre el MSE.

    Devuelve los pesos finales (intercepto + coeficientes) y el historial
    de MSE por época, para graficar Error vs Iteraciones.
    """
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    if batch_size is None:
        batch_size = n   # "None" = batch completo

    # Agregamos una columna de 1s para ajustar el intercepto junto con los
    # demás pesos, en un solo vector.
    X_b = np.hstack([np.ones((n, 1)), X])
    pesos = np.zeros(d + 1)

    historial_error = []

    for _ in range(epocas):
        indices = rng.permutation(n)   # barajamos las filas en cada época
        X_bar = X_b[indices]
        y_bar = y[indices]

        for inicio in range(0, n, batch_size):
            X_batch = X_bar[inicio:inicio + batch_size]
            y_batch = y_bar[inicio:inicio + batch_size]

            error_batch = (X_batch @ pesos) - y_batch                  # predicción - real
            gradiente = (2 / len(X_batch)) * (X_batch.T @ error_batch)  # gradiente del MSE
            pesos -= learning_rate * gradiente                         # paso de actualización

        # Medimos el error sobre TODO el train al final de cada época
        # (no solo el último batch), para que las curvas sean comparables
        # entre las 3 variantes.
        error_epoca = (X_b @ pesos) - y
        historial_error.append(np.mean(error_epoca ** 2))

    return pesos, historial_error


paso("Gradiente descendente: batch vs estocástico vs mini-batch")

n_train = X_train.shape[0]
variantes_gd = {
    "Batch (todo el train)": n_train,
    "Estocástico (1 fila)": 1,
    "Mini-batch (32 filas)": 32,
}

resultados_gd = {}
for nombre, batch_size in variantes_gd.items():
    pesos, historial = gradiente_descendente(
        X_train, y_train, learning_rate=0.01, epocas=200,
        batch_size=batch_size, random_state=42,
    )
    resultados_gd[nombre] = (pesos, historial)
    print(f"{nombre}: MSE final (train) = {historial[-1]:.3f}")

# Gráfico Error vs Iteraciones (una curva por variante) que pide el enunciado
CARPETA_MODELO = BASE_DIR / "graficos_modelo"
os.makedirs(CARPETA_MODELO, exist_ok=True)

fig, ax = plt.subplots(figsize=(9, 5))
for nombre, (pesos, historial) in resultados_gd.items():
    ax.plot(historial, label=nombre)
ax.set_xlabel("Época")
ax.set_ylabel("MSE (train)")
ax.set_title("Gradiente descendente: Error vs Iteraciones")
ax.legend()
fig.savefig(CARPETA_MODELO / "gd_error_vs_iteraciones.png", dpi=100, bbox_inches="tight")
plt.close(fig)

# Comparamos los pesos del batch gradient descent contra los coeficientes de
# OLS: si convergió bien, deberían ser casi iguales (mismo problema, dos
# formas distintas de resolverlo: fórmula cerrada vs. iterando).
pesos_batch, _ = resultados_gd["Batch (todo el train)"]
comparacion_ols_gd = pd.DataFrame({
    "OLS": [modelo_ols.intercept_] + list(modelo_ols.coef_),
    "Gradiente (batch)": pesos_batch,
}, index=["intercepto"] + X_cols)
print("Comparación OLS vs gradiente descendente (batch):")
print(comparacion_ols_gd)



# los datos de test pasan por el mismo tratamiento que los de train.