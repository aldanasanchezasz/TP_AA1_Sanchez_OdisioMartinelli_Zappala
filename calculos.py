# Analisis descriptivo y tratamiento de variables del dataset





# Librerías
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from funciones_basicas import BASE_DIR, paso
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Ridge, Lasso
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



#### ZN ####
# No aporta información para el modelo por l oque decidimos eliminarla.
df = df.drop(columns='ZN')



#### RAD ####
# altamente correlacionada con TAX -> misma info. Decidimos eliminarla.
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


# ---------------------------------------------------------------------------------------------------
# MODELO: RIDGE Y LASSO (regularización)
# ---------------------------------------------------------------------------------------------------

''' Ridge (L2) y Lasso (L1) agregan a la función de costo de OLS una
    penalización sobre el tamaño de los coeficientes, controlada por el
    hiperparámetro alpha (a mayor alpha, más penalización = coeficientes
    más chicos):

      - Ridge: penaliza la suma de los coeficientes al CUADRADO. Los achica
        pero nunca los lleva exactamente a 0.
      - Lasso: penaliza la suma de los coeficientes en VALOR ABSOLUTO.
        Puede llevar coeficientes exactamente a 0 -> hace selección de
        variables automática.

    Arrancamos con alpha=1.0 (default de sklearn) para comparar contra OLS;
    el tuneo de alpha es la consigna 5. '''

paso("Modelo: Ridge y Lasso")

modelo_ridge = Ridge(alpha=1.0, random_state=42)
modelo_ridge.fit(X_train, y_train)
pred_train_ridge = modelo_ridge.predict(X_train)
pred_test_ridge = modelo_ridge.predict(X_test)

modelo_lasso = Lasso(alpha=1.0, random_state=42)
modelo_lasso.fit(X_train, y_train)
pred_train_lasso = modelo_lasso.predict(X_train)
pred_test_lasso = modelo_lasso.predict(X_test)


def reportar_metricas(nombre, y_tr, pred_tr, y_te, pred_te):
    """Imprime R², RMSE y MAE en train y test para un modelo ya entrenado."""
    rmse_tr = np.sqrt(mean_squared_error(y_tr, pred_tr))
    rmse_te = np.sqrt(mean_squared_error(y_te, pred_te))
    print(f"{nombre}:")
    print(f"  R²   train: {r2_score(y_tr, pred_tr):.3f}  |  R²   test: {r2_score(y_te, pred_te):.3f}")
    print(f"  RMSE train: {rmse_tr:.3f}  |  RMSE test: {rmse_te:.3f}")
    print(f"  MAE  train: {mean_absolute_error(y_tr, pred_tr):.3f}  |  MAE  test: {mean_absolute_error(y_te, pred_te):.3f}")


reportar_metricas("OLS", y_train, pred_train_ols, y_test, pred_test_ols)
reportar_metricas("Ridge (alpha=1.0)", y_train, pred_train_ridge, y_test, pred_test_ridge)
reportar_metricas("Lasso (alpha=1.0)", y_train, pred_train_lasso, y_test, pred_test_lasso)

# Comparamos los coeficientes de los 3 modelos lado a lado. Nos interesa
# sobre todo ver si Lasso llevó alguna variable a 0 (selección de
# variables) -> por EDA, las candidatas con menor relación con MEDV eran
# CHAS y B.
comparacion_coefs = pd.DataFrame({
    "OLS": modelo_ols.coef_,
    "Ridge": modelo_ridge.coef_,
    "Lasso": modelo_lasso.coef_,
}, index=X_cols).sort_values("OLS")
print("Comparación de coeficientes (OLS vs Ridge vs Lasso):")
print(comparacion_coefs)

variables_eliminadas = list(comparacion_coefs.index[comparacion_coefs["Lasso"] == 0])
print(f"Variables que Lasso llevó a 0: {variables_eliminadas}")

''' ¿Cuál es mejor? Con estos resultados, LASSO gana:

      R² test   OLS=0.341   Ridge=0.348   Lasso=0.433
      RMSE test OLS=7.401   Ridge=7.366   Lasso=6.870
      MAE  test OLS=4.710   Ridge=4.684   Lasso=4.321

    Ridge (que solo achica coeficientes, sin eliminar ninguno) queda
    prácticamente igual que OLS -> confirma lo que ya sabíamos por el VIF
    bajo (máx. ~4): no había multicolinealidad que Ridge necesitara
    corregir.

    Lasso, en cambio, llevó a 0 nada menos que 7 de las 11 variables
    (DIS, NOX, AGE, INDUS, B, CRIM, CHAS) y se quedó solo con LSTAT,
    PTRATIO, TAX y RM. Su R² de TRAIN bajó (0.546 vs 0.647 de OLS) -> a
    primera vista parece "peor", pero en TEST mejora bastante. Lo que está
    pasando es que OLS estaba sobreajustando un poco: la brecha
    train-test de OLS es 0.647-0.341=0.306, mientras que la de Lasso es
    0.546-0.433=0.113 -> mucho más chica, señal de que generaliza mejor.

    Tiene sentido con lo que vimos en el análisis descriptivo: CHAS tenía
    Mutual Information casi nula (~0.02) con MEDV, y varias de las
    variables eliminadas (NOX, INDUS, AGE, DIS) forman parte del mismo
    "cluster" correlacionado entre sí que vimos en el heatmap (aportan
    señal solapada/redundante, aunque ninguna por separado tuviera VIF
    alto). Lasso, al poder eliminarlas del todo, se quedó con las 4
    variables que realmente aportan señal independiente.

    Conclusión: para ESTE split, Lasso (alpha=1.0) es el mejor de los tres
    -> mejor error de test Y modelo más simple (4 variables en vez de 11).
    Ridge no ayudó porque el problema no era multicolinealidad, era ruido
    de variables poco informativas -> ahí Lasso tiene ventaja estructural
    sobre Ridge. Falta ver si esto se sostiene al tunear alpha (consigna
    5) y si Ridge con un alpha mayor lo alcanza. '''


# ---------------------------------------------------------------------------------------------------
# MODELO: ELASTIC NET (regularización L1 + L2)
# ---------------------------------------------------------------------------------------------------

''' Elastic Net combina las dos penalizaciones anteriores en una sola:

        penalización = l1_ratio * (L1, como Lasso) + (1 - l1_ratio) * (L2, como Ridge)

    l1_ratio=1.0 sería Lasso puro, l1_ratio=0.0 sería Ridge puro. Con
    l1_ratio=0.5 (default de sklearn) buscamos un punto intermedio: puede
    seguir llevando coeficientes a 0 (como Lasso) pero de forma más
    "suave" cuando hay variables correlacionadas entre sí -> en vez de
    quedarse arbitrariamente con una del grupo y tirar el resto a 0 (lo
    que hace Lasso), reparte el peso entre ellas. Relevante acá porque
    justamente eliminamos un cluster de variables correlacionadas
    (NOX-INDUS-AGE-DIS). '''

from sklearn.linear_model import ElasticNet

paso("Modelo: Elastic Net")

modelo_elastic = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
modelo_elastic.fit(X_train, y_train)
pred_train_elastic = modelo_elastic.predict(X_train)
pred_test_elastic = modelo_elastic.predict(X_test)

reportar_metricas("Elastic Net (alpha=1.0, l1_ratio=0.5)", y_train, pred_train_elastic, y_test, pred_test_elastic)

comparacion_coefs["ElasticNet"] = pd.Series(modelo_elastic.coef_, index=X_cols)   # por índice, no por posición (comparacion_coefs ya está reordenado)
print("Comparación de coeficientes (OLS vs Ridge vs Lasso vs Elastic Net):")
print(comparacion_coefs)

variables_eliminadas_en = list(comparacion_coefs.index[comparacion_coefs["ElasticNet"] == 0])
print(f"Variables que Elastic Net llevó a 0: {variables_eliminadas_en}")

''' Con l1_ratio=0.5, Elastic Net se comportó justo como se esperaba: un
    punto intermedio entre Ridge y Lasso.

      R² test    OLS=0.341  Ridge=0.348  Lasso=0.433  ElasticNet=0.430
      RMSE test  OLS=7.401  Ridge=7.366  Lasso=6.870  ElasticNet=6.884

    Solo eliminó 2 variables (CRIM y CHAS) en vez de las 7 que eliminó
    Lasso, y sus coeficientes quedan más parecidos en magnitud a los de
    OLS/Ridge (ninguno se "dispara" como pasaba antes de arreglar el bug).
    Aun así, su R² y RMSE de test quedan prácticamente empatados con
    Lasso (0.430 vs 0.433) — es decir, logró casi el mismo poder
    predictivo que Lasso pero conservando 9 de las 11 variables en vez de
    solo 4. Eso lo hace un candidato más "conservador": si el resultado de
    Lasso fuera específico de este split en particular (mucha poda de
    variables), Elastic Net da un resultado similar sin apostar tan fuerte
    a que esas 7 variables realmente no aportan nada. El l1_ratio es otro
    hiperparámetro a tunear en la consigna 5, además de alpha. '''


# ---------------------------------------------------------------------------------------------------
# ¿CONSEGUIMOS UN BUEN FITTING?
# ---------------------------------------------------------------------------------------------------

paso("¿Conseguimos un buen fitting?")

resumen_modelos = pd.DataFrame({
    "OLS":        [r2_score(y_train, pred_train_ols),     r2_score(y_test, pred_test_ols)],
    "Ridge":      [r2_score(y_train, pred_train_ridge),   r2_score(y_test, pred_test_ridge)],
    "Lasso":      [r2_score(y_train, pred_train_lasso),   r2_score(y_test, pred_test_lasso)],
    "ElasticNet": [r2_score(y_train, pred_train_elastic), r2_score(y_test, pred_test_elastic)],
}, index=["R2_train", "R2_test"]).T
resumen_modelos["brecha_train_test"] = resumen_modelos["R2_train"] - resumen_modelos["R2_test"]
print(resumen_modelos)

''' Con los 4 modelos, el R² de test queda entre 0.34 y 0.43. Es decir,
    el mejor de ellos (Lasso/ElasticNet) explica poco menos de la mitad de
    la variación de MEDV -> NO es un buen fitting todavía. Hay margen
    grande de mejora.

    ¿Por qué? Principalmente porque son todos modelos LINEALES, y ya
    vimos en el análisis descriptivo (matriz de Spearman vs Pearson, y
    los gráficos con LOWESS) que varias de las relaciones más fuertes con
    MEDV son NO lineales: CRIM tenía Pearson=-0.23 pero Spearman=-0.51
    (la brecha más grande de todas), y LSTAT, NOX, AGE también mostraban
    curvatura. Una recta (por más regularizada que esté) tiene un techo
    estructural para capturar esas relaciones -> por eso ninguna de las 4
    variantes lineales pasa de R²≈0.43, regularicen o no.

    Lo que SÍ mejoró la regularización fue la brecha train-test: OLS tiene
    la brecha más grande (sobreajusta un poco más), Lasso/ElasticNet la
    brecha más chica (generalizan más consistente, aunque con menor techo
    de R²). Eso es una mejora real, pero no alcanza para decir que el
    fitting en sí es "bueno" en términos absolutos.

    Con el alcance de este TP (solo regresión lineal, con o sin
    regularización) no se puede resolver del todo: para capturar esas
    relaciones no lineales haría falta agregar términos polinómicos/log
    de las variables más curvas (CRIM, LSTAT, NOX) o pasar directamente a
    un modelo no lineal (fuera del alcance de este trabajo, que es
    específicamente sobre regresión lineal). Lo dejamos como una
    limitación reconocida para la conclusión final. '''


# =====================================================================================================
# OPTIMIZACION DE HIPERPARAMETROS (consigna 5)
# =====================================================================================================


# ---------------------------------------------------------------------------------------------------
# Gradiente descendente: variando el learning_rate
# ---------------------------------------------------------------------------------------------------

''' Usamos batch completo (para aislar el efecto del learning_rate del
    ruido de estocástico/mini-batch que ya vimos en la sección anterior) y
    probamos varios valores. '''

paso("Gradiente descendente: variando learning_rate")

learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5]
resultados_lr = {}
for lr in learning_rates:
    pesos, historial = gradiente_descendente(
        X_train, y_train, learning_rate=lr, epocas=300,
        batch_size=n_train, random_state=42,
    )
    resultados_lr[lr] = historial
    print(f"learning_rate={lr}: MSE final (train) = {historial[-1]:.3g}")

''' Con lr=0.0001 y lr=0.001 el error baja pero muy lento: a la época 300
    todavía está lejos del óptimo de OLS (~32). Con lr=0.01 y lr=0.1
    converge bien, cerca del óptimo. Con lr=0.3 y lr=0.5 EXPLOTA: el error
    crece sin control (llega a números astronómicos / infinito) en vez de
    bajar -> el paso es tan grande que en cada actualización se "pasa" del
    mínimo y cada vez rebota más lejos.

    Graficamos solo los learning_rate que no divergen: si incluyéramos
    0.3/0.5 en el mismo gráfico, su escala (números gigantes) aplastaría
    las demás curvas y no se vería nada. '''

fig, ax = plt.subplots(figsize=(9, 5))
for lr in [0.0001, 0.001, 0.01, 0.1]:
    ax.plot(resultados_lr[lr], label=f"lr={lr}")
ax.set_xlabel("Época")
ax.set_ylabel("MSE (train)")
ax.set_title("Gradiente descendente (batch): Error vs Iteraciones según learning_rate")
ax.legend()
fig.savefig(CARPETA_MODELO / "gd_learning_rate.png", dpi=100, bbox_inches="tight")
plt.close(fig)

print("learning_rate=0.3 y 0.5: el error DIVERGE (no se grafican, rompen la escala del resto).")


# ---------------------------------------------------------------------------------------------------
# Gradiente descendente: variando la cantidad de épocas (con el mejor learning_rate)
# ---------------------------------------------------------------------------------------------------

''' Con lr=0.1 (el que mejor convergió arriba), vemos cuántas épocas hacen
    falta realmente para acercarse al óptimo de OLS, y si seguir
    entrenando después de ese punto sigue ayudando o ya no cambia nada
    (la curva se "aplana"). '''

paso("Gradiente descendente: variando épocas (learning_rate=0.1)")

epocas_prueba = [10, 50, 100, 300]
for ep in epocas_prueba:
    _, historial = gradiente_descendente(
        X_train, y_train, learning_rate=0.1, epocas=ep,
        batch_size=n_train, random_state=42,
    )
    print(f"épocas={ep}: MSE final (train) = {historial[-1]:.3f}  (óptimo OLS ~ {rmse_train_ols**2:.3f})")


# ---------------------------------------------------------------------------------------------------
# Ridge y Lasso: variando alpha
# ---------------------------------------------------------------------------------------------------

''' Barremos alpha en escala logarítmica (los valores de alpha razonables
    suelen abarcar varios órdenes de magnitud) y medimos, para cada uno:
      - R² por VALIDACIÓN CRUZADA (5-fold) SOBRE TRAIN -- no sobre test.
        Si eligiéramos alpha mirando el R² de test directamente, el test
        dejaría de ser una medida honesta de generalización (estaríamos
        "espiándolo" para elegir el modelo). Por eso partimos train en 5
        pliegues, entrenamos en 4 y validamos en el restante, rotando, y
        promediamos -> así elegimos alpha sin tocar test para nada. Recién
        en la sección de "comparación final" evaluamos UNA VEZ en test,
        con el alpha ya elegido.
      - cantidad de coeficientes que Lasso llevó a 0 (su "poda" de
        variables, para responder ¿en qué punto empieza a eliminar?) '''

from sklearn.model_selection import KFold, cross_val_score

paso("Ridge y Lasso: variando alpha (con validación cruzada sobre train)")

alphas = np.logspace(-3, 3, 13)   # 0.001, ..., 1000 (13 valores log-espaciados)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

''' Incluimos también a Elastic Net (l1_ratio=0.5 fijo, variando solo
    alpha) en el MISMO barrido con CV -- si tuneamos Ridge y Lasso pero
    dejamos Elastic Net en su alpha por defecto, la comparación final no
    sería justa (estaríamos comparando dos modelos optimizados contra uno
    que no). '''

r2_cv_ridge, r2_cv_lasso, r2_cv_elastic = [], [], []
n_variables_lasso = []

for alpha in alphas:
    r2_cv_ridge.append(
        cross_val_score(Ridge(alpha=alpha, random_state=42), X_train, y_train, cv=kfold, scoring="r2").mean()
    )
    r2_cv_lasso.append(
        cross_val_score(Lasso(alpha=alpha, random_state=42, max_iter=10000), X_train, y_train, cv=kfold, scoring="r2").mean()
    )
    r2_cv_elastic.append(
        cross_val_score(ElasticNet(alpha=alpha, l1_ratio=0.5, random_state=42, max_iter=10000), X_train, y_train, cv=kfold, scoring="r2").mean()
    )
    # La cantidad de variables activas la miramos reentrenando con TODO
    # train (no por fold) -- es solo para describir el modelo, no para
    # elegir alpha, así que no hace falta cross-validarla.
    m_lasso_full = Lasso(alpha=alpha, random_state=42, max_iter=10000).fit(X_train, y_train)
    n_variables_lasso.append(np.sum(m_lasso_full.coef_ != 0))

tabla_alpha = pd.DataFrame({
    "alpha": alphas,
    "R2_cv_Ridge": r2_cv_ridge,
    "R2_cv_Lasso": r2_cv_lasso,
    "R2_cv_ElasticNet": r2_cv_elastic,
    "variables_activas_Lasso": n_variables_lasso,
})
print(tabla_alpha.to_string(index=False))

# Gráfico: R² de validación cruzada vs alpha (escala log en X)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(alphas, r2_cv_ridge, marker="o", label="Ridge")
ax.plot(alphas, r2_cv_lasso, marker="o", label="Lasso")
ax.plot(alphas, r2_cv_elastic, marker="o", label="Elastic Net (l1_ratio=0.5)")
ax.set_xscale("log")
ax.set_xlabel("alpha (escala log)")
ax.set_ylabel("R² (validación cruzada, 5-fold sobre train)")
ax.set_title("R² de validación cruzada según alpha — Ridge vs Lasso vs Elastic Net")
ax.legend()
fig.savefig(CARPETA_MODELO / "ridge_lasso_alpha.png", dpi=100, bbox_inches="tight")
plt.close(fig)

# Gráfico: cantidad de variables activas de Lasso vs alpha (su "camino de
# selección" — a qué alpha empieza a eliminar variables)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(alphas, n_variables_lasso, marker="o", color="tab:orange")
ax.set_xscale("log")
ax.set_xlabel("alpha (escala log)")
ax.set_ylabel("Cantidad de variables con coeficiente ≠ 0")
ax.set_title("Lasso: cuántas variables sobreviven según alpha")
fig.savefig(CARPETA_MODELO / "lasso_seleccion_variables.png", dpi=100, bbox_inches="tight")
plt.close(fig)

''' ¿Qué observamos?

    Con R² de VALIDACIÓN CRUZADA (no de test) como criterio, el óptimo de
    Lasso está en alpha=0.0316 (R²_cv=0.5881) -- muy poca regularización,
    todavía con las 11 variables activas. Lasso recién empieza a "podar"
    variables a partir de alpha~0.3 (pasa de 11 a 10), la poda fuerte es
    entre alpha=0.316 y alpha=1 (de 10 a 4), y desde alpha=10 en adelante
    ya eliminó todo (0 variables, solo predice la media -> R²_cv negativo).
    Es decir: según CV, NO conviene que Lasso pode tanto como podíamos
    antes -- el alpha=1.0 que usamos al principio (que eliminaba 7
    variables) está en una zona donde el R²_cv ya viene cayendo
    (0.5187), no en el óptimo.

    Ridge, en cambio, tiene su óptimo en alpha=10 (R²_cv=0.5898) -- bastante
    más regularización que el alpha=1.0 por defecto (R²_cv=0.5877 ahí),
    pero tampoco tanta como el alpha=100 que habíamos encontrado antes
    mirando el test directamente (ESE resultado era producto de haber
    "espiado" el test, no algo real).

    Elastic Net (l1_ratio=0.5) tiene su óptimo también en alpha=0.0316
    (R²_cv=0.5896, el más alto de los tres en su punto óptimo) -- con tan
    poca regularización, se comporta casi como OLS.

    Conclusión importante: el R²_cv apenas varía entre alpha=0.001 y
    alpha=3 para los tres modelos (todos rondando 0.55-0.59) -- hay una
    meseta ancha, no un pico marcado. Esto sugiere que, para este dataset,
    la regularización ayuda un poco pero no es el factor determinante del
    desempeño (coherente con el VIF bajo que ya habíamos visto: no había
    mucha varianza/multicolinealidad que corregir). '''


# =====================================================================================================
# COMPARACION FINAL DE MODELOS (consigna 6)
# =====================================================================================================

''' Recopilamos TODOS los modelos, ya con los mejores hiperparámetros
    encontrados en esta sección, y los comparamos con una misma métrica.
    Elegimos RMSE de test como métrica principal para decidir el "mejor":
    a diferencia de R² (que es relativo, 0 a 1), RMSE queda en las mismas
    unidades que MEDV (miles de dólares) -> es directamente interpretable
    ("nos equivocamos en promedio X mil dólares"), y penaliza más los
    errores grandes que MAE (relevante si nos importa evitar
    predicciones muy desacertadas en casas caras/baratas). '''

paso("Comparación final de modelos")

# Reentrenamos Ridge, Lasso y Elastic Net con su mejor alpha según
# VALIDACIÓN CRUZADA sobre train (no según test -- test recién se usa acá
# abajo, una sola vez, para reportar el resultado final de cada modelo ya
# elegido).
mejor_alpha_ridge = tabla_alpha.loc[tabla_alpha["R2_cv_Ridge"].idxmax(), "alpha"]
mejor_alpha_lasso = tabla_alpha.loc[tabla_alpha["R2_cv_Lasso"].idxmax(), "alpha"]
mejor_alpha_elastic = tabla_alpha.loc[tabla_alpha["R2_cv_ElasticNet"].idxmax(), "alpha"]

modelo_ridge_tuneado = Ridge(alpha=mejor_alpha_ridge, random_state=42).fit(X_train, y_train)
modelo_lasso_tuneado = Lasso(alpha=mejor_alpha_lasso, random_state=42, max_iter=10000).fit(X_train, y_train)
modelo_elastic_tuneado = ElasticNet(alpha=mejor_alpha_elastic, l1_ratio=0.5, random_state=42, max_iter=10000).fit(X_train, y_train)

# Gradiente descendente con el mejor learning_rate/épocas encontrados (batch)
pesos_gd_tuneado, _ = gradiente_descendente(
    X_train, y_train, learning_rate=0.1, epocas=300, batch_size=n_train, random_state=42,
)
X_train_b = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test_b = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
pred_train_gd = X_train_b @ pesos_gd_tuneado
pred_test_gd = X_test_b @ pesos_gd_tuneado

modelos_finales = {
    "OLS": (pred_train_ols, pred_test_ols),
    "Gradiente descendente (lr=0.1, 300 épocas)": (pred_train_gd, pred_test_gd),
    f"Ridge (alpha={mejor_alpha_ridge:.3g})": (modelo_ridge_tuneado.predict(X_train), modelo_ridge_tuneado.predict(X_test)),
    f"Lasso (alpha={mejor_alpha_lasso:.3g})": (modelo_lasso_tuneado.predict(X_train), modelo_lasso_tuneado.predict(X_test)),
    f"Elastic Net (alpha={mejor_alpha_elastic:.3g}, l1_ratio=0.5)": (modelo_elastic_tuneado.predict(X_train), modelo_elastic_tuneado.predict(X_test)),
}

comparacion_final = pd.DataFrame({
    nombre: {
        "R2_train": r2_score(y_train, pred_tr),
        "R2_test": r2_score(y_test, pred_te),
        "RMSE_train": np.sqrt(mean_squared_error(y_train, pred_tr)),
        "RMSE_test": np.sqrt(mean_squared_error(y_test, pred_te)),
        "MAE_test": mean_absolute_error(y_test, pred_te),
    }
    for nombre, (pred_tr, pred_te) in modelos_finales.items()
}).T.sort_values("RMSE_test")

print(comparacion_final)
print(f"\nMejor modelo según RMSE de test: {comparacion_final.index[0]}")

''' Resultado final, con Ridge/Lasso/ElasticNet tuneados por VALIDACIÓN
    CRUZADA sobre train (RMSE test, de menor a mayor):

      Ridge (alpha=10)                        R2_test=0.382  RMSE_test=7.167  MAE_test=4.517
      Elastic Net (alpha=0.0316, l1_ratio=.5)  R2_test=0.378  RMSE_test=7.194  MAE_test=4.540
      Lasso (alpha=0.0316)                     R2_test=0.357  RMSE_test=7.314  MAE_test=4.641
      Gradiente descendente (lr=0.1)           R2_test=0.345  RMSE_test=7.382  MAE_test=4.696
      OLS                                      R2_test=0.341  RMSE_test=7.401  MAE_test=4.710

    GANADOR: Ridge con alpha=10.

    Importante: estos números son MÁS MODESTOS que los que habíamos
    reportado antes de corregir el data leakage (cuando elegíamos alpha
    mirando el R² de test directamente, Ridge parecía llegar a R²=0.440 y
    Lasso a R²=0.433). Ese resultado anterior estaba inflado -- al elegir
    el alpha que mejor le quedaba justo al test, estábamos ajustando un
    hiperparámetro "a medida" de esos datos puntuales, no aprendiendo algo
    que generalice. Con CV, el R²_test real de Ridge es 0.382, bastante
    más bajo que el 0.440 que habíamos visto antes -- ESA es la diferencia
    concreta que genera hacer bien (o mal) la validación de
    hiperparámetros.

    Dicho esto, la conclusión cualitativa se mantiene: Ridge > Elastic Net
    > Lasso > gradiente descendente ~ OLS, y la brecha train-test de Ridge
    (0.262) sigue siendo más chica que la de OLS (0.306) -> la
    regularización sigue ayudando a generalizar, solo que en una medida
    más modesta de lo que parecía antes de arreglar el leakage. La
    diferencia entre el 1° y el 3° puesto (Ridge vs Lasso, RMSE 7.17 vs
    7.31) es chica -> con otro split de train/test el orden podría variar
    un poco, pero todos los regularizados quedan por delante de OLS/GD sin
    regularizar. '''


# =====================================================================================================
# CONCLUSION DEL TRABAJO (consigna 7)
# =====================================================================================================

''' 1) LIMPIEZA Y PREPARACIÓN DE DATOS
    Partimos de 556 filas con nulos dispersos (hasta 28 por columna) y
    algunos outliers extremos en CRIM. Decidimos CONSERVAR los outliers de
    CRIM (son barrios reales de alta criminalidad, no errores de carga) y
    eliminar ZN y RAD (poco aporte / redundante con TAX). Para los
    valores faltantes, separamos primero train/test (para no filtrar
    información de test hacia train) y usamos KNNImputer sobre variables
    estandarizadas -- descartando antes las ~21-26 filas sin MEDV o con
    demasiados nulos simultáneos, donde no había información real para
    imputar nada. El VIF (máx. ~4) confirmó que no había multicolinealidad
    severa entre las predictoras que sobrevivieron.

    2) MODELOS
    Implementamos y comparamos 5 enfoques de regresión lineal: OLS
    (fórmula cerrada), gradiente descendente (batch/estocástico/
    mini-batch), y regularización Ridge, Lasso y Elastic Net. Para elegir
    el alpha de cada uno usamos validación cruzada (5-fold) SOLO sobre
    train, sin tocar test -- en un primer intento habíamos elegido alpha
    mirando directamente el R² de test, lo que infla artificialmente el
    resultado (ese alpha queda "hecho a medida" del test, no de algo que
    generalice). Con el tuneo corregido, Ridge (alpha=10) terminó siendo
    el mejor de los cinco en las 3 métricas de test (R²=0.382, RMSE=7.17,
    MAE=4.52), por delante de Elastic Net y Lasso ya tuneados de la misma
    forma. La lección más importante de esta etapa: comparar modelos
    regularizados SIN tunear sus hiperparámetros primero (o tuneándolos
    mirando el test) puede llevar a conclusiones equivocadas y a números
    optimistas que no se sostienen.

    3) ¿ES UN BUEN MODELO?
    No del todo. Ni con regularización se superó R²_test ≈ 0.38 -- el
    modelo explica bastante menos de la mitad de la variación de MEDV. La
    causa principal, identificada ya en el análisis descriptivo, es que
    varias de las relaciones más fuertes con el precio (CRIM, LSTAT, NOX,
    AGE) son NO LINEALES (Spearman bastante mayor que Pearson), algo que
    ninguna variante de regresión LINEAL -regularizada o no- puede
    capturar por diseño. Lo que sí logra la regularización es reducir el
    sobreajuste (la brecha train-test de OLS es 0.306; la de Ridge
    tuneado, 0.262) -- generaliza un poco mejor, aunque con un techo de
    precisión limitado por la forma lineal del modelo, y con una mejora
    más modesta de lo que parecía antes de corregir el leakage.

    4) PRÓXIMOS PASOS (fuera del alcance de este TP)
    Para mejorar el fitting haría falta capturar la no linealidad:
    transformar variables (ej. log(CRIM)), agregar términos polinómicos,
    o pasar a un modelo no lineal (árboles, KNN regressor, etc.) -- temas
    de trabajos prácticos posteriores de la cursada. '''

