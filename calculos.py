# Librerías
from pathlib import Path
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from funciones_basicas import BASE_DIR, paso

matplotlib.use("Agg")  # para gráficos: backend sin ventana, solo escribe archivos PNG









# ---------------------------------------------------------------------------------------------------
# ANALISIS DESCRIPTIVO
# ---------------------------------------------------------------------------------------------------

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





# ----------------------------------------------------------------------------------------------------
# GRAFICOS DE ANALISIS PREVIO
# ----------------------------------------------------------------------------------------------------







# --------------------------------------------------------------------------------------------------
# TRATAMIENTO DE VARIABLES
# --------------------------------------------------------------------------------------------------

# VARIABLES CATEGORICAS
df['CHAS'] = df['CHAS'].astype('category') # ya está codificada.





# --------------------------------------------------------------------------------------------------
# ANALSIS DE VARIABLES Y GRAFICOS
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



####  ####