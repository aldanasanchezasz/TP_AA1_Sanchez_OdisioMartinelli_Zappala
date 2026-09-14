from funciones_basicas import paso
from datos import df

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
