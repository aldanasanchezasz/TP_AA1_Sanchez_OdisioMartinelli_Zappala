# Funciones básicas para correr los demás scripts.



import pandas as pd
from pathlib import Path



# Carpeta donde está ESTE archivo .py -> todas las rutas cuelgan de acá,
# así funciona sin importar desde qué directorio ejecutes el script.
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:            # por si se corre en celdas/consola sin __file__
    BASE_DIR = Path.cwd()
print("Carpeta base:", BASE_DIR)



# Función para imprimir un separador y un título en la salida
def paso(titulo):
    """Imprime un separador para ubicarte en la salida."""
    print("\n" + "=" * 60)
    print(titulo)
    print("=" * 60)