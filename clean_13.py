import os
import numpy as np
import shutil

# Directorio de entrada y salida
directorio_entrada = "/home/fisica/bandas/banda13/imagenes_validas"  # Cambia por tu ruta real
directorio_salida = "/home/fisica/bandas/banda13/imagenes_limpias"  # Cambia por tu ruta real
os.makedirs(directorio_salida, exist_ok=True)

archivos = [f for f in os.listdir(directorio_entrada) if f.endswith(".npy") and f.startswith("RadFC_")]

for archivo in archivos:
    ruta = os.path.join(directorio_entrada, archivo)
    try:
        matriz = np.load(ruta)
        # Validar dimensiones y contenido
        if matriz.shape != (920, 920):
            print(f"Dimensión incorrecta en {archivo}, se omite.")
            continue
        if np.isnan(matriz).all() or np.all(matriz == matriz.flat[0]):
            print(f"Imagen vacía o constante en {archivo}, se omite.")
            continue
        # Nuevo filtro: valores fuera de rango físico razonable
        if matriz.max() > 150 or matriz.min() < 0:
            print(f"Valores fuera de rango físico en {archivo}, se omite.")
            continue
        # Nuevo filtro: desviación estándar muy baja (imagen casi uniforme)
        if np.std(matriz) < 1:
            print(f"Imagen casi uniforme (std < 1) en {archivo}, se omite.")
            continue
        # Copiar archivo válido al directorio de salida
        shutil.copy(ruta, os.path.join(directorio_salida, archivo))
    except Exception as e:
        print(f"Error procesando {archivo}: {e}")