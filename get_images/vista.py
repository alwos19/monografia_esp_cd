import os
import re

# === Directorios ===
banda13_dir = "/home/jhon/bandas/banda13/imagenes_validas"
banda9_dir  = "/home/jhon/bandas/banda9/imagenes_validas"

# === Expresión regular para extraer timestamp ===
patron = re.compile(r"RadFC_(\d{14})_Rad\d+\.npy")

def extraer_timestamps(directorio):
    timestamps = set()
    for f in os.listdir(directorio):
        match = patron.match(f)
        if match:
            timestamps.add(match.group(1))
    return timestamps

# === Extraer timestamps de ambas bandas ===
t13 = extraer_timestamps(banda13_dir)
t9  = extraer_timestamps(banda9_dir)

# === Comparar ===
sobrantes_13 = sorted(list(t13 - t9))

print(f"Total imágenes Banda13: {len(t13)}")
print(f"Total imágenes Banda9 : {len(t9)}")
print(f"Faltan {len(sobrantes_13)} imágenes en Banda9 que sí están en Banda13\n")

if sobrantes_13:
    print("Timestamps sobrantes (solo están en Banda13):")
    for ts in sobrantes_13[:20]:  # muestra solo los primeros 20
        print(ts)
    print("\n... (usa la lista completa abajo)")

    # Guardar lista completa
    with open("sobrantes_banda13.txt", "w") as f:
        for ts in sobrantes_13:
            f.write(ts + "\n")
    print("✅ Lista completa guardada en 'sobrantes_banda13.txt'")
else:
    print("✅ Ambas bandas tienen los mismos timestamps.")
