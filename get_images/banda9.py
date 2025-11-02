import os
import re

banda13_dir = "/home/jhon/bandas/banda13/imagenes_validas"
banda9_dir  = "/home/jhon/bandas/banda9/imagenes_validas"

# Expresión para extraer timestamp del nombre
patron = re.compile(r"RadFC_(\d{14})_Rad\d+\.npy")

def extraer_timestamps(directorio):
    timestamps = set()
    for f in os.listdir(directorio):
        match = patron.match(f)
        if match:
            timestamps.add(match.group(1))
    return timestamps

# Obtener timestamps de ambas bandas
t13 = extraer_timestamps(banda13_dir)
t9  = extraer_timestamps(banda9_dir)

# Timestamps que están en Banda9 pero no en Banda13
sobrantes_9 = sorted(list(t9 - t13))

print(f"Total imágenes Banda13: {len(t13)}")
print(f"Total imágenes Banda9 : {len(t9)}")
print(f"Imágenes que SOBRAN en Banda9: {len(sobrantes_9)}\n")

if sobrantes_9:
    print("Timestamps que están en Banda9 pero no en Banda13:")
    for ts in sobrantes_9:
        print(ts)
    with open("sobrantes_banda9.txt", "w") as f:
        for ts in sobrantes_9:
            f.write(ts + "\n")
    print("\n✅ Lista guardada en 'sobrantes_banda9.txt'")
else:
    print("✅ Ambas bandas tienen exactamente los mismos timestamps.")
