import os
import numpy as np

banda13_dir = "/home/jhon/bandas/banda13/imagenes_validas"
banda9_dir = "/home/jhon/bandas/banda9/imagenes_validas"

def verificar_archivos(directorio):
    validos = []
    corruptos = []
    for f in sorted(os.listdir(directorio)):
        if not f.endswith(".npy"):
            continue
        ruta = os.path.join(directorio, f)
        try:
            arr = np.load(ruta)
            if arr.size == 0:
                corruptos.append((f, "vacío"))
            else:
                validos.append(f)
        except Exception as e:
            corruptos.append((f, str(e)))
    return validos, corruptos

# Verificar ambas bandas
validos_13, corruptos_13 = verificar_archivos(banda13_dir)
validos_9, corruptos_9 = verificar_archivos(banda9_dir)

print(f"Banda13 → válidos: {len(validos_13)}, corruptos: {len(corruptos_13)}")
print(f"Banda9  → válidos: {len(validos_9)}, corruptos: {len(corruptos_9)}")

if corruptos_13:
    print("\nArchivos corruptos en Banda13:")
    for f, motivo in corruptos_13[:10]:
        print(f" - {f}: {motivo}")

if corruptos_9:
    print("\nArchivos corruptos en Banda9:")
    for f, motivo in corruptos_9[:10]:
        print(f" - {f}: {motivo}")

# Guardar resultados completos
with open("corruptos_banda13.txt", "w") as f:
    for fn, motivo in corruptos_13:
        f.write(f"{fn} | {motivo}\n")

with open("corruptos_banda9.txt", "w") as f:
    for fn, motivo in corruptos_9:
        f.write(f"{fn} | {motivo}\n")

print("\n✅ Resultados guardados en 'corruptos_banda13.txt' y 'corruptos_banda9.txt'")
