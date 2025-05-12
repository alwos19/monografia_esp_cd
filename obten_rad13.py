import os
from netCDF4 import Dataset
import numpy as np

# Ruta del directorio que contiene los archivos .nc
input_directory = './file'
output_directory = './extracted_variables'  # Directorio donde se guardarán las variables

# Crear el directorio de salida si no existe
os.makedirs(output_directory, exist_ok=True)

# Iterar por cada archivo .nc en el directorio
for file_name in os.listdir(input_directory):
    if file_name.endswith('.nc'):  # Verificar que sea un archivo .nc
        file_path = os.path.join(input_directory, file_name)
        try:
            dataset = Dataset(file_path, mode='r')
            print(f"Procesando archivo: {file_name}")

            # Verificar si la variable 'Rad13' existe en el archivo
            if 'Rad13' in dataset.variables.keys():
                # Extraer los datos de la variable 'Rad13'
                data = dataset.variables['Rad13'][:]
                
                # Convertir MaskedArray a un arreglo NumPy estándar
                data = np.array(data)  # Esto elimina cualquier máscara asociada
                
                # Guardar los datos en un archivo .npy con el nombre del archivo de origen
                output_file = os.path.join(output_directory, f"{file_name.replace('.nc', '_Rad13.npy')}")
                np.save(output_file, data)
                print(f"Variable 'Rad13' guardada en: {output_file}")
            else:
                print(f"La variable 'Rad13' no existe en el archivo: {file_name}")

            # Cerrar el archivo
            dataset.close()

        except FileNotFoundError:
            print(f"El archivo '{file_path}' no existe.")
        except Exception as e:
            print(f"Ocurrió un error al procesar '{file_name}': {e}")

print("\nProcesamiento completado. Los archivos se guardaron en el directorio:", output_directory)