# Descarga, Procesamiento y Modelos Deep Learning para Bandas 9 y 13 del Satélite GOES

Este repositorio contiene los scripts necesarios para descargar imágenes satelitales del GOES. Específicamente, incluye herramientas para obtener las bandas 9 y 13 de cada hora durante todo el año 2024 en el territorio colombiano. Los scripts de descarga se encuentran en el directorio get_images, donde también hay un script especial para igualar el número de archivos entre ambas bandas, requisito esencial para los modelos que entrenan con dos bandas simultáneamente. Todos requieren Python instalado para su ejecución. Además, en Analisis_Exploratorio encontrarás notebooks con el análisis preliminar, mientras que en conv2_Pruebas y Modelos están los desarrollos finales de deep learning.

## Descarga de imágenes y compresión de archivos de las bandas 9 y 13
En una consola con ambiente python activo, ejecutar el siguiente comando:
##### $ python get_cut_compress.py --function=get_Rad --date_ini=2024-01-01-00:00 --date_fin=2024-12-31-23:59

## Organizar en directorios diferentes las imagenes de las bandas 9 y 13
En una consola con ambiente python ejecutar:
##### $ python obten_rad13.py

Luego para la banda 9:
##### $ python obten_rad9.py

## Obtener imagenes validas (eliminación de Outliers)
Las imágenes del satélite GOES pueden contener errores que las hacen inutilizables para análisis y modelos de predicción. Para solucionarlo, hemos creado un script que filtra automáticamente las imágenes válidas. Este script identifica y conserva solo las imágenes con píxeles dentro del rango requerido, asegurando la calidad de los datos para el análisis exploratorio y el entrenamiento de modelos. Organizar las rutas de acuerdo a la necesidad. Ejecutar el siguiente comando:
##### $ python clean_outliers.py
