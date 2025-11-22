# Predicción del Movimiento de Nubes para contribuir a una mejor gestión de la operación de plantas de energía fotovoltaica en Colombia.
Este proyecto aborda el desafío crítico de la **variabilidad en la generación de energía solar** en Colombia mediante el desarrollo de un sistema predictivo del movimiento de nubes utilizando **deep learning** e **imágenes satelitales GOES de la NASA**.

## Comenzando 🚀
Estas instrucciones te permitirán obtener una copia del proyecto en funcionamiento en tu máquina local para propósitos de desarrollo y pruebas.

### Pre-requisitos 📋
- Python 3.8 o superior
- 8GB de RAM mínimo (16GB recomendado)
- 200GB de espacio libre para datasets
- GPU con CUDA (opcional pero recomendado para entrenamiento)

### Instalación 🔧
Paso 1: Clonar el repositorio:
##### $git clone https://github.com/alwos19/monografia_esp_cd.git

Paso 2: Crear entorno virtual:
##### $python -m venv venv
#Linux
##### $source venv/bin/activate
#Windows 
##### $venv\Scripts\activate

### Ejecutando los Scripts ⚙️
 Con los siguientes scripts que se encuentran en el directorio **get_images** descargaras y limpiaras las imagenes con formato .npy de las bandas 9 y 13 para el año 2024.

**#Descarga**

 (venv)$ python get_cut_compress.py --function=get_Rad --date_ini=2024-01-01-00:00 --date_fin=2024-12-31-23:59

**#Separar bandas 9 y 13**

(venv)$ python obten_rad13.py

(venv)$ python obten_rad9.py

**#Obtener imágenes validas**

(venv)$ python clean_outliers.py

### Análisis Exploratorio 📋
El directorio **Analisis_Exploratorio** contiene el análisis estadístico de las bandas 9 y 13 mediante dos archivos .ipynb, incluyendo estadísticas descriptivas, caracterización de nubes y patrones temporales, para optimizar los datos destinados al entrenamiento de modelos de deep learning.

### Ejecutando Modelos ⚙️
El directorio **Modelos** contiene los cuadernos (.ipynb) con las arquitecturas de aprendizaje profundo implementadas, destacando los modelos CNN y ConvLSTM en versiones mono y multi-banda, cuyas celdas permiten ejecutar paso a paso el entrenamiento y evaluación para obtener los resultados finales.

## Autores ✒️

* **Estefanía Silva**
* **Jhon Alejandro Jaramillo**

## Expresiones de Gratitud 🎁

Queremos agradecer a nuestro tutor, Julian David Arias Lonodoño, por su orientación constante y por impulsarnos a dar lo mejor de nosotros mismos en cada etapa de este proyecto.
Nuestro reconocimiento a la Universidad de Antioquia por el acceso a los laboratorios de cómputo de alto rendimiento, sin los cuales el entrenamiento de los modelos  no habría sido posible. Así mismo, al tutor Esteban Silva Villa por darnos la idea de usar datos satelitales.
A nuestros amigos y familiares, gracias por su paciencia y por comprendernos en los momentos de mayor presión. Este logro es también suyo.
Por último, agradecemos la excelente dinámica de trabajo que logramos como equipo, basada en la confianza y el compromiso mutuo.







