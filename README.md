En este repositorio encontrarás una serie de scripts que son necesarios para la descarga de imágenes satelitales del GOES (Geostationary Operational Environmental Satellites). Estos scripts nos ayudarán a organizar las imágenes para obtener solo la banda 13 de cada hora, mes y año del 2024. Para llevar a cabo la ejecución de estos scripts, debes estar activo en un equipo con sistema operativo Linux, en este caso Ubuntu 22.04 LTS y tener el programa Python en su versión más reciente para la importación y ejecución de librerías.

## Descarga de imágenes y compresión de archivos
En una consola de Linux ejecutar el siguiente comando:
python get_cut_compress.py --function=get_Rad --date_ini=2024-01-01-00:00 --date_fin=2024-12-31-23:59

## Organizar en un directorio las imágenes de la banda 13
En una consola de Linux ejecutar el siguiente comando:
python obten_rad13.py