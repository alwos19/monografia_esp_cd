import pandas as pd

# Obtener datos del ONI (Índice Oceánico de El Niño)
oni_url = 'https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt'
oni_data = pd.read_csv(oni_url, sep='\s+')

# Crear un diccionario para mapear las siglas a los números de los meses
seas_to_month = {
    'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4, 'AMJ': 5, 'MJJ': 6,
    'JJA': 7, 'JAS': 8, 'ASO': 9, 'SON': 10, 'OND': 11, 'NDJ': 12
}

# Crear la nueva columna 'Mes' usando el diccionario
oni_data['Mes'] = oni_data['SEAS'].map(seas_to_month)


oni_data.rename(columns={'YR': 'Year', 'Mes': 'Month', 'TOTAL': 'ONI'}, inplace=True)
# Convertir el año y el mes en una columna de fecha
oni_data['Fecha'] = pd.to_datetime(oni_data[['Year', 'Month']].assign(DAY=1))

# Resamplear los datos del ONI a frecuencia mensual
oni_data.set_index('Fecha', inplace=True)

oni_data.to_excel('Data\ONIData.xlsx')