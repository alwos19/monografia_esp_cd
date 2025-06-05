from pydataxm.pydatasimem import ReadSIMEM

def get_data_simem()-> None:
    """
    This function retrieves data from the SIMEM database using the ReadSIMEM class from the pydataxm.pydatasimem module.
    It fetches data for two datasets: 'AECA28' and 'B0E933', with a specified date range.
    The retrieved data is then saved to Excel files in the 'Data' directory.
    """
    # Define the datasets and their corresponding names

    datasets = {'AECA28':'VertimientosHidricos','B0E933':'ReservasHidraulicasEnergía'}
    fecha_inicio = '2013-01-01'
    fecha_fin = '2025-03-31'

    for dataset_id in datasets:

        generacion = ReadSIMEM(dataset_id, fecha_inicio, fecha_fin)
        data = generacion.main(filter=False)

        data.to_excel(f'Data\{datasets[dataset_id]}.xlsx', index=False)
    
    return None