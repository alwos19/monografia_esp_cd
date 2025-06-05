#Librerias
import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (mean_absolute_error, 
                             r2_score,
                             root_mean_squared_error)
#from sktime.performance_metrics.forecasting import mean_absolute_scaled_error

#Factor de Inflación de la Varianza VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error


# Función para graficar atributos en barras, pair plot o box-plot
def multiple_plot(ncols, data, columns, target_var, plot_type, title, rot): 
    '''
    Returns the figure build from input parameters.

        Parameters:
            ncols       [integer]    Number of columns for subplots.
            data        [dataframe]  Features dataframe.
            columns     [list]       List of names of featutes in dataframe to plot.
            target_var  [string]     Name of column of target variable or feature.
            plot_type   [string]     Name of graphic. [countplot, boxplt or scatterplot]
            title       [string]     Title for figure
            rot         [integer]    Rotation angle for x axis labels
        Returns:
            Plot of figure
        
    Ejemplos:    
        multiple_plot(1, d , None, 'bad_credit', 'countplot', 'Frecuencia de instancias para la variable bad_credit',0)
        multiple_plot(1, d , 'purpose', 'age_yrs', 'boxplot', 'Distribución de la variable próposito vs la edad',90)
        multiple_plot(1, d , numCols, None, 'scatterplot', 'Relación entre las variables numéricas',30)
        multiple_plot(3, d , catCols, None, 'countplot', 'Frecuencia de instancias para variables categóricas',30)
        multiple_plot(3, d , catCols, 'age_yrs', 'boxplot', 'Distribución de la variables categóticas vs. la edad',30)
    '''
    
    
    # Paletas de colores y colores de las gráficas
    paletas = ['nipy_spectral','hsv','jet_r','Paired','Set2','Dark2','tab10','husl','mako']
    color = ['steelblue','forestgreen', 'amber']  


    # Parámetros iniciales
    title_dist = 1.1  # Ajusta la distancia vertical del título en el gráfico
    x = -1            # Ubicación en el eje x del gráfico
    y =  0            # Ubicación en el eje y del gráfico
    nrows = 1         # Número inicial de filas

    
    # Ajustar el número de filas según el tipo de gráfico y la cantidad de columnas
    if isinstance(columns, list):
        nrows = math.ceil(len(columns) / ncols)

    # Crear el gráfico según el tipo especificado
    if ((nrows <= 1 and ncols <= 1) or plot_type == 'scatterplot'):

        # Countplot
        if plot_type == 'countplot':
            # Configurar el gráfico countplot
            fig, axes = plt.subplots(1, 1, figsize=(6, 4))
            ax_cond = axes
            sns.countplot(data=data,
                          x=target_var,
                          ax=axes,
                          palette=paletas[0],
                          zorder=1,
                          order=data[target_var].value_counts().index,
                          alpha=0.8
                          )
            # Personalizar el eje x
            ax_cond.set_xticklabels(ax_cond.get_xticklabels(), rotation=rot)
            # Configurar título
            ax_cond.set_title(title, fontsize=14, fontweight="bold", y=title_dist)

        # Boxplot
        elif plot_type == 'boxplot':
            # Configurar el gráfico boxplot
            fig, axes = plt.subplots(1, 1, figsize=(6, 4))
            ax_cond = axes
            sns.boxplot(data=data,
                        x=columns,
                        y=target_var,
                        ax=axes,
                        palette=paletas[0],
                        zorder=1
                        )
            # Personalizar el eje x
            ax_cond.set_xticklabels(ax_cond.get_xticklabels(), rotation=rot)
            # Configurar título
            ax_cond.set_title(title, fontsize=14, fontweight="bold", y=title_dist)

            
        # Scatterplot Matrix (Pairplot)
        elif plot_type == 'scatterplot':
            # Configurar el gráfico pairplot
            plot = sns.pairplot(data[columns],
                                palette=paletas[0],
                                diag_kws={'color': color[1]},
                                plot_kws={'color': color[0]},
                                diag_kind='kde'
                                )
            # Ajustar el tamaño del gráfico
            plot.fig.set_size_inches(12, 12)
            # Añadir un título al pairplot
            plot.fig.suptitle(title, fontsize=14, fontweight="bold")
            # Ajustar el diseño para evitar solapamientos
            plt.subplots_adjust(top=0.9)
            # Mostrar el gráfico
            plt.show()            

    # Graficar más de un subplot
    else:
        # Crear subplots con el número especificado de filas y columnas
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, (nrows * 3) + 1))

        for i, column in enumerate(columns):
            x = x + 1

            # Reiniciar x e incrementar y si x alcanza el número de columnas
            if x >= ncols:
                y = y + 1
                x = 0

            # Configurar el subplot actual
            if nrows == 1:
                ax_cond = axes[i]
                title_dist = 1.1
            else:
                ax_cond = axes[y, x]

            # Crear el gráfico según el tipo especificado
            if plot_type == 'countplot':
                # Countplot
                sns.countplot(data=data,
                              x=column,
                              ax=ax_cond,
                              palette=paletas[0],
                              zorder=1,
                              edgecolor='black',
                              linewidth=0.5,
                              order=data[column].value_counts().index
                              )

            elif plot_type == 'boxplot':
                # Boxplot
                sns.boxplot(data=data,
                            x=column,
                            y=target_var,
                            ax=ax_cond,
                            palette=paletas[0],
                            zorder=1
                            )

            # Añadir cuadrícula en el eje y
            ax_cond.grid(axis='y', zorder=0)
            # Personalizar el eje x
            ax_cond.set_xticklabels(ax_cond.get_xticklabels(), rotation=rot)
            # Configurar título del subplot
            ax_cond.set_title(column, fontsize=10)
            # Ajustar tamaño de las etiquetas
            ax_cond.tick_params(labelsize=8)
            # Limpiar etiquetas del eje x
            ax_cond.set_xlabel("")

        # Ajustar el diseño y el título general del conjunto de subplots
        fig.tight_layout()
        fig.suptitle(title, fontsize=14, fontweight="bold", y=title_dist - 0.15)
        plt.subplots_adjust(top=0.9)

        # Eliminar subplots sin datos si hay más de una fila
        if nrows > 1:
            for ax in axes.flat:
                if not bool(ax.has_data()):
                    fig.delaxes(ax)  # Eliminar si no hay datos en el subplot

                    
# Funcion para imprimir la curva ROC
def plot_roc_curve(fpr, tpr):
    '''
    Funcion para imprimir la curva ROC
    '''
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

# Función para convertir una matriz de correlación de pandas en formato tidy    
def tidy_corr_matrix(corr_mat):
    '''
    Función para convertir una matriz de correlación de pandas en formato tidy
    '''
    corr_mat = corr_mat.stack().reset_index()
    corr_mat.columns = ['variable_1','variable_2','r']
    corr_mat = corr_mat.loc[corr_mat['variable_1'] != corr_mat['variable_2'], :]
    corr_mat['abs_r'] = np.abs(corr_mat['r'])
    corr_mat = corr_mat.sort_values('abs_r', ascending=False)
    
    return(corr_mat)

#Función para calcular VIF (Variance Inflation Factor):
   

def checkVIF(X):
    '''
    Se Utiliza VIF para solucionar la multicolinealidad. VIF indica el grado de indecencia de esa variable. 
    Los valores de los umbrales típicos que se suelen utilizar son entre 5 y 10, siendo más exigentes los valores más bajos.
    '''    
    vif = pd.DataFrame()
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return(vif)

def identificar_outliers(df, numCols):
    """
    Identifica los índices de los valores outliers en un DataFrame para una lista de variables numéricas.

    Parámetros:
    df (pd.DataFrame): El DataFrame que contiene los datos.
    numCols (list): Lista de nombres de las columnas numéricas a analizar.

    Retorna:
    list: Lista de índices de los valores outliers.
    """
    # Lista para almacenar los índices de los outliers
    outliers_indices = []

    for var in numCols:
        # Calcular el rango intercuartílico (IQR)
        Q1 = df[var].quantile(0.25)
        Q3 = df[var].quantile(0.75)
        IQR = Q3 - Q1

        # Definir los límites para identificar outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identificar los índices de los outliers
        outliers = df[(df[var] < lower_bound) | (df[var] > upper_bound)].index.tolist()
        
        # Agregar los índices a la lista de outliers
        outliers_indices.extend(outliers)

    return outliers_indices

def eval_model(model, X_train, y_train):
    """
    Permite evaluar el rendimiento de un modelo de regresión utilizando varias métricas.

    Parámetros:
    model (object): El modelo de regresión que se va a evaluar.
    X_train (array-like): Conjunto de características de entrenamiento.
    y_train (array-like): Valores reales de la variable objetivo para el conjunto de entrenamiento.

    Retorna:
    dict: Un diccionario que contiene las siguientes métricas de evaluación:
        - "mae": Error absoluto medio (Mean Absolute Error).
        - "rmse": Raíz del error cuadrático medio (Root Mean Squared Error).
        - "r2": Coeficiente de determinación (R²).
        - "mase": Error absoluto medio escalado (Mean Absolute Scaled Error).
    """
    y_pred = model.predict(X_train)

    metrics = {
        "mae": round(mean_absolute_error(y_train, y_pred), 5),
        "rmse": round(root_mean_squared_error(y_train, y_pred), 5),
        "r2": round(r2_score(y_train, y_pred), 5),
        "mase": round(mean_absolute_scaled_error(y_train, y_pred, y_train=y_train), 5)
    }

    return metrics

def plot_param_perf(x, y_data, title, x_label, y_label):
    """
    Grafica el rendimiento del modelo en función de un parámetro ajustado.

    Parámetros:
    x (iterable): Valores del parámetro ajustado.
    y_data (dict): Diccionario con las métricas de rendimiento para entrenamiento y prueba.
    title (str): Título del gráfico.
    x_label (str): Etiqueta del eje x.
    y_label (str): Etiqueta del eje y.
    """
    y_train = y_data["train"]
    y_test = y_data["test"]

    # Graficar las métricas de rendimiento para entrenamiento y prueba
    sns.lineplot(x=x, y=y_train, label="train")
    sns.lineplot(x=x, y=y_test, label="test")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    

def search_param(base_model, X_train, y_train, X_test, y_test, base_params, search_param, search_range):
    """
    Busca el mejor valor para un parámetro específico del modelo, evaluando el rendimiento
    en términos de R² y MASE para los conjuntos de entrenamiento y prueba.

    Parámetros:
    base_model (object): El modelo base que se va a ajustar.
    X_train (array-like): Conjunto de características de entrenamiento.
    y_train (array-like): Valores reales de la variable objetivo para el conjunto de entrenamiento.
    X_test (array-like): Conjunto de características de prueba.
    y_test (array-like): Valores reales de la variable objetivo para el conjunto de prueba.
    base_params (dict): Parámetros base del modelo.
    search_param (str): El nombre del parámetro que se va a ajustar.
    search_range (iterable): Rango de valores para el parámetro que se va a buscar.

    Retorna:
    tuple: Dos diccionarios que contienen las métricas R² y MASE para los conjuntos de entrenamiento y prueba.
    """
    r2_train = []
    r2_test = []
    mase_train = []
    mase_test = []

    for param in search_range:
        
        # Actualizar los parámetros del modelo con el valor actual del parámetro en búsqueda
        model_params = base_params
        model_params[search_param] = param
        current_model = base_model.set_params(**model_params)

        print(f"Ajustando para {search_param}={param}")
        current_model.fit(X_train, y_train)

        # Guardar R² para entrenamiento y prueba
        r2_train.append(r2_score(y_train, current_model.predict(X_train)))
        r2_test.append(r2_score(y_test, current_model.predict(X_test)))

        # Guardar MASE para entrenamiento y prueba
        mase_train.append(
            mean_absolute_scaled_error(
                y_train, current_model.predict(X_train), y_train=y_train
            )
        )
        mase_test.append(
            mean_absolute_scaled_error(
                y_test, current_model.predict(X_test), y_train=y_test
            )
        )

    # Guardar métricas en diccionarios
    r2_scores = {"train": r2_train, "test": r2_test}
    mase_scores = {"train": mase_train, "test": mase_test}
    return r2_scores, mase_scores