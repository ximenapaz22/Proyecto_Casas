import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from train import filename
from train import X_test
from train import y_test
from prep import train
from prep import test
from prep import sample

import pandas as pd

def cargar_datos(ruta):
    """
    Carga los datos desde un archivo CSV.

    Parámetros:
    ruta (str): La ruta del archivo CSV.

    Retorna:
    pd.DataFrame: Un DataFrame de pandas que contiene los datos cargados desde el archivo CSV.
    """
    return pd.read_csv(ruta)

def encontrar_valores_faltantes(dataframe, umbral):
    """
    Encuentra valores faltantes en un DataFrame y devuelve las columnas que tienen más de cierto número de valores faltantes.

    Parámetros:
    dataframe (pd.DataFrame): El DataFrame en el que se buscarán los valores faltantes.
    umbral (int): El número mínimo de valores faltantes que debe tener una columna para ser considerada.

    Retorna:
    pd.Series: Una serie que contiene el número de valores faltantes para cada columna.
    """
    missing_value = dataframe.isnull().sum()
    return missing_value[missing_value > umbral]


def asignar_variables(dataframe):
    """
    Asigna las variables para un modelo a partir de un DataFrame.

    Parámetros:
    dataframe (pd.DataFrame): El DataFrame del cual se extraerán las variables.

    Retorna:
    pd.DataFrame, pd.Series: X, y, donde X es un DataFrame que contiene las características y
                             y es una serie que contiene la variable objetivo.
    """
    X = dataframe.drop(['Id', 'SalePrice'], axis=1).select_dtypes(exclude=['object'])
    y = dataframe['SalePrice']
    return X, y


def split(X, y, test_size=0.2, random_state=None):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.

    Parámetros:
    X (pd.DataFrame): DataFrame que contiene las características.
    y (pd.Series): Serie que contiene la variable objetivo.
    test_size (float or int): Proporción del conjunto de datos que se utilizará como conjunto de prueba.
    random_state (int, opcional): Semilla para la generación de números aleatorios para la división de datos.

    Retorna:
    tuple: Una tupla que contiene los conjuntos X_train, X_test, y_train, y_test.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def entrenar_modelo(X_train, y_train, **kwargs):
    """
    Entrena un modelo de regresión utilizando XGBoost.

    Parámetros:
    X_train (pd.DataFrame): DataFrame que contiene las características de entrenamiento.
    y_train (pd.Series): Serie que contiene la variable objetivo de entrenamiento.
    **kwargs: Otros argumentos opcionales que se pueden pasar al constructor de XGBRegressor.

    Retorna:
    XGBRegressor: El modelo entrenado.
    """
    modelo = XGBRegressor(**kwargs)
    modelo.fit(X_train, y_train)
    return modelo


