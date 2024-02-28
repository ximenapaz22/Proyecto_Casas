import pytest
from src.utils import *

# Prueba para la función que asigna variables
def test_asignar_variables():
    train_data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6], 'target': [7, 8, 9]})
    
    X, y = asignar_variables(train_data)
    
    assert len(X) == 3  
    assert len(y) == 3
    assert 'feature1' in X.columns  
    assert 'feature2' in X.columns
    assert 'target' not in X.columns  

# Prueba para la función split
    
def datos_prueba():
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    return pd.DataFrame(X), pd.Series(y)

def test_split(datos_prueba):
    X, y = datos_prueba
    X_train, X_test, y_train, y_test = split(X, y, test_size=0.2, random_state=42)
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20

# Prueba para la función entrenar_modelo
def test_entrenar_modelo(datos_prueba):
    X, y = datos_prueba
    modelo = entrenar_modelo(X, y, random_state=42)
    assert isinstance(modelo, XGBRegressor)
    assert modelo.get_params()['random_state'] == 42


# Test para la función cargar_datos

def datos_prueba2():
    datos = pd.DataFrame({'A': [1, 2, None, 4, 5],
                          'B': [None, 2, 3, 4, 5],
                          'C': [1, None, 3, None, 5]})
    return datos

def test_cargar_datos(datos_prueba2):
    datos_csv = datos_prueba2.to_csv('datos_prueba2.csv', index=False)
    datos_cargados = cargar_datos('datos_prueba2.csv')
    assert datos_prueba2.equals(datos_cargados)

# Test para la función encontrar_valores_faltantes
def test_encontrar_valores_faltantes(datos_prueba2):
    valores_faltantes = encontrar_valores_faltantes(datos_prueba2, umbral=2)
    assert valores_faltantes.equals(pd.Series([2, 3], index=['A', 'C']))






