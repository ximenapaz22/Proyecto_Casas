import sys
sys.path.append('/Users/ximenapaz/github/ITAM/Proyecto_Casas/src')
from src.utils import *

from prep import train

X, y = asignar_variables(train)


#Utilizamos el 20% como test y el 80% como training
X_train, X_test, y_train, y_test = split(X, y, test_size=0.2, random_state=123)

#Se utiliza el modelo descenso de gradiente con los datos de entrenamiento
modelo = entrenar_modelo(X_train, y_train)

#Se calcula MSE, aunque este valor puede depender de la escala de nuestros datos
p = modelo.predict(X_test)

try:
    # Calcular el error cuadrático medio
    mse = mean_squared_error(y_test, p)
    print("Error cuadrático medio (MSE):", mse)
except Exception as e:
    print("Error al calcular el error cuadrático medio:", e)


#Se calcula el MAE, de acuerdo al costo de nuestras casas el mae obtenido se considera bueno
try:
    # Calcular el error absoluto medio
    mae = mean_absolute_error(y_test, p)
    print("Error absoluto medio (MAE):", mae)
except Exception as e:
    print("Error al calcular el error absoluto medio:", e)


# Se calcula el coeficiente de Determinación que indica la proporción de la
#variabilidad en la variable dependiente que es predecible a partir de las
#variables independientes. Entre el valor sea más cercano a 1 nuestra predicción
#es mejor por lo que .85 es una buena predicción
try:
    r2 = r2_score(y_test, p)
    print("Coeficiente de Determinación (R2):", r2)
except Exception as e:
    print("Error al calcular el coeficiente de Determinación:", e)



#Guardar modelo
parser = argparse.ArgumentParser(description='Guardar modelo entrenado')
parser.add_argument('--model_file', type=str, default='modelo_final.sav', help='Nombre del archivo para guardar el modelo entrenado')
args = parser.parse_args()

pickle.dump(modelo, open(args.model_file, 'wb'))

