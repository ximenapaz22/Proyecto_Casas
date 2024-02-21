import sys
sys.path.append('/Users/ximenapaz/github/ITAM/Proyecto_Casas/src')
from src.utils import *

from prep import train

#Asignamos las variables para nuestro modelo
X, y = asignar_variables(train)

#Utilizamos el 20% como test y el 80% como training
X_train, X_test, y_train, y_test = split(X, y, test_size=0.2, random_state=123)

#Se utiliza el modelo descenso de gradiente con los datos de entrenamiento
modelo = entrenar_modelo(X_train, y_train)

#Se calcula MSE, aunque este valor puede depender de la escala de nuestros datos
p = modelo.predict(X_test)

mse = mean_squared_error(y_test, p)
print(mse)

#Se calcula el MAE, de acuerdo al costo de nuestras casas el mae obtenido se considera bueno
mae = mean_absolute_error(y_test, p)
print(mae)

# Se calcula el coeficiente de Determinación que indica la proporción de la
#variabilidad en la variable dependiente que es predecible a partir de las
#variables independientes. Entre el valor sea más cercano a 1 nuestra predicción
#es mejor por lo que .85 es una buena predicción
r2 = r2_score(y_test, p)
print(r2)


#Guardar modelo
parser = argparse.ArgumentParser(description='Guardar un modelo entrenado.')
parser.add_argument('--filename', type=str, default='modelo_final.sav', help='Nombre del archivo para guardar el modelo entrenado.')
args = parser.parse_args()
modelo = XGBRegressor()
filename = args.filename
pickle.dump(modelo, open(filename, 'wb'))