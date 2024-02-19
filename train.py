from prep import train
from prep import train_test_split
from prep import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import pickle

#Asignamos las variables para nuestro modelo
X = train.drop(['Id', 'SalePrice'], axis=1).select_dtypes(exclude=['object'])
y = train['SalePrice']

#Utilizamos el 20% como test y el 80% como training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#Se utiliza el modelo descenso de gradiente con los datos de entrenamiento
modelo = XGBRegressor()
modelo.fit(X_train, y_train, )

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

# save the model to disk
filename = 'modelo_final.sav'
pickle.dump(modelo, open(filename, 'wb'))
