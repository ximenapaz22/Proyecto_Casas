from src.utils import *

# Cargar modelo
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)

#Predicciones
test_predictions = loaded_model.predict(test.drop('Id', axis=1).select_dtypes(exclude=['object']))

# Crear un DataFrame con las predicciones
predictions = pd.DataFrame({'Id': test['Id'], 'SalePrice': test_predictions})

# Guardar las predicciones en un archivo CSV
predictions.to_csv('predictions.csv', index=False)

# Mostrar las predicciones
print(predictions)
