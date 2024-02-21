import sys
sys.path.append('/Users/ximenapaz/github/ITAM/Proyecto_Casas/src')
from src.utils import *
from train import filename
from train import X_test
from train import y_test   
from prep import test

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
