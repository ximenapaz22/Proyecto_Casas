import sys
sys.path.append('/Users/ximenapaz/github/ITAM/Proyecto_Casas/src')
from src.utils import *
from train import filename
from train import X_test
from train import y_test   
from prep import test

# Cargar modelo
parser = argparse.ArgumentParser(description='Cargar y evaluar un modelo guardado.')
parser.add_argument('--filename', type=str, default='modelo_final.sav', help='Nombre del archivo que contiene el modelo entrenado.')
args = parser.parse_args()
filename = args.filename
loaded_model = pickle.load(open(filename, 'rb'))

# Calcular el resultado del modelo y mostrarlo
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
