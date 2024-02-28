import sys
sys.path.append('/Users/ximenapaz/github/ITAM/Proyecto_Casas/src')
from src.utils import *
from train import X_test
from train import y_test   
from prep import test


# Define el parser de argumentos de línea de comandos
parser = argparse.ArgumentParser(description='Cargar y evaluar un modelo guardado')
parser.add_argument('--model_file', type=str, default='modelo_final.sav', help='Nombre del archivo que contiene el modelo entrenado')
args = parser.parse_args()

# Cargar el modelo desde el archivo especificado por el usuario
try:
    with open(args.model_file, 'rb') as file:
        loaded_model = pickle.load(file)
    print("El modelo se ha cargado correctamente desde '{}'.".format(args.model_file))
except Exception as e:
    print("Error al cargar el modelo:", e)

# Hacer predicciones con el modelo cargado
predicciones = loaded_model.predict(X_test)

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
