import argparse
import sys
sys.path.append('/Users/ximenapaz/github/ITAM/Proyecto_Casas/src')
from src.utils import *


parser = argparse.ArgumentParser(description='Carga de datos desde csv')
parser.add_argument('--ruta_train', type=str, default='/Users/ximenapaz/github/ITAM/Proyecto_Casas/data/train.csv', help='Ruta al archivo CSV de entrenamiento.')
parser.add_argument('--ruta_test', type=str, default='/Users/ximenapaz/github/ITAM/Proyecto_Casas/data/test.csv', help='Ruta al archivo CSV de prueba.')
parser.add_argument('--ruta_sample', type=str, default='/Users/ximenapaz/github/ITAM/Proyecto_Casas/data/sample_submission.csv', help='Ruta al archivo CSV de muestra.')

args = parser.parse_args()

# Cargar los datos desde los archivos especificados por el usuario
train = pd.read_csv(args.ruta_train)
test = pd.read_csv(args.ruta_test)
sample = pd.read_csv(args.ruta_sample)

# Mostrar los primeros registros de cada conjunto de datos
print("Datos de entrenamiento:")
print(train.head())
print("\nDatos de prueba:")
print(test.head())
print("\nDatos de muestra:")
print(sample.head())

print(train.describe()) 

#Se buscarán los missing values en nuestra train data
try:
    # Definir el umbral y encontrar columnas con valores faltantes
    umbral = 1
    number_of_cols = encontrar_valores_faltantes(train, umbral)
    print("Número de columnas con valores faltantes:", number_of_cols)
except Exception as e:
    print("Error al encontrar valores faltantes:", e)


#Grafica de análisis
train.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year of sold')
plt.ylabel('House Price')
plt.title('Precio de las casas vs año')

#Se eliminan las filas que tengan valores nulos en la columna SalePrice
train.dropna(axis=0, subset=['SalePrice'], inplace=True)

#Se calcula la correlación

columnas_object = train.select_dtypes(include=['object']).columns
train_sin_object = train.drop(columns=columnas_object)
print(train_sin_object.head())


correlation_matrix = train_sin_object.corr()

# Selecciona las columnas con alta correlación con 'SalePrice'
high_corr_columns = correlation_matrix['SalePrice'][abs(correlation_matrix['SalePrice']) > 0.5].index

# Plotea el heatmap de correlación
plt.figure(figsize=(24, 4))
sns.heatmap(correlation_matrix.loc[['SalePrice'], high_corr_columns], annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of numerical columns with SalePrice')
plt.show()

plt.figure(figsize=(12, 8))
plt.hist(train['SalePrice'], bins=50, color='blue', alpha=0.5)
plt.title('Frequency Distribution for Sale Prices')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.show()

# Histogramas
train[high_corr_columns].hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
# Display the histograms
plt.show()





