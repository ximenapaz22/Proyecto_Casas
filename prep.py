import sys
sys.path.append('/Users/ximenapaz/github/ITAM/Proyecto_Casas/src')
from src.utils import *



#Funcion para importar
ruta_archivo_train = '/Users/ximenapaz/github/ITAM/Proyecto_Casas/data/train.csv'
train = cargar_datos(ruta_archivo_train)
train.head()

ruta_archivo_test = '/Users/ximenapaz/github/ITAM/Proyecto_Casas/data/test.csv'
test = cargar_datos(ruta_archivo_test)
test.head()

ruta_archivo_sample = '/Users/ximenapaz/github/ITAM/Proyecto_Casas/data/sample_submission.csv'
sample = cargar_datos(ruta_archivo_sample)
sample.head()

train.describe()

#Se buscarán los missing values en nuestra train data
umbral = 1
number_of_cols = encontrar_valores_faltantes(train, umbral)
print(number_of_cols)

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





