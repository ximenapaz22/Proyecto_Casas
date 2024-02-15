import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

train = pd.read_csv('data/train.csv')
train.head()

test = pd.read_csv('data/test.csv')
test.head()

sample = pd.read_csv('data/sample_submission.csv')
sample.head()

train.describe()

#Se buscarán los missing values en nuestra train data
missing_value = train.isnull().sum()
number_of_cols = missing_value[missing_value>1]
print(number_of_cols)

#Grafica de análisis
train.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year of sold')
plt.ylabel('House Price')
plt.title('Precio de las casas vs año')

#Se eliminan las filas que tengan valores nulos en la columna SalePrice
train.dropna(axis=0, subset=['SalePrice'], inplace=True)

#Se calcula la correlación

correlation_matrix = train.corr()

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

