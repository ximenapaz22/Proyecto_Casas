# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)

#Finalmente se calculan las predicciones para el archivo de test y se dan los
# y se crea un csv con los resultados
test_predictions = modelo.predict(test.drop('Id', axis=1).select_dtypes(exclude=['object']))

predictions = pd.DataFrame({'Id': test['Id'], 'SalePrice': test_predictions})

predictions.to_csv('predictions.csv', index=False)
predictions
