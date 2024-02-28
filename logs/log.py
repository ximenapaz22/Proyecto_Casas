import logging
from datetime import datetime
from prep import train, test


# Configurar el logger
logging.basicConfig(level=logging.DEBUG)

# Obtener el timestamp actual
now = datetime.now()
date_time = now.strftime("%Y%m%d-%H%M%S")

# Crear los nombres de los archivos de log
log_file_name = f"logs/{date_time}.log"

# Configurar el logger para que escriba en un archivo
logging.basicConfig(filename=log_file_name, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

# Registrar los pasos de la ejecución del script
logging.info("Iniciando ejecución del script")

# Pasos en la ejecución del script
try:
    # Paso 1: Cargar datos
    logging.info("Paso 1: Cargando datos")
    # Código para cargar datos
    
    # Paso 2: Limpiar datos
    logging.info("Paso 2: Limpiando datos")
    # Código para limpiar datos
    
    # Paso 3: Crear features
    logging.info("Paso 3: Creando features")
    # Código para crear features
    
    # Otros pasos...

except Exception as e:
    # Manejar excepciones
    logging.error(f"Error durante la ejecución del paso: {e}", exc_info=True)

# Obtener información sobre los datasets
num_rows_train = len(train)
num_cols_train = len(train.columns)
num_rows_test = len(test)
num_cols_test = len(test.columns)

# Registrar información sobre los datasets
logging.debug(f"Número de filas en el conjunto de entrenamiento: {num_rows_train}")
logging.debug(f"Número de columnas en el conjunto de entrenamiento: {num_cols_train}")
logging.debug(f"Número de filas en el conjunto de prueba: {num_rows_test}")
logging.debug(f"Número de columnas en el conjunto de prueba: {num_cols_test}")

# Otros registros de información específica del script...

# Cerrar el logger al final de la ejecución
logging.shutdown()

