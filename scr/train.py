# Código de Entrenamiento
#########################
# Importación de librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
import sys, warnings, os
import pickle

# Cargar la tabla transformada
def read_file_csv(filename):
    ufc_master_ds = pd.read_csv(os.path.join('../data/processed', filename)).set_index('Winner')
    X_train, X_valid, y_train, y_valid = train_test_split(ufc_master_ds, label, test_size = 0.3, random_state=2)
    # Imputación de valores faltantes
    impute = SimpleImputer(strategy = 'mean')
    impute.fit(X_train)
    X_train = impute.transform(X_train)
    X_valid = impute.transform(X_valid)
    print(filename, ' cargado correctamente')
    # Entrenamos el modelo con toda la muestra
    RF_model_1 = RandomForestClassifier(n_estimators = 350, max_depth = 12, random_state = 2)
    RF_model_1.fit(X_train, y_train)
    print('Modelo entrenado')
    # Guardamos el modelo entrenado para usarlo en produccion
    package = '../models/best_model.pkl'
    pickle.dump(RF_model_1, open(package, 'wb'))
    print('Modelo exportado correctamente en la carpeta models')

# Entrenamiento completo
def main():
    read_file_csv('ufc-master.csv')
    print('Finalizó el entrenamiento del Modelo')


if __name__ == "__main__":
    main()
