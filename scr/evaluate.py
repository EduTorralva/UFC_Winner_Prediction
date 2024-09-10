# Código de Evaluación 
#######################
import pandas as pd
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import *
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
import sys, warnings, os


def eval_model(filename):
    ufc_master_ds = pd.read_csv(os.path.join('../data/processed', filename)).set_index('Winner')
    print(filename, ' cargado correctamente')
    # Leemos el modelo entrenado para usarlo
    package = '../models/best_model.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')
    X_train, X_valid, y_train, y_valid = train_test_split(ufc_master_ds, label, test_size = 0.3, random_state=2)
    # Imputación de valores faltantes
    impute = SimpleImputer(strategy = 'mean')
    impute.fit(X_train)
    X_train = impute.transform(X_train)
    X_valid = impute.transform(X_valid)
    y_pred_test=model.predict(X_test)
    # Generamos métricas de diagnóstico
    cm_test = confusion_matrix(y_test,y_pred_test)
    print("Matriz de confusion: ")
    print(cm_test)
    accuracy_test=accuracy_score(y_test,y_pred_test)
    print("Accuracy: ", accuracy_test)
    precision_test=precision_score(y_test,y_pred_test)
    print("Precision: ", precision_test)
    recall_test=recall_score(y_test,y_pred_test)
    print("Recall: ", recall_test)


# Validación desde el inicio
def main():
    df = eval_model('ufc-master.csv')
    print('Finalizó la validación del Modelo')


if __name__ == "__main__":
    main()
    
