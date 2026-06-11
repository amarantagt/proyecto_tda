from sklearn.preprocessing import (MinMaxScaler,StandardScaler,RobustScaler)
import pandas as pd

# Primero realizamos una función para normalizar los datos de un dataset
def normalizacion(X, metodo = "zscore"):
    if metodo == "minmax":
        return MinMaxScaler().fit_transform(X)
    elif metodo == "zscore":
        return StandardScaler().fit_transform(X)
    elif metodo == "robust":
        return RobustScaler().fit_transform(X)
    else:
        print("Se recibió un método desconocido")


# Probamos nuestra función solo con el valor de la calidad del vino
df_red = pd.read_csv("data/winequality-red.csv", sep=";")
X_red = df_red.drop(columns=["quality"]).values
X_red_norm = normalizacion(X_red,metodo="robust")
print(X_red_norm[:5])