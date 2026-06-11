"""Este archivo tiene por único objetivo entender de manera superficial los datasets
antes de realizar el proyecto para el que será utilizado"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Primero realizaremos unas funciones que nos ayudarán con la visualización de los datos

def distribucion_calidad(df, nombre):
    print("\nDistribución de quality:")
    print(df["quality"].value_counts().sort_index())
    plt.figure(figsize=(6, 4))
    df["quality"].value_counts().sort_index().plot(
        kind="bar"
    )
    plt.title(f"Distribución de calidad vino {nombre}")
    plt.xlabel("Quality")
    plt.ylabel("Cantidad")

    plt.tight_layout()
    plt.savefig(
        f"figures/distribucion_calidad_vino_{nombre}.png"
    )
    plt.close()

def boxplots(df, nombre):
    columnas = [c for c in df.columns if c != "quality"]
    plt.figure(figsize=(14, 8))
    df[columnas].boxplot(rot=90)
    plt.title(f"Boxplots vino {nombre}")
    plt.tight_layout()
    plt.savefig(
        f"figures/boxplot_vino_{nombre}.png"
    )
    plt.close()


# Ahora realizamos la exploración de los datos
print("Primera exploración de los datasets")
print("Dataset 1: winequality-red.csv")

# Abrimos el dataset de vinos tintos
df_red = pd.read_csv("data/winequality-red.csv", sep=";")

# Revisamos las primeras cinco filas del dataset
head_red = df_red.head()
print(f"\nPrimeros datos de vinos tintos: \n {head_red}")

# Revisamos el tamaño del dataset
tamanho_red = df_red.shape
print(f"\nTamaño del dataset: {tamanho_red}")

# Revisamos si existen valores nulos dentro del dataset
valores_faltantes_1 = df_red.isnull().sum()
print(f"\nValores nulos del dataset: \n {valores_faltantes_1}")

# Ahora mostramos la descripción del dataset
print("\n Descripción del dataset: \n", df_red.describe())


# Ahora vamos con el dataset 2
print("Dataset 2: winequality-white.csv")

# Abrimos el dataset de vinos tintos
df_white = pd.read_csv("data/winequality-white.csv", sep=";")

# Revisamos las primeras cinco filas del dataset
head_white = df_white.head()
print(f"\nPrimeros datos de vinos tintos: \n {head_white}")

# Revisamos el tamaño del dataset
tamanho_white = df_white.shape
print(f"\nTamaño del dataset: {tamanho_white}")

# Revisamos si existen valores nulos dentro del dataset
valores_faltantes_2 = df_white.isnull().sum()
print(f"\nValores nulos del dataset: \n {valores_faltantes_2}")

# Ahora mostramos la descripción del dataset
print("\n Descripción del dataset: \n", df_white.describe())


# Por último generamos algunos gráficos
distribucion_calidad(df_red, "tinto")
boxplots(df_red, "tinto")
distribucion_calidad(df_white, "blanco")
boxplots(df_white, "blanco")