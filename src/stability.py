import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from andrews import encode, normalizacion
from sublevel_persistence import persistence_diagrams, bottleneck, wasserstein

""""
Primera implementación de stability.py, se realiza el paso de estbailidad al muestreo
con distintas resoluciones de discretización
"""

# Lista con el tamaño de las resoluciones de discretización
resoluciones = [64, 128, 256, 512]

# Experimento para una sola observaciun
def sampling_stability_single(x):
    resultados = {}
    diagramas_H0 = {}

    for n in resoluciones:
        curva = encode(x, N=n)
        H0, H1 = persistence_diagrams(curva)
        diagramas_H0[n] = H0

    pares = [
        (64, 128),
        (128, 256),
        (256, 512)
    ]

    for n1, n2 in pares:
        # distancia bottleneck
        dB = bottleneck(diagramas_H0[n1], diagramas_H0[n2])
        # distancia wasserstein
        dW = wasserstein(diagramas_H0[n1], diagramas_H0[n2])

        resultados[f"dB_{n1}_{n2}"] = dB
        resultados[f"dW_{n1}_{n2}"] = dW

    return resultados

# Experimento para muchas observaciones
def sampling_experiment(X, n_samples = 100):
    np.random.seed(42)
    
    indices = np.random.choice(len(X), size = min(n_samples, len(X)), replace = False)
    resultados = []

    for idx in indices :
        fila = X[idx]
        resultado = sampling_stability_single(fila)
        resultado["sample"] = idx
        resultados.append(resultado)
    
    return pd.DataFrame(resultados)

# Función para graficar bottlenck
def plot_bottleneck(df):
    plt.figure(figsize=(8,5))

    plt.boxplot(
        [
            df["dB_64_128"],
            df["dB_128_256"],
            df["dB_256_512"]
        ]
    )

    plt.xticks(
        [1,2,3],
        [
            "64-128",
            "128-256",
            "256-512"
        ]
    )

    plt.ylabel("Distancia Bottleneck")
    plt.title("Estabilidad de Muestreo")
    plt.tight_layout()
    plt.savefig("figures/stability_bottleneck.png")
    plt.close()

# Función para graficar wasserstein
def plot_wasserstein(df):
    plt.figure(figsize=(8, 5))

    plt.boxplot(
        [
            df["dW_64_128"],
            df["dW_128_256"],
            df["dW_256_512"]
        ]
    )

    plt.xticks(
        [1,2,3],
        [
            "64-128",
            "128-256",
            "256-512"
        ]
    )

    plt.ylabel("Distancia Wasserstein")
    plt.title("Estabilidad de Muestreo")
    plt.tight_layout()
    plt.savefig("figures/stability_wasserstein.png")
    plt.close()


# Función main para probar el módulo stability.py
if __name__ == "__main__":

    df = pd.read_csv(
        "data/winequality-red.csv",
        sep=";"
    )

    X = df.drop(columns=["quality"]).values
    X = normalizacion(X,metodo="zscore")
    resultados = sampling_experiment(X,n_samples=100)
    print(resultados.describe())

    resultados.to_csv(
        "results_sampling_stability.csv",
        index=False
    )

    plot_bottleneck(resultados)
    plot_wasserstein(resultados)
