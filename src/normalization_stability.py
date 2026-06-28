import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from andrews import encode, normalizacion
from sublevel_persistence import (
    persistence_diagrams,
    bottleneck,
    wasserstein
)

"""
Módulo normalization_stability.py:

Tiene como objetivo estudiar la estabilidad de las firmas topológicas frente
a distintos métodos de normalización (Min-Max, Z-Score y Robust Scaling),
comparando los diagramas de persistencia mediante las distancias Bottleneck
y Wasserstein.
"""

# Comparación para una sola observación
def normalization_stability_single(
    X_minmax,
    X_zscore,
    X_robust,
    idx,
    N=256
):
    """
    Calcula la estabilidad frente a la normalización para una única observación.

    Para una misma fila del dataset normalizada mediante Min-Max, Z-Score y
    Robust Scaling, genera las curvas de Andrews, calcula sus diagramas de
    persistencia y obtiene las distancias Bottleneck y Wasserstein entre cada
    par de normalizaciones.
    """

    diagramas = {}

    datasets = {
        "minmax": X_minmax,
        "zscore": X_zscore,
        "robust": X_robust
    }

    # Calculamos el diagrama para cada normalización
    for nombre, X in datasets.items():

        curva = encode(X[idx], N=N)
        H0, H1 = persistence_diagrams(curva)

        diagramas[nombre] = H0

    resultados = {}

    pares = [
        ("minmax", "zscore"),
        ("minmax", "robust"),
        ("zscore", "robust")
    ]

    for a, b in pares:

        resultados[f"dB_{a}_{b}"] = bottleneck(
            diagramas[a],
            diagramas[b]
        )

        resultados[f"dW_{a}_{b}"] = wasserstein(
            diagramas[a],
            diagramas[b]
        )

    return resultados


# Experimento sobre muchas observaciones
def normalization_experiment(
    X,
    n_samples=100,
    N=256
):
    """
    Ejecuta el experimento de estabilidad frente a la normalización sobre un
    subconjunto del dataset.

    El dataset se normaliza utilizando Min-Max, Z-Score y Robust Scaling.
    Posteriormente se seleccionan n_samples observaciones aleatorias y, para
    cada una de ellas, se calculan las distancias Bottleneck y Wasserstein
    entre los diagramas de persistencia obtenidos con los distintos métodos
    de normalización.

    Retorna un DataFrame con los resultados del experimento.
    """

    # Se normaliza todo el dataset con cada método
    X_minmax = normalizacion(X, metodo="minmax")
    X_zscore = normalizacion(X, metodo="zscore")
    X_robust = normalizacion(X, metodo="robust")

    np.random.seed(42)

    indices = np.random.choice(
        len(X),
        size=min(n_samples, len(X)),
        replace=False
    )

    resultados = []

    for idx in indices:

        resultado = normalization_stability_single(
            X_minmax,
            X_zscore,
            X_robust,
            idx,
            N
        )

        resultado["sample"] = idx

        resultados.append(resultado)

    return pd.DataFrame(resultados)


# Gráfico Bottleneck
def plot_bottleneck(df):
    """
    Genera un diagrama de cajas con las distancias Bottleneck obtenidas al
    comparar las distintas estrategias de normalización.

    El gráfico se guarda automáticamente en la carpeta figures.
    """

    plt.figure(figsize=(8,5))

    plt.boxplot([
        df["dB_minmax_zscore"],
        df["dB_minmax_robust"],
        df["dB_zscore_robust"]
    ])

    plt.xticks(
        [1,2,3],
        [
            "MinMax-ZScore",
            "MinMax-Robust",
            "ZScore-Robust"
        ]
    )

    plt.ylabel("Distancia Bottleneck")
    plt.title("Estabilidad frente a la Normalización")

    plt.tight_layout()

    plt.savefig(
        "figures/normalization_bottleneck.png"
    )

    plt.close()


# Gráfico Wasserstein
def plot_wasserstein(df):
    """
    Genera un diagrama de cajas con las distancias Wasserstein obtenidas al
    comparar las distintas estrategias de normalización.

    El gráfico se guarda automáticamente en la carpeta figures.
    """

    plt.figure(figsize=(8,5))

    plt.boxplot([
        df["dW_minmax_zscore"],
        df["dW_minmax_robust"],
        df["dW_zscore_robust"]
    ])

    plt.xticks(
        [1,2,3],
        [
            "MinMax-ZScore",
            "MinMax-Robust",
            "ZScore-Robust"
        ]
    )

    plt.ylabel("Distancia Wasserstein")
    plt.title("Estabilidad frente a la Normalización")

    plt.tight_layout()

    plt.savefig(
        "figures/normalization_wasserstein.png"
    )

    plt.close()


# Main
if __name__ == "__main__":

    df = pd.read_csv(
        "data/winequality-red.csv",
        sep=";"
    )

    X = df.drop(columns=["quality"]).values

    resultados = normalization_experiment(
        X,
        n_samples=100,
        N=256
    )

    print(resultados.describe())

    resultados.to_csv(
        "results_normalization_stability.csv",
        index=False
    )

    plot_bottleneck(resultados)
    plot_wasserstein(resultados)

    print("\nResultados guardados correctamente.")
