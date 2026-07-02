import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from andrews import encode, normalizacion
from sublevel_persistence import (
    persistence_diagrams, bottleneck, wasserstein,
    count_pairs, total_persistence, max_lifetime, persistent_entropy
)

# Función para agregar ruido gaussiano
def gaussian_noise(x, sigma=0.05):
    ruido = np.random.normal(loc=0, scale=sigma, size=x.shape)
    return x + ruido

# Función para agregar perturbaciones aleatorias acotadas
def bounded_noise(x, epsilon= 0.05):
    ruido = np.random.uniform(low=-epsilon,high=epsilon,size=x.shape)
    return x + ruido

# Magnitudes de perturbación a evaluar
sigmas_gaussianas = [0.01, 0.05, 0.1, 0.2, 0.5]
epsilons_acotados = [0.01, 0.05, 0.1, 0.2, 0.5]


# Experimento para una sola observación: compara el diagrama original
# contra el diagrama de la curva perturbada, para cada magnitud de
# perturbación y cada tipo de ruido.
def perturbation_stability_single(x, N=256):
    """
    Aplica ruido gaussiano y perturbaciones acotadas al vector de atributos
    x, a distintas magnitudes, genera las curvas de Andrews resultantes y
    compara sus diagramas de persistencia (H0) contra el diagrama de la
    curva sin perturbar, usando las distancias Bottleneck y Wasserstein.

    Además calcula la norma infinito ||f - g||_inf entre la curva original
    y cada curva perturbada, para poder verificar empíricamente el teorema
    de estabilidad en el paso 6.
    """

    resultados = []

    # Curva y diagrama originales (sin perturbar)
    curva_original = encode(x, N=N)
    H0_original, _ = persistence_diagrams(curva_original)

    # --- Ruido gaussiano aditivo ---
    for sigma in sigmas_gaussianas:

        x_ruidoso = gaussian_noise(x, sigma=sigma)
        curva_ruidosa = encode(x_ruidoso, N=N)
        H0_ruidoso, _ = persistence_diagrams(curva_ruidosa)

        dB = bottleneck(H0_original, H0_ruidoso)
        dW = wasserstein(H0_original, H0_ruidoso)
        norma_inf = np.max(np.abs(curva_original - curva_ruidosa))

        resultados.append({
            "tipo_ruido": "gaussiano",
            "magnitud": sigma,
            "dB": dB,
            "dW": dW,
            "norma_inf": norma_inf,
            "n_pares": len(H0_ruidoso),
            "n_pares_tau": count_pairs(H0_ruidoso, tau=0.1),
            "persistencia_total": total_persistence(H0_ruidoso),
            "vida_maxima": max_lifetime(H0_ruidoso),
            "entropia": persistent_entropy(H0_ruidoso)
        })

    # --- Perturbaciones aleatorias acotadas ---
    for epsilon in epsilons_acotados:

        x_ruidoso = bounded_noise(x, epsilon=epsilon)
        curva_ruidosa = encode(x_ruidoso, N=N)
        H0_ruidoso, _ = persistence_diagrams(curva_ruidosa)

        dB = bottleneck(H0_original, H0_ruidoso)
        dW = wasserstein(H0_original, H0_ruidoso)
        norma_inf = np.max(np.abs(curva_original - curva_ruidosa))

        resultados.append({
            "tipo_ruido": "acotado",
            "magnitud": epsilon,
            "dB": dB,
            "dW": dW,
            "norma_inf": norma_inf,
            "n_pares": len(H0_ruidoso),
            "n_pares_tau": count_pairs(H0_ruidoso, tau=0.1),
            "persistencia_total": total_persistence(H0_ruidoso),
            "vida_maxima": max_lifetime(H0_ruidoso),
            "entropia": persistent_entropy(H0_ruidoso)
        })

    return resultados


# Experimento sobre muchas observaciones
def perturbation_experiment(X, n_samples=100, N=256):
    """
    Ejecuta el experimento de robustez frente a perturbaciones sobre un
    subconjunto del dataset.

    Para cada observación muestreada, aplica ruido gaussiano y
    perturbaciones acotadas a distintas magnitudes, y calcula las
    distancias Bottleneck y Wasserstein entre el diagrama original y el
    diagrama perturbado, junto con la norma infinito entre ambas curvas.

    Retorna un DataFrame donde cada fila corresponde a una combinación de
    (muestra, tipo de ruido, magnitud).
    """

    np.random.seed(42)

    indices = np.random.choice(
        len(X),
        size=min(n_samples, len(X)),
        replace=False
    )

    filas = []

    for idx in indices:

        resultado_muestra = perturbation_stability_single(X[idx], N=N)

        for fila in resultado_muestra:
            fila["sample"] = idx
            filas.append(fila)

    return pd.DataFrame(filas)


# Gráfico: distancia Bottleneck vs magnitud de la perturbación
def plot_perturbation_bottleneck(df):
    """
    Genera un boxplot de la distancia Bottleneck en función de la magnitud
    de la perturbación, uno por tipo de ruido (gaussiano y acotado).
    El gráfico se guarda en figures/.
    """

    for tipo in df["tipo_ruido"].unique():

        subset = df[df["tipo_ruido"] == tipo]
        magnitudes = sorted(subset["magnitud"].unique())

        plt.figure(figsize=(8, 5))

        plt.boxplot(
            [subset[subset["magnitud"] == m]["dB"] for m in magnitudes]
        )

        plt.xticks(
            range(1, len(magnitudes) + 1),
            [str(m) for m in magnitudes]
        )

        plt.xlabel("Magnitud de la perturbacion")
        plt.ylabel("Distancia Bottleneck")
        plt.title(f"Robustez frente a perturbaciones ({tipo})")
        plt.tight_layout()
        plt.savefig(f"figures/perturbation_bottleneck_{tipo}.png")
        plt.close()


# Gráfico: distancia Wasserstein vs magnitud de la perturbación
def plot_perturbation_wasserstein(df):
    """
    Genera un boxplot de la distancia Wasserstein en función de la
    magnitud de la perturbación, uno por tipo de ruido.
    El gráfico se guarda en figures/.
    """

    for tipo in df["tipo_ruido"].unique():

        subset = df[df["tipo_ruido"] == tipo]
        magnitudes = sorted(subset["magnitud"].unique())

        plt.figure(figsize=(8, 5))

        plt.boxplot(
            [subset[subset["magnitud"] == m]["dW"] for m in magnitudes]
        )

        plt.xticks(
            range(1, len(magnitudes) + 1),
            [str(m) for m in magnitudes]
        )

        plt.xlabel("Magnitud de la perturbacion")
        plt.ylabel("Distancia Wasserstein")
        plt.title(f"Robustez frente a perturbaciones ({tipo})")
        plt.tight_layout()
        plt.savefig(f"figures/perturbation_wasserstein_{tipo}.png")
        plt.close()

if __name__ == "__main__":

    df = pd.read_csv(
        "data/winequality-red.csv",
        sep=";"
    )

    X = df.drop(columns=["quality"]).values
    X = normalizacion(X, metodo="zscore")

    resultados = perturbation_experiment(
        X,
        n_samples=100,
        N=256
    )

    print(resultados.groupby(["tipo_ruido", "magnitud"])[["dB", "dW", "norma_inf"]].describe())

    resultados.to_csv(
        "results_perturbation_stability.csv",
        index=False
    )

    plot_perturbation_bottleneck(resultados)
    plot_perturbation_wasserstein(resultados)

    print("\nResultados guardados correctamente.")