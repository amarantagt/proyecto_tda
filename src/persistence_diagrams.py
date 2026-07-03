import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gudhi as gd
from andrews import encode, normalizacion
from sublevel_persistence import persistence_diagrams
from perturbation_stability import gaussian_noise, bounded_noise, sigmas_gaussianas,epsilons_acotados
import matplotlib as mpl


mpl.rcParams["text.usetex"] = False
# Creación de carpeta para diagramas de persistencia
OUTPUT_DIR = "persistence_diagrams_figures"

# Funciones auxilires
def diagram_for_plot(diagram, margin=0.1):
    """
    Reemplaza las muertes infinitas por un valor finito para
    poder visualizar correctamente el diagrama.
    """
    if len(diagram) == 0:
        return diagram
    diagram_plot = diagram.copy()
    finite = diagram_plot[np.isfinite(diagram_plot[:, 1])]
    if len(finite) == 0:
        replacement = 1.0
    else:
        replacement = np.max(finite[:, 1]) * (1 + margin)

    diagram_plot[np.isinf(diagram_plot[:, 1]), 1] = replacement

    return diagram_plot


def save_persistence_diagram(diagram, filename, title):
    """
    Guarda un diagrama de persistencia como imagen PNG.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(6, 6))
    gd.plot_persistence_diagram(diagram_for_plot(diagram))

    plt.title(title)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close("all")

    print(f"{filename} listo")


def save_persistence_pair(H0, H1, dataset, metodo):
    """
    Guarda automáticamente H0 y H1.
    """

    save_persistence_diagram(
        H0,
        f"{dataset}_{metodo}_H0.png",
        f"{dataset.capitalize()} - {metodo} - H0"
    )

    save_persistence_diagram(H1,f"{dataset}_{metodo}_H1.png",f"{dataset.capitalize()} - {metodo} - H1")


# Diagramas iniciales del proyecto
def generate_dataset_diagrams():
    """
    Genera los diagramas de persistencia para los dos datasets
    utilizando las tres normalizaciones del proyecto.
    """

    datasets = {
        "red": "data/winequality-red.csv",
        "white": "data/winequality-white.csv"
    }

    metodos = ["minmax", "zscore", "robust"]

    for nombre_dataset, archivo in datasets.items():
        print(f"\nDataset: {nombre_dataset}")
        df = pd.read_csv(archivo, sep=";")
        X = df.drop(columns=["quality"]).values
        for metodo in metodos:
            print(f"  Normalización: {metodo}")
            X_norm = normalizacion(X, metodo)
            curve = encode(X_norm[0], N=256)
            H0, H1 = persistence_diagrams(curve)
            save_persistence_pair(H0, H1, nombre_dataset,metodo)

# Estabilidad frente a la normalización
def generate_normalization_example():
    """
    Genera los diagramas H0 utilizados para ilustrar el
    experimento de estabilidad frente a la normalización.
    """

    df = pd.read_csv(
        "data/winequality-red.csv",
        sep=";"
    )

    X = df.drop(columns=["quality"]).values

    metodos = ["minmax", "zscore", "robust"]

    for metodo in metodos:
        X_norm = normalizacion(X, metodo)
        curve = encode(X_norm[0], N=256)
        H0, _ = persistence_diagrams(curve)
        save_persistence_diagram(H0,f"normalization_{metodo}_H0.png", f"Normalización {metodo.upper()} - H0")

# Ejemplo con ruido
def generate_perturbation_example():
    """
    Genera los diagramas H0 correspondientes al experimento
    de perturbaciones.
    """
    df = pd.read_csv(
        "data/winequality-red.csv",
        sep=";"
    )
    X = df.drop(columns=["quality"]).values
    X = normalizacion(X, "zscore")
    x = X[0]
    curva_original = encode(x, N=256)
    H0_original, _ = persistence_diagrams(curva_original)
    save_persistence_diagram( H0_original,"perturbation_original_H0.png","Curva original - H0")

    # Experimento con ruido gaussiano
    for sigma in sigmas_gaussianas:
        curva = encode( gaussian_noise(x, sigma), N=256)
        H0, _ = persistence_diagrams(curva)
        save_persistence_diagram(H0,f"gaussian_{sigma}_H0.png",f"Ruido gaussiano σ={sigma}")

    # Perturbaciones aleatorias acotadas
    for epsilon in epsilons_acotados:
        curva = encode(bounded_noise(x, epsilon),N=256)
        H0, _ = persistence_diagrams(curva)
        save_persistence_diagram(H0,f"bounded_{epsilon}_H0.png",f"Ruido acotado ε={epsilon}")


if __name__ == "__main__":
    print("Diagramas de persistencia")
    print("\n Diagramas iniciales del proyecto")
    generate_dataset_diagrams()
    print("\nDiagramas para estabilidad de normalización")
    generate_normalization_example()
    print("\nDiagramas para perturbaciones")
    generate_perturbation_example()
    print("diagramas listos")
   