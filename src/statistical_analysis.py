import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Módulo statistical_analysis.py:

Compara la variabilidad (desviación estándar y coeficiente de variación)
de las estadísticas topológicas -- persistencia total, entropía
persistente, número de pares y vida máxima -- bajo las distintas
configuraciones consideradas en el proyecto: normalización y
perturbaciones (gaussiana y acotada, a distintas magnitudes).
"""

ESTADISTICAS = ["n_pares", "persistencia_total", "vida_maxima", "entropia"]


def variability_by_normalization(path="results_normalization_statistics.csv"):
    """
    Calcula media, desviación estándar y coeficiente de variación (CV)
    de cada estadística topológica, para cada método de normalización.
    """
    df = pd.read_csv(path)

    filas = []
    for metodo in ["minmax", "zscore", "robust"]:
        for stat in ESTADISTICAS:
            col = f"{stat}_{metodo}"
            media = df[col].mean()
            std = df[col].std()
            cv = std / media if media != 0 else np.nan

            filas.append({
                "configuracion": metodo,
                "estadistica": stat,
                "media": media,
                "std": std,
                "cv": cv
            })

    return pd.DataFrame(filas)


def variability_by_perturbation(path="results_perturbation_stability.csv"):
    """
    Calcula media, desviación estándar y coeficiente de variación (CV)
    de cada estadística topológica, para cada combinación de tipo de
    ruido y magnitud de la perturbación.
    """
    df = pd.read_csv(path)

    filas = []
    for (tipo, magnitud), grupo in df.groupby(["tipo_ruido", "magnitud"]):
        for stat in ESTADISTICAS:
            media = grupo[stat].mean()
            std = grupo[stat].std()
            cv = std / media if media != 0 else np.nan

            filas.append({
                "configuracion": f"{tipo}_{magnitud}",
                "tipo_ruido": tipo,
                "magnitud": magnitud,
                "estadistica": stat,
                "media": media,
                "std": std,
                "cv": cv
            })

    return pd.DataFrame(filas)


def plot_cv_comparison(df_norm, df_pert, nombre_guardar="figures/statistics_cv_comparison.png"):
    """
    Genera un gráfico de barras comparando el coeficiente de variación
    (CV = std / media) de cada estadística topológica, entre normalización
    y perturbación, para visualizar bajo qué configuración las firmas
    topológicas son más o menos variables.
    """
    fig, axes = plt.subplots(1, len(ESTADISTICAS), figsize=(5 * len(ESTADISTICAS), 5))

    for i, stat in enumerate(ESTADISTICAS):
        ax = axes[i]

        cv_norm = df_norm[df_norm["estadistica"] == stat].set_index("configuracion")["cv"]

        cv_pert = (
            df_pert[df_pert["estadistica"] == stat]
            .groupby("tipo_ruido")["cv"]
            .mean()
        )

        etiquetas = list(cv_norm.index) + [f"pert_{t}" for t in cv_pert.index]
        valores = list(cv_norm.values) + list(cv_pert.values)

        ax.bar(etiquetas, valores)
        ax.set_title(stat)
        ax.set_ylabel("Coeficiente de variacion (CV)")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(nombre_guardar)
    plt.close()


def plot_perturbation_variability_trend(df_pert, nombre_guardar="figures/statistics_perturbation_trend.png"):
    """
    Grafica cómo cambia la desviación estándar de cada estadística
    topológica a medida que aumenta la magnitud de la perturbación,
    separado por tipo de ruido.
    """
    fig, axes = plt.subplots(1, len(ESTADISTICAS), figsize=(5 * len(ESTADISTICAS), 5))

    for i, stat in enumerate(ESTADISTICAS):
        ax = axes[i]

        for tipo in df_pert["tipo_ruido"].unique():
            subset = df_pert[
                (df_pert["tipo_ruido"] == tipo) & (df_pert["estadistica"] == stat)
            ].sort_values("magnitud")

            ax.plot(subset["magnitud"], subset["std"], marker="o", label=tipo)

        ax.set_title(stat)
        ax.set_xlabel("Magnitud de la perturbacion")
        ax.set_ylabel("Desviacion estandar")
        ax.legend()

    plt.tight_layout()
    plt.savefig(nombre_guardar)
    plt.close()


if __name__ == "__main__":

    df_norm = variability_by_normalization("results_normalization_statistics.csv")
    df_pert = variability_by_perturbation("results_perturbation_stability.csv")

    print("Variabilidad por normalizacion:\n")
    print(df_norm.to_string(index=False))

    print("\nVariabilidad por perturbacion:\n")
    print(df_pert.to_string(index=False))

    df_norm.to_csv("results_variability_normalization.csv", index=False)
    df_pert.to_csv("results_variability_perturbation.csv", index=False)

    plot_cv_comparison(df_norm, df_pert)
    plot_perturbation_variability_trend(df_pert)

    print("\nResultados guardados correctamente.")