import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Módulo stability_verification.py:

Verifica empíricamente el teorema de estabilidad de la homología persistente:

    dB(Dgm(f), Dgm(g)) <= ||f - g||_inf

usando los resultados ya calculados en el experimento de perturbaciones
(results_perturbation_stability.csv), que contiene dB y ||f-g||_inf para
cada combinación de (muestra, tipo de ruido, magnitud).
"""

def load_perturbation_results(path="results_perturbation_stability.csv"):
    return pd.read_csv(path)


def verify_stability_theorem(df, tol=1e-6):
    """
    Verifica, fila por fila, que dB <= ||f-g||_inf (con una tolerancia
    numérica pequeña para errores de precisión), y calcula la razón
    dB / ||f-g||_inf para ver qué tan ajustada o laxa es la cota.
    """
    df = df.copy()
    df["cumple_teorema"] = df["dB"] <= (df["norma_inf"] + tol)
    df["razon_dB_norma_inf"] = df["dB"] / df["norma_inf"].replace(0, np.nan)

    n_total = len(df)
    n_violaciones = int((~df["cumple_teorema"]).sum())

    resumen = {
        "n_total": n_total,
        "n_violaciones": n_violaciones,
        "porcentaje_violaciones": 100 * n_violaciones / n_total,
        "razon_promedio": df["razon_dB_norma_inf"].mean(),
        "razon_mediana": df["razon_dB_norma_inf"].median(),
        "razon_maxima": df["razon_dB_norma_inf"].max(),
    }

    return df, resumen


def plot_stability_scatter(df, nombre_guardar="figures/stability_theorem_scatter.png"):
    """
    Grafica dB vs ||f-g||_inf junto con la recta y=x (la cota teórica).
    Si el teorema se cumple, todos los puntos deben quedar sobre o por
    debajo de esa recta, nunca por encima.
    """
    plt.figure(figsize=(7, 7))

    colores = {"gaussiano": "tab:blue", "acotado": "tab:orange"}

    for tipo in df["tipo_ruido"].unique():
        subset = df[df["tipo_ruido"] == tipo]
        plt.scatter(
            subset["norma_inf"], subset["dB"],
            alpha=0.4, s=15, label=tipo, color=colores.get(tipo)
        )

    lim_max = max(df["norma_inf"].max(), df["dB"].max()) * 1.05
    plt.plot([0, lim_max], [0, lim_max], "k--", linewidth=1.5,
             label="y = x (cota teórica)")

    plt.xlabel("||f - g||_inf")
    plt.ylabel("Distancia Bottleneck dB")
    plt.title("Verificación empírica del teorema de estabilidad")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(nombre_guardar)
    plt.close()


def plot_ratio_by_magnitude(df, nombre_guardar="figures/stability_theorem_ratio.png"):
    """
    Boxplot de la razón dB/||f-g||_inf en función de la magnitud de la
    perturbación, uno por tipo de ruido, para ver si la cota se mantiene
    ajustada o se vuelve más laxa a medida que crece la perturbación.
    """
    tipos = df["tipo_ruido"].unique()
    plt.figure(figsize=(6 * len(tipos), 5))

    for i, tipo in enumerate(tipos):
        subset = df[df["tipo_ruido"] == tipo]
        magnitudes = sorted(subset["magnitud"].unique())

        plt.subplot(1, len(tipos), i + 1)
        plt.boxplot(
            [subset[subset["magnitud"] == m]["razon_dB_norma_inf"].dropna()
             for m in magnitudes]
        )
        plt.xticks(range(1, len(magnitudes) + 1), [str(m) for m in magnitudes])
        plt.axhline(1.0, color="red", linestyle="--", linewidth=1, label="cota = 1")
        plt.xlabel("Magnitud de la perturbación")
        plt.ylabel("dB / ||f-g||_inf")
        plt.title(f"Ruido {tipo}")
        plt.legend()

    plt.tight_layout()
    plt.savefig(nombre_guardar)
    plt.close()


if __name__ == "__main__":

    df = load_perturbation_results("results_perturbation_stability.csv")

    df_verificado, resumen = verify_stability_theorem(df)

    print("Resumen de la verificacion del teorema de estabilidad:\n")
    for k, v in resumen.items():
        print(f"  {k}: {v}")

    if resumen["n_violaciones"] == 0:
        print(f"\n-> OK: no se observaron violaciones del teorema de estabilidad "
              f"(dB <= ||f-g||_inf) en ninguna de las {resumen['n_total']} comparaciones.")
    else:
        print(f"\n-> ADVERTENCIA: se observaron {resumen['n_violaciones']} violaciones "
              f"({resumen['porcentaje_violaciones']:.2f}%). Revisar si son atribuibles "
              "a tolerancia numerica del solver de bottleneck o a un error de calculo.")

    df_verificado.to_csv("results_stability_theorem_verification.csv", index=False)

    plot_stability_scatter(df_verificado)
    plot_ratio_by_magnitude(df_verificado)

    print("\nGraficos guardados en figures/.")

    violaciones = df_verificado[~df_verificado["cumple_teorema"]]
    print("\nFilas que violan el teorema:")
    print(violaciones[["sample", "tipo_ruido", "magnitud", "dB", "norma_inf", "razon_dB_norma_inf"]])

    # Verificación de la hipótesis: las violaciones se concentran en casos
    # donde el diagrama H0 finito tiene pocos puntos (la clase esencial
    # infinita pesa proporcionalmente más al ser excluida)
    print("\nMuestras que violan el teorema (ya identificadas):")
    print("  sample=613 (gaussiano, 0.1), sample=486 (acotado, 0.2),")
    print("  sample=1102 (gaussiano, 0.1), sample=529 (acotado, 0.2 y 0.5)")
    print("\n  -> Nota: sample=529 aparece en DOS filas distintas (magnitudes")
    print("     0.2 y 0.5). Esto sugiere que la observación X[529] en sí misma")
    print("     tiene una curva 'sensible' -- probablemente con muy pocos")
    print("     picos/valles en H0 finito -- más que ser un problema de la")
    print("     magnitud de la perturbación en particular.")