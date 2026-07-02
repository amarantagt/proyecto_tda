import gudhi as gd
from gudhi import bottleneck_distance
from gudhi.wasserstein import wasserstein_distance
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from andrews import encode, normalizacion

""" Módulo sublevel_persistence.py: 
Tiene como objetivo implementar el cálculo los diagramas de persistencia de un set de 
datos y sus estadísticas topológicas
"""
# Construimos funciones auxiliares para manejar correctamente el
# lifetime infinito

 # Función finite_diagram, devuelve los intervalos con muerte finita
def finite_diagram(diagram):
    if len(diagram) == 0 :
        return diagram
    
    return diagram[np.isfinite(diagram[:, 1])]

# Función diagram_for_plot, reemplaza muertes infinias por un valor más grande
def diagram_for_plot(diagram, margin = 0.1):
    if len(diagram) == 0:
        return diagram
    
    # Hacemos una copia del diagrama
    diagram_plot = diagram.copy()
    finite = diagram_plot[np.isfinite(diagram_plot[:, 1])]

    # Si no hay intervalos con lifetimes con muertes finitas
    # se reemplaza la muerte infinita con 1.0
    if len(finite) == 0:
        replacement = 1.0
    # Si existen intervalos con muertes finitas, se reemplazan las muertes infinitas
    # con el número máximo y un margen.
    else:
        replacement = np.max(finite[:, 1] * (1 + margin))
    
    diagram_plot[np.isinf(diagram_plot[:, 1]), 1] = replacement
    return diagram_plot


# Función persistence_diagrams, calcula el diagrama de persistencia de una curva
def persistence_diagrams(curve):
    curve = np.asarray(curve) # Nos aseguramos que curve sea un arreglo

    # Construimos el cubical complex PERIÓDICO sobre S^1
    # (la curva de Andrews es periódica: t=-pi y t=pi son el mismo punto)
    cc = gd.PeriodicCubicalComplex(
        dimensions=[len(curve)],
        top_dimensional_cells=curve,
        periodic_dimensions=[True]
    )
    cc.compute_persistence()

    # Extraemos el H0 y el H1
    H0 = np.array(cc.persistence_intervals_in_dimension(0))
    H1 = np.array(cc.persistence_intervals_in_dimension(1))

    return H0, H1

# Ahora calculamos la persistencia total
def total_persistence(diagram):
    diagram = finite_diagram(diagram) # Ahora en el total solo consideraremos los que tienen muerte finita
    if len(diagram) == 0:
        return 0.0
    lifetimes = diagram[:,1] - diagram[:,0] # Hacemos la resta death - birth
    return np.sum(lifetimes) # Retornamos las sumas de todas las vidas.

#  Calculamos la persistencia máxima encontrada.
def max_lifetime(diagram):
    diagram = finite_diagram(diagram)
    if len(diagram) == 0:
        return 0.0
    lifetimes = diagram[:, 1] - diagram[:, 0]
    return np.max(lifetimes)

# Función count_pairs, cuenta cuántos intervalos sobreviven más que un umbral tau
def count_pairs(diagram, tau = 0.1):
    diagram = finite_diagram(diagram)
    if len(diagram) == 0:
        return 0  
    lifetimes = diagram[:, 1] - diagram[:, 0]
    return int(np.sum(lifetimes > tau))

# Calculamos la entropía persistente
def persistent_entropy(diagram):
    diagram = finite_diagram(diagram)
    if len(diagram) == 0:
        return 0.0
    
    lifetimes = diagram[:, 1] - diagram[:, 0]
    lifetimes = lifetimes[lifetimes > 0]

    if len(lifetimes) == 0:
        return 0.0
    
    p = lifetimes / np.sum(lifetimes)

    return -np.sum(p * np.log(p))

# Funciones para implementar distancias entre diagramas

# Distancia bottleneck
def bottleneck(H_a, H_b):
    H_a = finite_diagram(H_a)
    H_b = finite_diagram(H_b)

    if len(H_a) == 0 and len(H_b) == 0:
        return 0.0
    
    return bottleneck_distance(H_a, H_b)

# Distancia wasserstein
def wasserstein(H_a, H_b):
    H_a = finite_diagram(H_a)
    H_b = finite_diagram(H_b)

    if len(H_a) == 0 and len(H_b) == 0:
        return 0.0
    
    return wasserstein_distance(H_a, H_b)



# La función topological_isgnature utiliza reune a las funciones anteriores
# para calcular los diagramas de persistencia y estadísticas de una curva.
def topological_signature(curve, tau = 0.1):
    H0, H1 = persistence_diagrams(curve) # Calculamos el diagrama de persistencia de la curva.

    # Ahora guardamos todos los valores en un diccionario
    return {
        "H0": H0,
        "H1" : H1,

        "pairs_H0" : count_pairs(H0, tau),
        "pairs_H1" : count_pairs(H1, tau),

        "total_persistence_H0" : total_persistence(H0),
        "total_persistence_H1" : total_persistence(H1),

        "max_H0" : max_lifetime(H0),
        "max_H1" : max_lifetime(H1),

        "entropy_H0" : persistent_entropy(H0),
        "entropy_H1" : persistent_entropy(H1)
    }



if __name__ == "__main__":
    df = pd.read_csv(
        "data/winequality-red.csv",
        sep=";"
    )

    X = df.drop(columns=["quality"]).values
    X = normalizacion(X, metodo="zscore")
    curve = encode(X[0], N=256)
    resultado = topological_signature(curve)

    print("\nFirma topológica:\n")
    print(resultado)

    H0, H1 = persistence_diagrams(curve)

    print("\nH0:")
    print(H0)

    print("\nH1:")
    print(H1)

    # =====================================================================
    # VERIFICACIÓN: ¿el complejo periódico está funcionando correctamente?
    # =====================================================================
    print("\n" + "=" * 60)
    print("VERIFICACIÓN DEL COMPLEJO PERIÓDICO")
    print("=" * 60)

    # --- Test 1: comparar contra el complejo NO periódico anterior ---
    cc_lineal = gd.CubicalComplex(
        dimensions=[len(curve)],
        top_dimensional_cells=curve
    )
    cc_lineal.compute_persistence()
    H0_lineal = np.array(cc_lineal.persistence_intervals_in_dimension(0))
    H1_lineal = np.array(cc_lineal.persistence_intervals_in_dimension(1))

    print(f"\n[Test 1] Complejo lineal (antiguo) vs periódico (nuevo)")
    print(f"  H0 lineal:   {len(H0_lineal)} pares")
    print(f"  H0 periódico:{len(H0)} pares")
    print(f"  H1 lineal:   {len(H1_lineal)} clases")
    print(f"  H1 periódico:{len(H1)} clases")
    if len(H0) != len(H0_lineal) or not np.array_equal(
        np.sort(H0_lineal, axis=0), np.sort(H0, axis=0)
    ):
        print("  -> OK: los diagramas son distintos, el complejo periódico "
              "SÍ está tratando el ciclo de forma diferente al lineal.")
    else:
        print("  -> ADVERTENCIA: los diagramas son iguales, revisar la "
              "implementación.")

    # --- Test 2: H1 debe tener EXACTAMENTE una clase (el ciclo global S^1) ---
    print(f"\n[Test 2] H1 debe tener 1 sola clase (el ciclo de S^1)")
    print(f"  Número de clases en H1: {len(H1)}")

    # --- Test 3: curva sintética simple, fácil de razonar a mano ---
    print(f"\n[Test 3] Curva sintética simple: seno de un periodo")
    t_test = np.linspace(-np.pi, np.pi, 100, endpoint=False)
    curva_test = np.sin(t_test)
    H0_test, H1_test = persistence_diagrams(curva_test)
    print(f"  H0: {len(H0_test)} pares (peaks/valles locales)")
    print(f"  H1: {len(H1_test)} clase(s)")
    print("  Para sin(t) en un periodo completo se esperan 2 puntos "
          "críticos (1 máximo, 1 mínimo) reflejados en H0, "
          "y 1 sola clase en H1.")

    print("\n" + "=" * 60)