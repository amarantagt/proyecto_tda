from sklearn.preprocessing import (MinMaxScaler,StandardScaler,RobustScaler)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

# Definimos la fórmula para las curvas de andrews como se menciona en el enunciado
def formula_curva_andrews(x, t):
    x = np.array(x)
    p = len(x)

    # Término base
    resultado = x[0] / np.sqrt(2)

    # Componentes armónicos alternados
    for i in range(1, p):
        frecuencia = (i // 2) + 1
        if i % 2 == 1:
            resultado += x[i] * np.sin(frecuencia * t)
        else:
            resultado += x[i] * np.cos(frecuencia * t)

    return resultado

# Función para graficar las curvas de andrews
def graficar_andrews(X_norm, y_clase, titulo, nombre_guardar):

    fig, ax = plt.subplots(figsize=(10, 5))
    
    y_valores = np.array(y_clase)
    
    # Muestra fija de 50 para que el gráfico sea legible
    if len(X_norm) > 50:
        np.random.seed(42)
        indices = np.random.choice(len(X_norm), size=50, replace=False)
        X_final = X_norm[indices]
        y_final = y_valores[indices]
    else:
        X_final = X_norm
        y_final = y_valores
        
    t = np.linspace(-np.pi, np.pi, 250)
    
    clases_unicas = np.unique(y_final)
    mapa_colores = plt.cm.get_cmap('plasma', len(clases_unicas))
    color_dict = {clase: mapa_colores(i) for i, clase in enumerate(clases_unicas)}
    
    # Construcción matemática de las curvas de Andrews
    for fila, clase in zip(X_final, y_final):
        resultado = fila[0] / np.sqrt(2)
        for i in range(1, len(fila)):
            frecuencia = (i // 2) + 1
            if i % 2 == 1:
                resultado += fila[i] * np.sin(frecuencia * t)
            else:
                resultado += fila[i] * np.cos(frecuencia * t)
                
        ax.plot(t, resultado, color=color_dict[clase], alpha=0.6)
        
    # Elementos estéticos del gráfico
    ax.set_title(titulo, fontsize=12, fontweight='bold', pad=10)
    
    lineas_leyenda = [Line2D([0], [0], color=color_dict[c], lw=2, label=f'Calidad {c}') for c in clases_unicas]
    ax.legend(handles=lineas_leyenda, loc='upper right', title="Calidad")
    ax.set_xlabel('t')
    ax.set_ylabel('$f_x(t)$')
    ax.set_xlim([-np.pi, np.pi])
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Guardar automáticamente la imagen en formato PNG
    plt.savefig(nombre_guardar, dpi=300, bbox_inches='tight')
    print(f"[OK] Guardado en: {nombre_guardar}")
    
    # Cerramos la figura para evitar que se abra en pantalla
    plt.close(fig)

if __name__ == "__main__":
    
    # =====================================================================
    # BLOQUE 1: VINO TINTO 
    # =====================================================================
    df_red = pd.read_csv("../data/winequality-red.csv", sep=";")
    y_red = df_red["quality"]
    X_red = df_red.drop(columns=["quality"]).values
    
    # Grafico 1: Tinto + MinMax (Se guarda dentro de la carpeta figures/)
    X_red_minmax = normalizacion(X_red, metodo="minmax")
    graficar_andrews(X_red_minmax, y_red, 
                     titulo="Vino Tinto - Normalización Min-Max", 
                     nombre_guardar="../figures/tinto_minmax.png")
    
    # Grafico 2: Tinto + Z-Score
    X_red_zscore = normalizacion(X_red, metodo="zscore")
    graficar_andrews(X_red_zscore, y_red, 
                     titulo="Vino Tinto - Normalización Z-Score", 
                     nombre_guardar="../figures/tinto_zscore.png")
                     
    # Grafico 3: Tinto + Robusto
    X_red_robust = normalizacion(X_red, metodo="robust")
    graficar_andrews(X_red_robust, y_red, 
                     titulo="Vino Tinto - Normalización Robusta (Mediana/IQR)", 
                     nombre_guardar="../figures/tinto_robust.png")


    # =====================================================================
    # BLOQUE 2: VINO BLANCO 
    # =====================================================================
    df_white = pd.read_csv("../data/winequality-white.csv", sep=";")
    y_white = df_white["quality"]
    X_white = df_white.drop(columns=["quality"]).values
    
    # Grafico 4: Blanco + MinMax
    X_white_minmax = normalizacion(X_white, metodo="minmax")
    graficar_andrews(X_white_minmax, y_white, 
                     titulo="Vino Blanco - Normalización Min-Max", 
                     nombre_guardar="../figures/blanco_minmax.png")
    
    # Grafico 5: Blanco + Z-Score
    X_white_zscore = normalizacion(X_white, metodo="zscore")
    graficar_andrews(X_white_zscore, y_white, 
                     titulo="Vino Blanco - Normalización Z-Score", 
                     nombre_guardar="../figures/blanco_zscore.png")
                     
    # Grafico 6: Blanco + Robusto
    X_white_robust = normalizacion(X_white, metodo="robust")
    graficar_andrews(X_white_robust, y_white, 
                     titulo="Vino Blanco - Normalización Robusta (Mediana/IQR)", 
                     nombre_guardar="../figures/blanco_robust.png")