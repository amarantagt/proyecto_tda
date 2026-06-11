# Proyecto semestral CC5514-1 Análisis Topológico de Datos: TDA Aplicado a las Curvas de Andrews
## Filtración de Subniveles y Firmas Topológicas de Curvas Individuales
#### Integrantes:
* Amaranta Godoy
* Felipe Quilodrán
* Valeria Serrano

## Descripción del proyecto

## Estructura del proyecto

```
proyecto/
├── src/
|   ├── exploracion_datos.py    → implementación de una exploración inicial de los datos.
│   ├── andrews.py              → implementación de la normalización y generación de Curvas de Andrews.
|   ├──stability.py             → implementación de experimentos.
│   └── sublevel_persistence.py → implementación de cálculo de diagramas de persistencia.

├── data/
│   ├── winequality-red.csv      → dataset 1 (Vinos tintos)
│   └── winequality-white.csv    → dataset 2 (Vinos blancos)
├── figures/
|   ├── boxplot_vino_blanco.png                 → boxplot dataset 2
|   ├── boxplot_vino_tinto.png                  → boxplot dataset 1
|   ├── distribucion_calidad_vino_blanco.png    → gráfico de distribución de calidad del dataset 2
|   └── distribucion_calidad_vino_tinto.png     → gráfico de distribución de calidad del dataset 1
├── resultado_exploracion.txt   → resultados de la ejecución de exploracion_datos.py
├── .gitignore
└── README.md
```

## Compilación del proyecto

## Resultados generados