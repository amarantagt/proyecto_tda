# Proyecto semestral CC5514-1 Análisis Topológico de Datos: TDA Aplicado a las Curvas de Andrews
## Robustez y Estabilidad de Firmas Topológicas de Curvas de Andrews
#### Integrantes:
* Amaranta Godoy
* Felipe Quilodrán
* Valeria Serrano

## Descripción del proyecto

## Estructura del proyecto

```
PROYECTO_TDA/
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
|   ├── distribucion_calidad_vino_tinto.png     → gráfico de distribución de calidad del dataset 1
|   ├── blanco_minmax.png
|   ├── blanco_robust.png
|   ├── blanco_zscore.png
|   ├── tinto_minmax.png
|   ├── tinto_robust.png
|   └── tinto_zscore.png
├── resultado_exploracion.txt   → resultados de la ejecución de exploracion_datos.py
├── .gitignore
└── README.md
```

## Ejecución del proyecto
Para realizar la exploración de datos y guardarla en resultado_exploracion.txt

```bash
python3 src/exploracion_datos.py > resultado_exploracion.txt
```

Para revisar los resultados de las pruebas de andrews.py 
```bash
python3 src/andrews.py 
```
## Resultados generados
