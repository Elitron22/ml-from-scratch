# Machine Learning from Scratch

Implementing machine learning and deep learning algorithms from scratch to understand the math and mechanics behind them.

Este repositorio es un espacio para implementar algoritmos y redes neuronales paso a paso. La implementación de los modelos está pendiente para que puedas construirla y explicar cada línea.

## Estructura

```text
ml-from-scratch/
├── data/                # Archivos locales de datos; no se suben a Git
├── notebooks/           # Experimentos y exploración
├── src/
│   └── load_mnist.py    # Descarga y prepara MNIST
├── tests/               # Pruebas que añadas al implementar modelos
├── .gitignore
├── README.md
└── requirements.txt
```

## Preparación en Windows (PowerShell)

Instala Python 3.12 y Git. Después, en cada equipo:

```powershell
git clone https://github.com/Elitron22/ml-from-scratch.git
cd ml-from-scratch
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python src\load_mnist.py
```

En macOS o Linux, crea el entorno con `python3 -m venv .venv` y usa `.venv/bin/python` en los comandos siguientes. Cada equipo tiene su propio `.venv`; no se comparte mediante Git.

La primera ejecución descarga MNIST. El conjunto se guarda en la caché local de scikit-learn. El script muestra las dimensiones de entrenamiento, validación y prueba; no entrena ningún modelo.

## Próximos pasos sugeridos

1. Inspeccionar una imagen y su etiqueta usando los arrays devueltos por `load_mnist()`.
2. Implementar una capa `784 → 10` y comprender sus dimensiones.
3. Añadir softmax, una función de pérdida y descenso de gradiente.
4. Añadir capas ocultas y retropropagación.

El [artículo de referencia](https://medium.com/@pankajgoyal4152/understanding-neural-networks-by-building-one-from-scratch-a-beginners-journey-3a11617313a4) utiliza una arquitectura `784 → 64 → 32 → 10` para clasificar dígitos.

## Trabajo entre equipos

Antes de empezar: `git pull`. Al terminar: `git add .`, `git commit -m "Describe el cambio"` y `git push`. Las versiones de las bibliotecas están fijadas en `requirements.txt` para repetir la instalación.
