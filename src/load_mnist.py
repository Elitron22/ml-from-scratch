"""Descarga MNIST y prepara arrays para una red escrita con NumPy."""

import numpy as np
from sklearn.datasets import fetch_openml


def load_mnist(validation_size=5_000, seed=42):
    """Devuelve (entrenamiento, validación, prueba), cada uno como (X, y).

    Cada fila de X es una imagen de 784 píxeles normalizados a [0, 1].
    Cada elemento de y es un dígito entero de 0 a 9.
    """
    X, y = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False
    )
    X = X.astype(np.float32) / 255.0
    y = y.astype(np.int64)

    # OpenML conserva la división tradicional: 60 000 imágenes y 10 000 de prueba.
    X_train_all, X_test = X[:60_000], X[60_000:]
    y_train_all, y_test = y[:60_000], y[60_000:]

    # Separamos validación solo del conjunto de entrenamiento.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X_train_all))
    validation_indices = order[:validation_size]
    train_indices = order[validation_size:]

    train = X_train_all[train_indices], y_train_all[train_indices]
    validation = X_train_all[validation_indices], y_train_all[validation_indices]
    test = X_test, y_test
    return train, validation, test


if __name__ == "__main__":
    train, validation, test = load_mnist()
    for name, (X, y) in (
        ("Entrenamiento", train),
        ("Validación", validation),
        ("Prueba", test),
    ):
        print(f"{name}: X={X.shape}, y={y.shape}")
    print(f"Primera etiqueta de entrenamiento: {train[1][0]}")
