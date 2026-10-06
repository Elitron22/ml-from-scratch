"""Download MNIST and prepare arrays for a network written with NumPy."""

import numpy as np
from sklearn.datasets import fetch_openml


def load_mnist(validation_size=5_000, seed=42):
    """Return (train, validation, test), each as an (X, y) pair.

    Each row of X is an image with 784 pixels normalized to [0, 1].
    Each element of y is an integer digit from 0 to 9.
    """
    X, y = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False
    )
    X = X.astype(np.float32) / 255.0
    y = y.astype(np.int64)

    # OpenML preserves the traditional split: 60,000 training and 10,000 test images.
    X_train_all, X_test = X[:60_000], X[60_000:]
    y_train_all, y_test = y[:60_000], y[60_000:]

    # Take the validation set only from the training partition.
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
        ("Training", train),
        ("Validation", validation),
        ("Test", test),
    ):
        print(f"{name}: X={X.shape}, y={y.shape}")
    print(f"First training label: {train[1][0]}")
