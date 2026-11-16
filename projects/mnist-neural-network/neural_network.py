"""A configurable feedforward neural network implemented with NumPy."""

import numpy as np


class NeuralNetwork:
    def __init__(self, input_size=784, hidden_layers=(128,), output_size=10, seed=42):
        """Each value in hidden_layers is the width of one hidden ReLU layer."""
        hidden_layers = tuple(hidden_layers)
        layer_sizes = (input_size, *hidden_layers, output_size)
        if any(
            not isinstance(size, (int, np.integer)) or size <= 0
            for size in layer_sizes
        ):
            raise ValueError("Layer sizes must be positive integers")

        self.rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        for layer in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[layer:layer + 2]
            scale = np.sqrt((2 if layer < len(hidden_layers) else 1) / fan_in)
            self.weights.append(self.rng.normal(0, scale, (fan_in, fan_out)))
            self.biases.append(np.zeros(fan_out))

    def _forward(self, X):
        activations = [X]
        preactivations = []
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            z = activations[-1] @ W + b
            preactivations.append(z)
            activations.append(np.maximum(z, 0))

        logits = activations[-1] @ self.weights[-1] + self.biases[-1]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        return activations, preactivations, probs

    def train_batch(self, X, y, learning_rate):
        """Compute gradients and update parameters using one batch."""
        activations, preactivations, probs = self._forward(X)
        n = len(y)
        loss = -np.log(np.clip(probs[np.arange(n), y], 1e-12, 1)).mean()

        delta = probs.copy()
        delta[np.arange(n), y] -= 1
        delta /= n

        weight_gradients = [None] * len(self.weights)
        bias_gradients = [None] * len(self.biases)
        for layer in range(len(self.weights) - 1, -1, -1):
            weight_gradients[layer] = activations[layer].T @ delta
            bias_gradients[layer] = delta.sum(axis=0)
            if layer > 0:
                delta = (delta @ self.weights[layer].T) * (preactivations[layer - 1] > 0)

        for layer in range(len(self.weights)):
            self.weights[layer] -= learning_rate * weight_gradients[layer]
            self.biases[layer] -= learning_rate * bias_gradients[layer]
        return loss

    def train(self, X, y, epochs=5, batch_size=128, learning_rate=0.1,
              X_validation=None, y_validation=None, augment_batch=None):
        """Train on shuffled batches and print the loss after each epoch."""
        X = np.asarray(X, dtype=np.float32).reshape(-1, self.weights[0].shape[0])
        y = np.asarray(y, dtype=np.int64).reshape(-1)

        for epoch in range(epochs):
            indices = self.rng.permutation(len(X))
            total_loss = 0.0
            for start in range(0, len(X), batch_size):
                batch = indices[start:start + batch_size]
                batch_X = X[batch]
                if augment_batch is not None:
                    batch_X = augment_batch(batch_X, self.rng)
                loss = self.train_batch(batch_X, y[batch], learning_rate)
                total_loss += loss * len(batch)

            message = f"Epoch {epoch + 1}: loss = {total_loss / len(X):.4f}"
            if X_validation is not None and y_validation is not None:
                message += f", validation accuracy = {self.accuracy(X_validation, y_validation):.2%}"
            print(message)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32).reshape(-1, self.weights[0].shape[0])
        return self._forward(X)[2].argmax(axis=1)

    def accuracy(self, X, y):
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        return np.mean(self.predict(X) == y)
