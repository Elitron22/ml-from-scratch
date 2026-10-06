# MNIST Neural Network from Scratch

Implementing machine learning and deep learning algorithms from scratch to understand the math and mechanics behind them.

This project is a workspace for implementing a neural network step by step. The model is intentionally left for you to build and explain line by line.

## Structure

```text
projects/mnist-neural-network/
├── data/                # Local data files; not committed to Git
├── notebooks/           # Experiments and exploration
├── src/
│   └── load_mnist.py    # Download and prepare MNIST
├── tests/               # Tests to add as you implement models
├── README.md
└── requirements.txt
```

## Setup on Windows (PowerShell)

Install Python 3.12 and Git. Then, on each computer:

```powershell
git clone https://github.com/Elitron22/ml-from-scratch.git
cd ml-from-scratch\projects\mnist-neural-network
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python src\load_mnist.py
```

On macOS or Linux, create the environment with `python3 -m venv .venv` and use `.venv/bin/python` for the remaining commands. Each computer has its own `.venv`; it is not shared through Git.

The first run downloads MNIST. The dataset is stored in scikit-learn's local cache. The script prints the training, validation, and test shapes; it does not train a model.

## Suggested next steps

1. Inspect an image and its label using the arrays returned by `load_mnist()`.
2. Implement a `784 → 10` layer and understand its dimensions.
3. Add softmax, a loss function, and gradient descent.
4. Add hidden layers and backpropagation.

The [reference article](https://medium.com/@pankajgoyal4152/understanding-neural-networks-by-building-one-from-scratch-a-beginners-journey-3a11617313a4) uses a `784 → 64 → 32 → 10` architecture to classify digits.

## Working across computers

From the repository root, run `git pull` before starting work. When finished, run `git add .`, `git commit -m "Describe your change"`, and `git push`. Library versions are pinned in this project's `requirements.txt` so the environment can be recreated.
