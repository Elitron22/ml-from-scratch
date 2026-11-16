# MNIST Neural Network from Scratch

Build a handwritten digit classifier with NumPy. The notebook loads and explores MNIST, then trains the network implemented in `neural_network.py`.

## Structure

```text
projects/mnist-neural-network/
├── .venv/                   # Local Python environment; ignored by Git
├── mnist_neural_network.ipynb
├── neural_network.py
├── README.md
└── requirements.txt
```

## Setup on Windows (PowerShell)

Install Python 3.12 and Git. Then, on each computer:

```powershell
git clone https://github.com/Elitron22/ml-from-scratch.git
cd ml-from-scratch\projects\mnist-neural-network
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name mnist-neural-network --display-name "Python (MNIST neural network)"
```

Open `mnist_neural_network.ipynb` in VS Code and select **Python (MNIST neural network)** as the notebook kernel. This kernel points to `projects/mnist-neural-network/.venv`, even when VS Code has the repository root open. VS Code's Python and Jupyter extensions provide the notebook interface. Each project creates its own `.venv`; environments are not committed to Git.

On macOS or Linux, create the environment with `python3 -m venv .venv`, install packages with `.venv/bin/python -m pip install -r requirements.txt`, and register the kernel with `.venv/bin/python -m ipykernel install --user --name mnist-neural-network --display-name "Python (MNIST neural network)"`.

## Notebook contents

1. Import NumPy, Matplotlib, and the MNIST downloader.
2. Download MNIST and normalize its pixel values.
3. Split it into training, validation, and test sets.
4. Display nine labeled images in a 3 x 3 grid.
5. Augment training batches with small rotations, shifts, and scaling.
6. Import and train the neural network from `neural_network.py`, then evaluate it.
7. Draw a digit, normalize its size and position, and ask the trained network to classify it.

The first run downloads MNIST into scikit-learn's local cache. Set `hidden_layers` in the final notebook cell to choose the number and width of hidden layers. For example, `(128,)` creates one hidden layer and `(128, 64)` creates two. The cell prints validation and test accuracy.

The [reference article](https://medium.com/@pankajgoyal4152/understanding-neural-networks-by-building-one-from-scratch-a-beginners-journey-3a11617313a4) uses a `784 → 64 → 32 → 10` architecture to classify digits.

## Working across computers

Run `git pull` before starting work. When finished, run `git add .`, `git commit -m "Describe your change"`, and `git push`. Dependencies, including `ipykernel`, are pinned in `requirements.txt` so the environment can be recreated.
