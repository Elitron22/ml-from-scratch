# MNIST Neural Network from Scratch

Build a handwritten digit classifier step by step in one notebook. The notebook includes MNIST loading, preparation, and basic exploration. The neural network is intentionally left for you to implement and explain line by line.

## Structure

```text
projects/mnist-neural-network/
├── mnist_neural_network.ipynb
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
```

Open this project folder in VS Code, open `mnist_neural_network.ipynb`, and select the project's `.venv` as the notebook kernel. VS Code's Python and Jupyter extensions provide the notebook interface. Each computer creates its own `.venv`; the environment is not committed to Git.

On macOS or Linux, create the environment with `python3 -m venv .venv` and use `.venv/bin/python` for package installation.

## Notebook contents

1. Import NumPy, Matplotlib, and the MNIST downloader.
2. Download MNIST and normalize its pixel values.
3. Split it into training, validation, and test sets.
4. Display an image and inspect its label and pixel values.
5. Implement the neural network yourself in the final section.

The first run downloads MNIST into scikit-learn's local cache. The notebook does not train a model yet.

The [reference article](https://medium.com/@pankajgoyal4152/understanding-neural-networks-by-building-one-from-scratch-a-beginners-journey-3a11617313a4) uses a `784 → 64 → 32 → 10` architecture to classify digits.

## Working across computers

Run `git pull` before starting work. When finished, run `git add .`, `git commit -m "Describe your change"`, and `git push`. Dependencies, including `ipykernel`, are pinned in `requirements.txt` so the environment can be recreated.
