# CNN from Scratch (NumPy)

Minimal neural network layers and a tiny training loop implemented from first principles using NumPy/SciPy. Includes two runnable examples on MNIST:
- mnist.py — a simple fully connected network (MLP) for digit classification.
- mnist_conv.py — a tiny convolutional network demo on a 0-vs-1 subset.

## Features
- Core layer abstractions: Dense, Convolutional, Reshape, Activations (Tanh, Sigmoid, Softmax)
- Losses: Mean Squared Error (MSE), Binary Cross-Entropy (BCE)
- Micro training loop with forward/backward pass and SGD
- Small, readable code intended for learning and experimentation

## Project structure
- activation.py — base Activation layer
- activations.py — Tanh, Sigmoid, Softmax implementations
- dense.py — fully connected (affine) layer
- convolutional.py — 2D convolution layer using scipy.signal
- reshape.py — reshape layer for bridging conv ↔ dense
- losses.py — MSE and BCE losses and their derivatives
- network.py — predict() and train() utilities
- mnist.py — MLP example on MNIST
- mnist_conv.py — small CNN example (0 vs 1)

## Requirements
- Python 3.8+
- NumPy
- SciPy
- Keras (for loading the MNIST dataset via `keras.datasets`)
  - Any supported backend is fine; if you prefer TensorFlow, install it as well.

### Quick install
Using a virtual environment is recommended.

Windows (PowerShell):

```
py -3 -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy scipy keras  # optionally: pip install tensorflow
```

macOS/Linux (bash):

```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy scipy keras  # optionally: pip install tensorflow
```

If you prefer pinning versions, create a `requirements.txt` and install with `pip install -r requirements.txt`.

## Running the examples
- Fully connected network:

```
python mnist.py
```

- Convolutional example (0 vs 1 subset):

```
python mnist_conv.py
```

Both scripts will:
1) Download MNIST via `keras.datasets` (first run only).
2) Preprocess the data.
3) Build a small network as a Python list of layers.
4) Train with SGD and print an error metric per epoch.
5) Print a few predictions at the end.

Note: These examples are intentionally small for educational purposes and may trade off accuracy/efficiency for clarity.

## How it works (high level)
- Layers expose `forward(x)` and `backward(grad, lr)`; gradients are computed analytically per layer.
- `network.train(...)` performs forward → loss → backward across the list of layers and applies SGD updates in-place.
- Activation and loss derivatives are implemented explicitly to keep the math visible.

## Extending
- Add a new activation: subclass `Activation` and provide the function and its derivative.
- Add a new layer: subclass `Layer` (see `layer.py`) and implement `forward`/`backward`.
- Swap losses or activations in the example scripts to observe training behavior changes.

## Troubleshooting
- SciPy is required for the convolution layer (`scipy.signal`). Ensure `pip show scipy` lists an installed version.
- If `from keras.datasets import mnist` fails, install a backend (e.g., `pip install tensorflow`) or ensure the `keras` package is installed.
- Training is CPU-only and intentionally simple; expect it to be slow for large epoch counts.

## License
See `LICENSE` for details.
