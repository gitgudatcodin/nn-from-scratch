"""
Neural Network From Scratch — NumPy Only
==========================================

A minimal, readable implementation of a fully-connected neural network
(Multi-Layer Perceptron) with:

  - Forward propagation
  - Backpropagation (hand-derived gradients, no autograd)
  - Gradient descent weight updates
  - Vectorized operations with NumPy (whole batch at once, no per-example loops)

No PyTorch, no TensorFlow, no scikit-learn.

Run it:
    python neural_network_from_scratch.py

It trains on two toy problems that a single straight line *cannot* solve:
  1. XOR
  2. Concentric circles

Shapes used throughout (batch-first, like PyTorch / Keras):
    X : (m, n_in)    m examples, n_in features each
    W : (n_in, n_out)
    b : (n_out,)
    Z, A : (m, n_out)
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Activation functions (and their derivatives)
# ---------------------------------------------------------------------------

def sigmoid(z):
    """Squashes any number into (0, 1). Used on the output layer."""
    return 1.0 / (1.0 + np.exp(-z))


def tanh(z):
    """Squashes any number into (-1, 1). Used on the hidden layers."""
    return np.tanh(z)


def tanh_derivative(a):
    """
    Derivative of tanh, written in terms of its *output* a = tanh(z).

        d/dz tanh(z) = 1 - tanh(z)^2 = 1 - a^2

    This is why we cache activations in the forward pass — backprop needs them.
    """
    return 1.0 - a ** 2


# ---------------------------------------------------------------------------
# 2. Loss function
# ---------------------------------------------------------------------------

def binary_cross_entropy(y_pred, y_true):
    """
    Average binary cross-entropy loss for one batch.

        L = -(1/m) * sum[ y*log(p) + (1-y)*log(1-p) ]

    y_pred is clipped so log() never sees exactly 0 or 1.
    """
    m = y_true.shape[0]
    eps = 1e-12
    p = np.clip(y_pred, eps, 1.0 - eps)
    return float(-np.sum(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)) / m)


# ---------------------------------------------------------------------------
# 3. The network
# ---------------------------------------------------------------------------

class NeuralNetwork:
    """
    A fully-connected network for binary classification.

    Example:
        NeuralNetwork([2, 16, 16, 1])
                         │   │   │   └─ output layer: 1 neuron (a probability)
                         │   │   └───── hidden layer 2: 16 neurons
                         │   └───────── hidden layer 1: 16 neurons
                         └───────────── input: 2 features

    Hidden layers use tanh, the output layer uses sigmoid.
    """

    def __init__(self, layer_sizes, seed=42):
        rng = np.random.default_rng(seed)

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1  # number of weight layers

        self.weights = []  # W for each layer
        self.biases = []   # b for each layer

        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # Xavier/Glorot initialization: keeps activations from exploding
            # or vanishing at the start of training.
            # Var(W) = 2 / (n_in + n_out)
            limit = np.sqrt(2.0 / (n_in + n_out))
            W = rng.standard_normal((n_in, n_out)) * limit
            b = np.zeros((n_out,))

            self.weights.append(W)
            self.biases.append(b)

        # Filled in during forward(), consumed during backward()
        self._cache_Z = []  # pre-activations  Z = A_prev @ W + b
        self._cache_A = []  # post-activations A = activation(Z), includes input X as A[0]

    # -- Forward propagation ------------------------------------------------

    def forward(self, X):
        """
        Push a whole batch through the network.

        For each layer l:
            Z[l] = A[l-1] @ W[l] + b[l]     (linear step)
            A[l] = activation(Z[l])         (non-linear step)

        Returns the final activation: shape (m, 1), values in (0, 1).
        """
        self._cache_A = [X]
        self._cache_Z = []

        A = X
        for i in range(self.num_layers):
            Z = A @ self.weights[i] + self.biases[i]
            is_output_layer = (i == self.num_layers - 1)
            A = sigmoid(Z) if is_output_layer else tanh(Z)

            self._cache_Z.append(Z)
            self._cache_A.append(A)

        return A

    # -- Backward propagation -----------------------------------------------

    def backward(self, y_true, learning_rate):
        """
        Compute gradients with the chain rule, working from the output
        layer back to the input, then take one gradient-descent step.

        Key simplification for the output layer:
            sigmoid + binary cross-entropy  =>  dZ_output = A_output - y
        (the sigmoid derivative and the loss derivative cancel neatly)

        Then for each layer, going backwards:
            dW       = A_prev.T @ dZ
            db       = sum(dZ, over the batch)
            dA_prev  = dZ @ W.T
            dZ_prev  = dA_prev * tanh'(A_prev)     # hidden layers only
        """
        m = y_true.shape[0]
        grads_W = [None] * self.num_layers
        grads_b = [None] * self.num_layers

        # Output layer gradient
        A_out = self._cache_A[-1]
        dZ = (A_out - y_true) / m  # shape (m, 1) — the /m matches the averaged loss

        for i in reversed(range(self.num_layers)):
            A_prev = self._cache_A[i]  # input to layer i

            # Gradients for this layer's parameters
            grads_W[i] = A_prev.T @ dZ        # (n_in, n_out)
            grads_b[i] = np.sum(dZ, axis=0)   # (n_out,)

            # Propagate the error signal to the previous layer
            if i > 0:
                dA_prev = dZ @ self.weights[i].T          # (m, n_in)
                dZ = dA_prev * tanh_derivative(A_prev)    # chain rule through tanh

        # Gradient descent update: W := W - lr * dW
        for i in range(self.num_layers):
            self.weights[i] -= learning_rate * grads_W[i]
            self.biases[i] -= learning_rate * grads_b[i]

    # -- Training & inference -------------------------------------------------

    def train(self, X, y, epochs=1000, learning_rate=0.1, print_every=200):
        """Full-batch gradient descent: every epoch uses all examples at once."""
        for epoch in range(1, epochs + 1):
            y_pred = self.forward(X)          # 1) forward
            loss = binary_cross_entropy(y_pred, y)
            self.backward(y, learning_rate)   # 2) backward + 3) update

            if epoch == 1 or epoch % print_every == 0:
                acc = self.accuracy(X, y)
                print(f"  epoch {epoch:>5} | loss {loss:.4f} | accuracy {acc:.1%}")

        return self

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))


# ---------------------------------------------------------------------------
# 4. Toy datasets (generated with NumPy only)
# ---------------------------------------------------------------------------

def make_xor():
    """The classic XOR problem — not linearly separable."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)
    return X, y


def make_circles(n=400, noise=0.08, seed=0):
    """Inner disc = class 0, outer ring = class 1."""
    rng = np.random.default_rng(seed)

    n_inner = n // 2
    r_inner = rng.uniform(0.0, 1.0, n_inner)
    a_inner = rng.uniform(0, 2 * np.pi, n_inner)
    inner = np.column_stack([r_inner * np.cos(a_inner), r_inner * np.sin(a_inner)])

    n_outer = n - n_inner
    r_outer = rng.uniform(1.6, 2.2, n_outer)
    a_outer = rng.uniform(0, 2 * np.pi, n_outer)
    outer = np.column_stack([r_outer * np.cos(a_outer), r_outer * np.sin(a_outer)])

    X = np.vstack([inner, outer]) + rng.normal(0, noise, (n, 2))
    y = np.vstack([np.zeros((n_inner, 1)), np.ones((n_outer, 1))])

    # Shuffle so the batch isn't ordered by class
    order = rng.permutation(n)
    return X[order], y[order]


# ---------------------------------------------------------------------------
# 5. Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=3)

    print("=" * 60)
    print("Demo 1: XOR  (network: 2 -> 8 -> 1)")
    print("=" * 60)
    X, y = make_xor()
    net = NeuralNetwork([2, 8, 1], seed=1)
    net.train(X, y, epochs=2000, learning_rate=0.5, print_every=500)

    print("\n  Predictions:")
    for xi, yi, pi in zip(X, y, net.predict_proba(X)):
        print(f"    {xi.astype(int)} -> prob {pi[0]:.3f} | predicted {int(pi[0] >= 0.5)} | true {int(yi[0])}")

    print()
    print("=" * 60)
    print("Demo 2: Concentric circles  (network: 2 -> 16 -> 16 -> 1)")
    print("=" * 60)
    X, y = make_circles(n=400, seed=0)
    # Hold out the last 100 examples to check generalization
    X_train, y_train = X[:300], y[:300]
    X_test, y_test = X[300:], y[300:]

    net = NeuralNetwork([2, 16, 16, 1], seed=42)
    net.train(X_train, y_train, epochs=1500, learning_rate=0.5, print_every=300)

    print(f"\n  Train accuracy: {net.accuracy(X_train, y_train):.1%}")
    print(f"  Test  accuracy: {net.accuracy(X_test, y_test):.1%}  <- unseen data")

    print()
    print("Try next:")
    print("  - Change [2, 16, 16, 1] to [2, 2, 1] and watch circles fail:")
    print("    too few hidden neurons = not enough capacity to bend the boundary.")
    print("  - Crank learning_rate to 5.0 and watch the loss blow up / oscillate.")
    print("  - Set learning_rate to 0.001 and watch it learn painfully slowly.")
