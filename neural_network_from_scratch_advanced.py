"""
Neural Network From Scratch — Advanced Single-File Edition
============================================================

A complete, small neural-network library in ONE file, built only on
NumPy. No PyTorch, no TensorFlow, no autograd — every gradient is
derived by hand and written out where it is used.

This file is fully self-contained and independent: the only import
in the entire file is NumPy.

What it covers
--------------
  1. Activation functions ..... sigmoid, tanh, relu, softmax
  2. Layers ................... Dense (fully connected), plus Sigmoid /
                                Tanh / ReLU / Softmax as layers
  3. Loss functions ........... MSELoss, BinaryCrossEntropy,
                                SoftmaxCrossEntropy
  4. Optimizers ............... SGD (with momentum), Adam
  5. The network .............. Sequential: forward, backward, and a
                                fit() loop with mini-batches, shuffling,
                                and validation-loss tracking
  6. Data utilities ........... toy datasets (XOR, circles, moons,
                                blobs, sine) + train_test_split, one_hot
  7. Gradient checking ........ verify backprop against brute-force
                                numerical gradients
  8. Demos .................... binary, non-linear, multi-class, and
                                regression problems, all in __main__

The one contract everything follows
-----------------------------------
Every layer implements two methods:

    output     = layer.forward(x)      # cache whatever backprop needs
    grad_input = layer.backward(grad)  # chain rule, one link at a time

`backward` receives dLoss/d(output) and returns dLoss/d(input).
A network is just a list of layers: forward runs it in order,
backward runs it in reverse order. That is all backpropagation is.

Shapes are batch-first everywhere:
    data         (batch, features)
    weights W    (in_features, out_features)
    biases  b    (out_features,)

Run it:
    python neural_network_from_scratch.py
"""

import numpy as np


# ===========================================================================
# 1. ACTIVATION FUNCTIONS
# ===========================================================================
# The non-linearity that makes a network more than a stack of linear
# equations. Without activations, any number of layers collapses into a
# single linear function. Each function is element-wise (except softmax,
# which works row-by-row) and shape-preserving.

def sigmoid(x):
    """Squash values into (0, 1). Classic for binary outputs."""
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x):
    """Squash values into (-1, 1). Zero-centered; nice for hidden layers."""
    return np.tanh(x)


def relu(x):
    """Rectified Linear Unit: 0 if negative, identity if positive."""
    return np.maximum(0.0, x)


def softmax(x):
    """
    Turn raw scores (logits) into probabilities that sum to 1, per row.
    x has shape (batch, classes).

    The row max is subtracted before exp() for numerical stability —
    it does not change the result, but prevents overflow on big logits.
    """
    shifted = x - np.max(x, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)


# ===========================================================================
# 2. LAYERS
# ===========================================================================

class Layer:
    """Base class. Stateless layers (activations) inherit this as-is."""

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def params_and_grads(self):
        """List of (parameter, gradient) pairs. Empty if no parameters."""
        return []


class Dense(Layer):
    """
    Fully connected layer: every input connected to every output.

        forward:   output = input @ W + b
        backward:  dW       = input.T @ grad_output
                   db       = sum(grad_output, over the batch)
                   d_input  = grad_output @ W.T

    Initialization matters: 'xavier' (default) scales the random weights
    by fan-in/fan-out and pairs well with tanh/sigmoid; 'he' pairs with
    ReLU. Bad init is a leading cause of "my network won't learn".
    """

    def __init__(self, in_features, out_features, init="xavier", seed=None):
        rng = np.random.default_rng(seed)
        if init == "xavier":
            scale = np.sqrt(2.0 / (in_features + out_features))
        elif init == "he":
            scale = np.sqrt(2.0 / in_features)
        else:
            raise ValueError(f"Unknown init '{init}', use 'xavier' or 'he'")

        self.W = rng.standard_normal((in_features, out_features)) * scale
        self.b = np.zeros(out_features)
        self._input = None  # cached in forward(), needed for dW
        self.dW = None
        self.db = None

    def forward(self, x):
        self._input = x
        return x @ self.W + self.b

    def backward(self, grad_output):
        self.dW = self._input.T @ grad_output
        self.db = np.sum(grad_output, axis=0)
        return grad_output @ self.W.T

    def params_and_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]


# --- Activation layers: each caches its output (or input), because its
# --- derivative is written most simply in terms of it. ---------------------

class Sigmoid(Layer):
    def forward(self, x):
        self._output = sigmoid(x)
        return self._output

    def backward(self, grad_output):
        # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        return grad_output * self._output * (1.0 - self._output)


class Tanh(Layer):
    def forward(self, x):
        self._output = tanh(x)
        return self._output

    def backward(self, grad_output):
        # d/dx tanh(x) = 1 - tanh(x)^2
        return grad_output * (1.0 - self._output ** 2)


class ReLU(Layer):
    def forward(self, x):
        self._input = x
        return relu(x)

    def backward(self, grad_output):
        # Derivative is 1 where the input was positive, 0 elsewhere.
        return grad_output * (self._input > 0)


class Softmax(Layer):
    """
    Logits -> probabilities, row by row.

    Included for inference and completeness. For *training* a classifier,
    prefer SoftmaxCrossEntropy (Section 3): it takes raw logits directly,
    is more stable, and its gradient simplifies. This layer's backward()
    applies the full softmax Jacobian-vector product:
        dL/dx = s * (g - sum(g * s))      per row, s = softmax output
    """

    def forward(self, x):
        self._output = softmax(x)
        return self._output

    def backward(self, grad_output):
        s = self._output
        dot = np.sum(grad_output * s, axis=1, keepdims=True)
        return s * (grad_output - dot)


# ===========================================================================
# 3. LOSS FUNCTIONS
# ===========================================================================
# A loss turns "how wrong are we?" into one number, and provides the
# first gradient of the backward pass: dLoss/d(predictions).
# All losses average over the batch, and their gradients include that
# averaging factor — so layers never need to know the batch size.

class Loss:
    def forward(self, predictions, targets):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class MSELoss(Loss):
    """Mean Squared Error, for regression.  L = mean((pred - y)^2)"""

    def forward(self, predictions, targets):
        self._diff = predictions - targets
        return float(np.mean(self._diff ** 2))

    def backward(self):
        # d/dpred mean((pred - y)^2) = 2 * (pred - y) / num_elements
        return 2.0 * self._diff / self._diff.size


class BinaryCrossEntropy(Loss):
    """
    For binary classification. Predictions must already be probabilities
    in (0, 1) — end your network with a Sigmoid layer.

        L = -mean[ y*log(p) + (1-y)*log(1-p) ]
    """

    def forward(self, predictions, targets):
        eps = 1e-12  # keep log() away from exactly 0
        self._p = np.clip(predictions, eps, 1.0 - eps)
        self._y = targets
        return float(-np.mean(targets * np.log(self._p)
                              + (1.0 - targets) * np.log(1.0 - self._p)))

    def backward(self):
        grad = -(self._y / self._p - (1.0 - self._y) / (1.0 - self._p))
        return grad / self._p.size


class SoftmaxCrossEntropy(Loss):
    """
    For multi-class classification — the one to use.

    Predictions are RAW LOGITS: do not put a Softmax layer at the end of
    your network. This loss applies softmax internally (stably) and
    combines it with cross-entropy, because together their gradient
    simplifies to the beautifully simple:

        grad = (probabilities - one_hot_targets) / batch_size

    (Sigmoid + BinaryCrossEntropy enjoys the same kind of cancellation,
    which is why its combined gradient is just (p - y) — try deriving
    both; it is the best backprop exercise there is.)

    Targets may be class indices, shape (batch,), or one-hot rows,
    shape (batch, classes). Both are accepted.
    """

    def forward(self, logits, targets):
        m = logits.shape[0]
        if targets.ndim == 1:                    # class indices
            self._indices = targets.astype(int)
        else:                                    # one-hot rows
            self._indices = np.argmax(targets, axis=1)

        self._probs = softmax(logits)
        eps = 1e-12
        correct = self._probs[np.arange(m), self._indices]
        return float(-np.mean(np.log(np.clip(correct, eps, 1.0))))

    def backward(self):
        m = self._probs.shape[0]
        grad = self._probs.copy()
        grad[np.arange(m), self._indices] -= 1.0  # subtract one-hot target
        return grad / m


# ===========================================================================
# 4. OPTIMIZERS
# ===========================================================================
# An optimizer receives (parameter, gradient) pairs — the actual arrays
# inside the layers — and updates the parameters IN PLACE.
# State (momentum buffers, Adam moments) is keyed by id(parameter); that
# works because in-place updates keep each array's identity stable.

class Optimizer:
    def step(self, params_and_grads):
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent, with optional momentum.

        plain:     p := p - lr * grad
        momentum:  v := momentum * v - lr * grad ;  p := p + v
    """

    def __init__(self, lr=0.1, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self._velocity = {}

    def step(self, params_and_grads):
        for param, grad in params_and_grads:
            if grad is None:
                continue
            if self.momentum == 0.0:
                param -= self.lr * grad
            else:
                key = id(param)
                v = self._velocity.get(key)
                if v is None:
                    v = np.zeros_like(param)
                    self._velocity[key] = v
                v *= self.momentum
                v -= self.lr * grad
                param += v


class Adam(Optimizer):
    """
    Adam — usually the best default optimizer. Keeps a running average
    of the gradient (m) and of its square (v) per parameter, corrects
    their early-step bias, and scales each parameter's step by them:

        m := b1*m + (1-b1)*g        v := b2*v + (1-b2)*g^2
        p := p - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m = {}
        self._v = {}
        self._t = 0

    def step(self, params_and_grads):
        self._t += 1
        for param, grad in params_and_grads:
            if grad is None:
                continue
            key = id(param)
            if key not in self._m:
                self._m[key] = np.zeros_like(param)
                self._v[key] = np.zeros_like(param)
            m, v = self._m[key], self._v[key]
            m *= self.beta1
            m += (1.0 - self.beta1) * grad
            v *= self.beta2
            v += (1.0 - self.beta2) * grad ** 2
            m_hat = m / (1.0 - self.beta1 ** self._t)
            v_hat = v / (1.0 - self.beta2 ** self._t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ===========================================================================
# 5. THE NETWORK — Sequential
# ===========================================================================

class Sequential:
    """
    A network is a list of layers.

        forward:  data flows layer 0 -> 1 -> ... -> output
        backward: gradients flow in the exact reverse order

    fit() is the entire training algorithm that frameworks wrap in
    machinery. Per mini-batch it is exactly four steps:
        1. predictions = forward(X_batch)
        2. loss_value  = loss.forward(predictions, y_batch)
        3. backward(loss.backward())        # fill in every gradient
        4. optimizer.step(params_and_grads) # nudge every parameter
    """

    def __init__(self, layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params_and_grads(self):
        pairs = []
        for layer in self.layers:
            pairs.extend(layer.params_and_grads())
        return pairs

    def predict(self, X):
        """Raw output for X: probabilities, logits, or values, depending
        on how the network was built."""
        return self.forward(X)

    def fit(self, X, y, loss, optimizer, epochs=100, batch_size=None,
            shuffle=True, seed=0, X_val=None, y_val=None, verbose=True):
        """
        Train the network.

            X, y        training data; X is (n_examples, n_features)
            loss        a Loss object, e.g. BinaryCrossEntropy()
            optimizer   an Optimizer object, e.g. Adam(lr=0.01)
            epochs      full passes over the data
            batch_size  None = full-batch gradient descent
            X_val/y_val optional held-out data, loss reported per epoch
                        and never trained on (watch it for overfitting)

        Returns history = {"loss": [...], "val_loss": [...]}.
        """
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        if batch_size is None:
            batch_size = n
        history = {"loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            order = rng.permutation(n) if shuffle else np.arange(n)
            epoch_loss, seen = 0.0, 0

            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                X_batch, y_batch = X[idx], y[idx]

                predictions = self.forward(X_batch)              # 1) forward
                batch_loss = loss.forward(predictions, y_batch)  # 2) loss
                self.backward(loss.backward())                   # 3) backward
                optimizer.step(self.params_and_grads())          # 4) update

                epoch_loss += batch_loss * len(idx)
                seen += len(idx)

            history["loss"].append(epoch_loss / seen)
            msg = f"  epoch {epoch:>4} | loss {history['loss'][-1]:.4f}"
            if X_val is not None:
                val_loss = loss.forward(self.forward(X_val), y_val)
                history["val_loss"].append(val_loss)
                msg += f" | val_loss {val_loss:.4f}"
            if verbose:
                print(msg)

        return history

    def __repr__(self):
        names = ", ".join(type(layer).__name__ for layer in self.layers)
        return f"Sequential([{names}])"


# ===========================================================================
# 6. DATA UTILITIES
# ===========================================================================
# Toy datasets, deliberately ones a straight line cannot solve (except
# the sine, which a straight line obviously cannot fit either). If your
# network can learn these, the machinery works.

def one_hot(indices, num_classes):
    """Class indices (n,) -> one-hot rows (n, num_classes)."""
    indices = np.asarray(indices).astype(int)
    out = np.zeros((indices.shape[0], num_classes))
    out[np.arange(indices.shape[0]), indices] = 1.0
    return out


def train_test_split(X, y, test_fraction=0.25, seed=0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(X.shape[0])
    n_test = int(round(X.shape[0] * test_fraction))
    test_idx, train_idx = order[:n_test], order[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def make_xor():
    """XOR: the textbook non-linearly-separable problem."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)
    return X, y


def make_circles(n=400, noise=0.08, seed=0):
    """Inner disc = 0, outer ring = 1. y has shape (n, 1)."""
    rng = np.random.default_rng(seed)
    n_in = n // 2
    r = rng.uniform(0.0, 1.0, n_in)
    a = rng.uniform(0, 2 * np.pi, n_in)
    inner = np.column_stack([r * np.cos(a), r * np.sin(a)])
    r = rng.uniform(1.6, 2.2, n - n_in)
    a = rng.uniform(0, 2 * np.pi, n - n_in)
    outer = np.column_stack([r * np.cos(a), r * np.sin(a)])
    X = np.vstack([inner, outer]) + rng.normal(0, noise, (n, 2))
    y = np.vstack([np.zeros((n_in, 1)), np.ones((n - n_in, 1))])
    order = rng.permutation(n)
    return X[order], y[order]


def make_moons(n=400, noise=0.12, seed=0):
    """Two interleaving half-circles. y has shape (n, 1)."""
    rng = np.random.default_rng(seed)
    n1 = n // 2
    t1 = rng.uniform(0, np.pi, n1)
    moon1 = np.column_stack([np.cos(t1), np.sin(t1)])
    t2 = rng.uniform(0, np.pi, n - n1)
    moon2 = np.column_stack([1.0 - np.cos(t2), 0.5 - np.sin(t2)])
    X = np.vstack([moon1, moon2]) + rng.normal(0, noise, (n, 2))
    y = np.vstack([np.zeros((n1, 1)), np.ones((n - n1, 1))])
    order = rng.permutation(n)
    return X[order], y[order]


def make_blobs(n=300, centers=((-2, -2), (0, 2), (2, -1)), std=0.6, seed=0):
    """
    Gaussian blobs, one per class — for multi-class demos.
    Returns X (n, 2) and integer class labels y (n,).
    """
    rng = np.random.default_rng(seed)
    centers = np.asarray(centers, dtype=float)
    per = n // len(centers)
    Xs, ys = [], []
    for k, c in enumerate(centers):
        Xs.append(rng.normal(c, std, (per, 2)))
        ys.append(np.full(per, k))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    order = rng.permutation(X.shape[0])
    return X[order], y[order]


def make_sine(n=200, noise=0.05, seed=0):
    """Regression data: y = sin(2*pi*x) + noise. X (n, 1), y (n, 1)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, 1))
    y = np.sin(2 * np.pi * X) + rng.normal(0, noise, (n, 1))
    return X, y


# ===========================================================================
# 7. GRADIENT CHECKING
# ===========================================================================
# How do you KNOW your hand-derived backprop is right? Brute force:
# nudge each parameter by +/- eps, re-measure the loss, and compare the
# numerical slope (loss+ - loss-) / 2eps against backprop's gradient.
# They must agree to several decimal places. If you ever write your own
# layer, run this on it before you trust a single training run.

def gradient_check(net, loss, X, y, eps=1e-5, tol=1e-5):
    """
    Compare backprop gradients vs numerical gradients for every
    parameter in `net`. Returns True if all agree within `tol`,
    and prints the worst error either way.
    """
    # Analytical gradients from backprop
    loss.forward(net.forward(X), y)
    net.backward(loss.backward())
    analytical = [(p, g.copy()) for p, g in net.params_and_grads()]

    worst = 0.0
    for param, analytic_grad in analytical:
        it = np.nditer(param, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            original = param[idx]
            param[idx] = original + eps
            loss_plus = loss.forward(net.forward(X), y)
            param[idx] = original - eps
            loss_minus = loss.forward(net.forward(X), y)
            param[idx] = original  # restore exactly
            numerical = (loss_plus - loss_minus) / (2 * eps)
            worst = max(worst, abs(numerical - analytic_grad[idx]))
            it.iternext()

    ok = worst < tol
    print(f"  gradient check: max |backprop - numerical| = {worst:.2e} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


# ===========================================================================
# 8. DEMOS
# ===========================================================================

def demo_gradient_checks():
    print("=" * 64)
    print("Demo 0: Gradient checks — is the backprop math actually right?")
    print("=" * 64)
    rng = np.random.default_rng(0)

    # (a) Tanh hidden layer + SoftmaxCrossEntropy (multi-class)
    X = rng.standard_normal((5, 3))
    y_idx = np.array([0, 1, 2, 1, 0])
    net = Sequential([Dense(3, 4, seed=1), Tanh(), Dense(4, 3, seed=2)])
    print(" Tanh net + SoftmaxCrossEntropy:")
    assert gradient_check(net, SoftmaxCrossEntropy(), X, y_idx)

    # (b) Sigmoid output + BinaryCrossEntropy
    Xb = rng.standard_normal((5, 2))
    yb = (rng.random((5, 1)) > 0.5).astype(float)
    net = Sequential([Dense(2, 4, seed=3), Tanh(), Dense(4, 1, seed=4), Sigmoid()])
    print(" Tanh/Sigmoid net + BinaryCrossEntropy:")
    assert gradient_check(net, BinaryCrossEntropy(), Xb, yb)

    # (c) ReLU hidden layer + MSELoss (regression)
    Xr = rng.standard_normal((4, 2))
    yr = rng.standard_normal((4, 1))
    net = Sequential([Dense(2, 5, init="he", seed=5), ReLU(), Dense(5, 1, seed=6)])
    print(" ReLU net + MSELoss:")
    assert gradient_check(net, MSELoss(), Xr, yr)
    print()


def demo_xor():
    print("=" * 64)
    print("Demo 1: XOR — binary, full-batch SGD  (2 -> 8 -> 1)")
    print("=" * 64)
    X, y = make_xor()
    net = Sequential([Dense(2, 8, seed=1), Tanh(), Dense(8, 1, seed=2), Sigmoid()])
    net.fit(X, y, loss=BinaryCrossEntropy(), optimizer=SGD(lr=0.5),
            epochs=500, verbose=False)
    probs = net.predict(X)
    for xi, p, yi in zip(X, probs, y):
        print(f"  {xi.astype(int)} -> prob {p[0]:.3f} | predicted "
              f"{int(p[0] >= 0.5)} | true {int(yi[0])}")
    print(f"  accuracy: {np.mean((probs >= 0.5) == y):.0%}\n")


def demo_circles():
    print("=" * 64)
    print("Demo 2: Circles — mini-batches, Adam, held-out test set")
    print("        (2 -> 16 -> 16 -> 1)")
    print("=" * 64)
    X, y = make_circles(n=400, seed=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_fraction=0.25, seed=0)
    net = Sequential([
        Dense(2, 16, seed=1), Tanh(),
        Dense(16, 16, seed=2), Tanh(),
        Dense(16, 1, seed=3), Sigmoid(),
    ])
    net.fit(X_tr, y_tr, loss=BinaryCrossEntropy(), optimizer=Adam(lr=0.02),
            epochs=100, batch_size=32, X_val=X_te, y_val=y_te, verbose=False)
    acc_tr = np.mean((net.predict(X_tr) >= 0.5) == y_tr)
    acc_te = np.mean((net.predict(X_te) >= 0.5) == y_te)
    print(f"  train accuracy: {acc_tr:.1%}")
    print(f"  test  accuracy: {acc_te:.1%}  <- data never trained on\n")


def demo_multiclass():
    print("=" * 64)
    print("Demo 3: Multi-class blobs — ReLU, SoftmaxCrossEntropy")
    print("        Note: NO Softmax layer; the loss handles it. (2 -> 32 -> 3)")
    print("=" * 64)
    X, y = make_blobs(n=300, seed=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_fraction=0.25, seed=0)
    net = Sequential([
        Dense(2, 32, init="he", seed=1), ReLU(),
        Dense(32, 3, seed=2),  # 3 raw logits out
    ])
    net.fit(X_tr, y_tr, loss=SoftmaxCrossEntropy(), optimizer=Adam(lr=0.02),
            epochs=100, batch_size=32, verbose=False)
    pred = np.argmax(net.predict(X_te), axis=1)
    print(f"  test accuracy: {np.mean(pred == y_te):.1%}\n")


def demo_regression():
    print("=" * 64)
    print("Demo 4: Regression — fit sin(2*pi*x) with MSELoss")
    print("        Same machinery; no output activation. (1 -> 32 -> 32 -> 1)")
    print("=" * 64)
    X, y = make_sine(n=200, seed=0)
    net = Sequential([
        Dense(1, 32, seed=1), Tanh(),
        Dense(32, 32, seed=2), Tanh(),
        Dense(32, 1, seed=3),
    ])
    history = net.fit(X, y, loss=MSELoss(), optimizer=Adam(lr=0.01),
                      epochs=300, batch_size=32, verbose=False)
    print(f"  final MSE: {history['loss'][-1]:.4f}")
    print("   x      predicted   true")
    for x in [0.0, 0.125, 0.25, 0.375, 0.5, 0.75]:
        pred = net.predict(np.array([[x]]))[0, 0]
        print(f"  {x:.3f}    {pred:+.3f}      {np.sin(2 * np.pi * x):+.3f}")
    print()


def demo_momentum():
    print("=" * 64)
    print("Demo 5: Moons with SGD + momentum (the fifth dataset)")
    print("        (2 -> 16 -> 1)")
    print("=" * 64)
    X, y = make_moons(n=400, seed=0)
    net = Sequential([Dense(2, 16, seed=1), Tanh(), Dense(16, 1, seed=2), Sigmoid()])
    history = net.fit(X, y, loss=BinaryCrossEntropy(),
                      optimizer=SGD(lr=0.2, momentum=0.9),
                      epochs=60, batch_size=32, verbose=False)
    acc = np.mean((net.predict(X) >= 0.5) == y)
    print(f"  loss: {history['loss'][0]:.4f} -> {history['loss'][-1]:.4f}")
    print(f"  accuracy: {acc:.1%}\n")


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=3)
    demo_gradient_checks()
    demo_xor()
    demo_circles()
    demo_multiclass()
    demo_regression()
    demo_momentum()

    print("Try next:")
    print("  - Remove XOR's hidden layer: Sequential([Dense(2,1), Sigmoid()])")
    print("    caps at 75% — that IS logistic regression, and it proves why")
    print("    hidden layers exist.")
    print("  - Shrink circles to Dense(2, 2) hidden layers and watch accuracy")
    print("    fall: not enough capacity to bend the decision boundary.")
    print("  - Write your own layer (e.g. LeakyReLU) and gradient_check() it")
    print("    before training anything with it.")
