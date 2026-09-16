"""
Neural Networks From Scratch: Forward and Backward

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - numerical_gradient
import numpy as np

def numerical_gradient(f, x, eps=1e-5):
    # Ensure float dtype to prevent integer truncation during perturbation
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x, dtype=float)

    for idx in np.ndindex(x.shape):
        orig = x[idx]

        # f(x + eps)
        x[idx] = orig + eps
        f_plus = f(x)

        # f(x - eps)
        x[idx] = orig - eps
        f_minus = f(x)

        # Central difference: (f(x + eps) - f(x - eps)) / (2 * eps)
        grad[idx] = (f_plus - f_minus) / (2.0 * eps)

        # Restore original value
        x[idx] = orig

    return grad

# Step 2 - gradient_check
import numpy as np

def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    """
    Return max relative error between analytic and numeric gradients.
    """
    a = np.asarray(analytic_grad, dtype=float)
    n = np.asarray(numeric_grad, dtype=float)

    numerator = np.abs(a - n)
    denominator = np.maximum(np.maximum(np.abs(a), np.abs(n)), tol)

    return float(np.max(numerator / denominator))

# Step 3 - make_dense
import numpy as np


def make_dense(in_dim, out_dim, weight_init_fn=None):
    """Create a fully connected layer.

    Inputs:
      in_dim: int, input feature size
      out_dim: int, output feature size
      weight_init_fn: callable(in_dim, out_dim) -> (W, b)

    Returns layer dict with keys:
      params: {'W': (in_dim, out_dim), 'b': (out_dim,)}
      forward(x) -> (y, cache) with y shape (batch, out_dim)
      backward(dout, cache) -> (dx, grads) with grads {'W', 'b'}
    """
    if weight_init_fn is None:

        def default_init(in_dim, out_dim):
            rng = np.random.RandomState(0)
            W = rng.randn(in_dim, out_dim) * 0.1
            b = np.zeros(out_dim)
            return W, b

        weight_init_fn = default_init

    W, b = weight_init_fn(in_dim, out_dim)
    params = {"W": W, "b": b}

    def forward(x):
        """Computes y = x @ W + b.

        x: (batch, in_dim)
        Returns:
          y: (batch, out_dim)
          cache: tuple for backward pass
        """
        y = np.dot(x, params["W"]) + params["b"]
        cache = (x, params["W"])
        return y, cache

    def backward(dout, cache):
        """Computes gradients with respect to x, W, and b.

        dout: upstream gradient, shape (batch, out_dim)
        cache: (x, W)
        Returns:
          dx: (batch, in_dim)
          grads: {'W': dW, 'b': db}
        """
        x, W = cache

        # dx = dout @ W.T
        dx = np.dot(dout, W.T)

        # dW = x.T @ dout
        dW = np.dot(x.T, dout)

        # db = sum along batch axis
        db = np.sum(dout, axis=0)

        grads = {"W": dW, "b": db}
        return dx, grads

    return {"params": params, "forward": forward, "backward": backward}

# Step 4 - make_activation
import numpy as np


def make_activation(kind="relu"):
    """Create a genuinely nonlinear elementwise activation layer.

    Args:
        kind: str nonlinearity name. Default 'relu' must implement ReLU
          (zero negatives, pass non-negatives). Other kinds optional ('sigmoid',
          'tanh').

    Returns:
        Layer dict with:
          forward(x) -> (y, cache)
            x, y: np.ndarray shape (batch, dim)
          backward(dout, cache) -> (dx, {})
            dout, dx: np.ndarray shape (batch, dim)
            param grad dict is always empty (no learnable params)

    Must be elementwise and non-affine; analytic dx must match
    numerical_gradient / gradient_check.
    """
    kind = kind.lower()

    if kind == "relu":

        def forward(x):
            y = np.maximum(0, x)
            cache = x
            return y, cache

        def backward(dout, cache):
            x = cache
            # Derivative is 1 for x > 0, 0 for x <= 0
            dx = dout * (x > 0)
            return dx, {}

    elif kind == "sigmoid":

        def forward(x):
            # Numerically stable sigmoid
            pos_mask = x >= 0
            neg_mask = ~pos_mask
            y = np.empty_like(x, dtype=np.float64)
            y[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
            exp_x = np.exp(x[neg_mask])
            y[neg_mask] = exp_x / (1.0 + exp_x)
            cache = y
            return y, cache

        def backward(dout, cache):
            y = cache
            dx = dout * y * (1.0 - y)
            return dx, {}

    elif kind == "tanh":

        def forward(x):
            y = np.tanh(x)
            cache = y
            return y, cache

        def backward(dout, cache):
            y = cache
            dx = dout * (1.0 - y**2)
            return dx, {}

    else:
        raise ValueError(f"Unsupported activation kind: {kind}")

    return {
        "params": {},
        "forward": forward,
        "backward": backward,
    }

# Step 5 - initialize_weights
import numpy as np


def initialize_weights(in_dim, out_dim, scheme="he"):
    """Return (W, b) for a dense layer.

    Inputs:
      in_dim: int fan-in
      out_dim: int fan-out
      scheme: str initialization family (default 'he')

    Returns:
      W: np.ndarray shape (in_dim, out_dim), finite, symmetry-breaking,
          scale stable with depth (fan-in dependent)
      b: np.ndarray shape (out_dim,), near zero
    """
    scheme = scheme.lower()

    if scheme in ("he", "kaiming"):
        std = np.sqrt(2.0 / in_dim)
        W = np.random.randn(in_dim, out_dim) * std

    elif scheme in ("xavier", "glorot"):
        std = np.sqrt(2.0 / (in_dim + out_dim))
        W = np.random.randn(in_dim, out_dim) * std

    elif scheme == "lecun":
        std = np.sqrt(1.0 / in_dim)
        W = np.random.randn(in_dim, out_dim) * std

    else:
        raise ValueError(f"Unsupported initialization scheme: {scheme}")

    b = np.zeros(out_dim, dtype=np.float64)
    return W.astype(np.float64), b

# Step 6 - make_loss
import numpy as np


def make_loss(kind="cross_entropy"):
    """Return a classification loss_fn(logits, labels) -> (loss, d_logits).

    Inputs to loss_fn:
      logits: (batch, C) float array of raw class scores
      labels: (batch,) int array of class indices in [0, C)
    Outputs:
      loss: Python float, mean scalar loss over the batch (finite)
      d_logits: (batch, C) gradient of loss w.r.t. logits (finite)
    Must pass gradient_check, be minimized by confident correct predictions,
    and stay finite under saturated logits.
    """
    if kind != "cross_entropy":
        raise ValueError(f"Unsupported loss kind: {kind}")

    def loss_fn(logits, labels):
        batch_size = logits.shape[0]

        # Shift logits for numerical stability against overflow: max(logits) along axis=1
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)

        # Compute stable softmax probabilities
        exp_scores = np.exp(shifted_logits)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Log-sum-exp trick for stable negative log-likelihood calculation:
        # log(probs[i, y_i]) = shifted_logits[i, y_i] - log(sum(exp(shifted_logits[i])))
        log_sum_exp = np.log(np.sum(exp_scores, axis=1))
        correct_class_logits = shifted_logits[np.arange(batch_size), labels]
        loss = -np.mean(correct_class_logits - log_sum_exp)

        # Gradient of categorical cross-entropy with softmax w.r.t. logits:
        # dL/d(z_ik) = (probs[i, k] - 1{k == y_i}) / batch_size
        d_logits = probs.copy()
        d_logits[np.arange(batch_size), labels] -= 1.0
        d_logits /= batch_size

        return float(loss), d_logits

    return loss_fn

# Step 7 - make_sequential (not yet solved)
# TODO: implement

# Step 8 - forward_backward (not yet solved)
# TODO: implement

# Step 9 - make_optimizer (not yet solved)
# TODO: implement

# Step 10 - train_step (not yet solved)
# TODO: implement

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

