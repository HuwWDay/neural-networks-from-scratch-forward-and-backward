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

# Step 4 - make_activation (not yet solved)
# TODO: implement

# Step 5 - initialize_weights (not yet solved)
# TODO: implement

# Step 6 - make_loss (not yet solved)
# TODO: implement

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

