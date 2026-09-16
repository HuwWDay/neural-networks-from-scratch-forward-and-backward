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

# Step 7 - make_sequential
def make_sequential(layers):
    """Compose protocol-honoring layers into one sequential model.

    Inputs:
      layers: list of layer dicts, each with
        forward(x) -> (y, cache),
        backward(dout, cache) -> (dx, grads_dict),
        params: dict of ndarrays (possibly empty).

    Returns a dict with:
      forward(x) -> (y, caches)
        y: final activation after applying every layer in order
        caches: opaque structure needed by backward
      backward(dout, caches) -> (dx, grads_list)
        dx: gradient w.r.t. the original input x
        grads_list: list of length len(layers); grads_list[i] is the
          grads_dict from layers[i] ({} for param-free layers)
      params: aggregated live view of all layer params, length len(layers),
        same order as layers (so in-place updates affect the model)
    """
    # Live view of parameter dicts in the same order as layers
    params = [layer.get("params", {}) for layer in layers]

    def forward(x):
        caches = []
        out = x
        for layer in layers:
            out, cache = layer["forward"](out)
            caches.append(cache)
        return out, caches

    def backward(dout, caches):
        grads_list = [None] * len(layers)
        current_dout = dout

        # Traverse backward in reverse topological order
        for i in reversed(range(len(layers))):
            current_dout, grads = layers[i]["backward"](current_dout, caches[i])
            grads_list[i] = grads

        return current_dout, grads_list

    return {
        "forward": forward,
        "backward": backward,
        "params": params,
    }

# Step 8 - forward_backward
def forward_backward(model, loss_fn, x, y):
    """Run one full forward-backward sweep on a batch.

    Inputs:
      model: sequential dict with 'forward', 'backward', 'params'
             model['forward'](x) -> (logits, caches)
             model['backward'](d_logits, caches) -> (dx, param_grads)
      loss_fn: callable (logits, y) -> (loss, d_logits)
      x: np.ndarray (batch, in_dim)
      y: np.ndarray (batch,) integer labels

    Returns:
      loss: float, scalar batch loss
      param_grads: nested np.ndarrays matching model['params'] layout
                    (gradients of loss w.r.t. every parameter)
    """
    # 1. Forward pass through the network
    logits, caches = model["forward"](x)

    # 2. Compute loss and initial upstream gradient w.r.t. logits
    loss, d_logits = loss_fn(logits, y)

    # 3. Backward pass through the network
    _, param_grads = model["backward"](d_logits, caches)

    return loss, param_grads

# Step 9 - make_optimizer
import numpy as np


def make_optimizer(params, lr=1e-2, kind="sgd", **kwargs):
    """Build an optimizer that updates params in place.

    Inputs:
      params: arrays, possibly nested in lists/dicts (or dict of arrays) to optimize
      lr: float learning rate
      kind: str algorithm name (e.g. 'sgd', 'adam', 'momentum')

    Returns:
      dict with key 'step'. step(grads) applies one in-place update
      using grads structured like params. Parameter shapes must stay
      unchanged. Repeated steps must reduce a simple convex objective
      within a modest fixed budget and keep values finite.
    """
    kind = kind.lower()

    # Flatten nested params to track state per leaf array
    flattened_params = []

    def _collect(p):
        if isinstance(p, dict):
            for k, v in p.items():
                _collect(v)
        elif isinstance(p, (list, tuple)):
            for item in p:
                _collect(item)
        elif isinstance(p, np.ndarray):
            flattened_params.append(p)

    _collect(params)

    # State containers for momentum/Adam
    state = {
        "t": 0,
        "m": [np.zeros_like(p) for p in flattened_params],
        "v": [np.zeros_like(p) for p in flattened_params],
    }

    # Hyperparameters
    beta1 = kwargs.get("beta1", 0.9)
    beta2 = kwargs.get("beta2", 0.999)
    eps = kwargs.get("eps", 1e-8)

    def step(grads):
        # Extract gradients in matching order
        flattened_grads = []

        def _collect_grads(g):
            if isinstance(g, dict):
                for k, v in g.items():
                    _collect_grads(v)
            elif isinstance(g, (list, tuple)):
                for item in g:
                    _collect_grads(item)
            elif isinstance(g, np.ndarray):
                flattened_grads.append(g)

        _collect_grads(grads)

        state["t"] += 1
        t = state["t"]

        for i, (p, g) in enumerate(zip(flattened_params, flattened_grads)):
            if kind == "sgd":
                # In-place parameter update: p -= lr * g
                p -= lr * g

            elif kind == "momentum":
                state["m"][i] = beta1 * state["m"][i] + (1.0 - beta1) * g
                p -= lr * state["m"][i]

            elif kind == "adam":
                # First and second moment updates
                state["m"][i] = beta1 * state["m"][i] + (1.0 - beta1) * g
                state["v"][i] = beta2 * state["v"][i] + (1.0 - beta2) * (g**2)

                # Bias correction
                m_hat = state["m"][i] / (1.0 - beta1**t)
                v_hat = state["v"][i] / (1.0 - beta2**t)

                p -= lr * m_hat / (np.sqrt(v_hat) + eps)

            else:
                raise ValueError(f"Unsupported optimizer kind: {kind}")

    return {"step": step}

# Step 10 - train_step
def train_step(model, loss_fn, optimizer, x_batch, y_batch):
    """Perform one complete optimization step over a minibatch.

    Inputs:
      model: sequential model dict with 'forward', 'backward', and 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x_batch: np.ndarray of shape (B, D)
      y_batch: np.ndarray of shape (B,) integer class labels

    Returns:
      loss: float, scalar batch loss evaluated BEFORE the parameter update.
      Model parameters are updated in place; shapes unchanged and values finite.
    """
    # 1. Forward sweep: compute logits and activation caches
    logits, caches = model["forward"](x_batch)

    # 2. Evaluate loss and seed upstream gradient w.r.t logits before updating
    loss, d_logits = loss_fn(logits, y_batch)

    # 3. Backward sweep: backpropagate gradients to every parameter
    _, grads = model["backward"](d_logits, caches)

    # 4. Update parameters in place
    optimizer["step"](grads)

    return float(loss)

# Step 11 - train
import numpy as np


def train(model, loss_fn, optimizer, x, y, epochs, batch_size, seed=0):
    """Run a deterministic minibatch training loop.

    Inputs:
      model: sequential model dict with 'forward', 'backward', 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x: np.ndarray of shape (N, D) training features
      y: np.ndarray of shape (N,) integer class labels
      epochs: int, number of full passes over the data
      batch_size: int, minibatch size
      seed: int, RNG seed for deterministic shuffling / batching

    Returns:
      history: list[float] of length `epochs`; history[t] is the mean
      train_step loss over minibatches in epoch t.
      Model parameters are updated in place; shapes unchanged.
    """
    n_samples = x.shape[0]
    rng = np.random.RandomState(seed)
    history = []

    for _ in range(epochs):
        indices = rng.permutation(n_samples)
        epoch_losses = []

        for start_idx in range(0, n_samples, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            x_batch = x[batch_indices]
            y_batch = y[batch_indices]

            # 1. Forward pass
            logits, caches = model["forward"](x_batch)

            # 2. Compute loss and gradient w.r.t logits before parameter update
            batch_loss, d_logits = loss_fn(logits, y_batch)
            epoch_losses.append(batch_loss)

            # 3. Backward pass
            _, grads = model["backward"](d_logits, caches)

            # 4. In-place parameter update
            optimizer["step"](grads)

        history.append(float(np.mean(epoch_losses)))

    return history

# Step 12 - design_network
import numpy as np


def design_network(input_dim, num_classes, seed=0):
    """Design and train a net that solves a nonlinear classification task.

    Inputs:
      input_dim: int, feature dimension
      num_classes: int, number of classes
      seed: int, RNG seed for reproducibility

    Returns:
      model: trained sequential model (forward/backward/params)
      metrics: dict with
        'accuracy': float >= 0.90 on an evaluation set,
        'x': np.ndarray (N, input_dim) eval features (N >= 50),
        'y': np.ndarray (N,) integer eval labels.
    """
    rng = np.random.RandomState(seed)

    def generate_data(n_samples):
        # Multi-arm spiral / interleaved concentric manifold
        # Guaranteed non-linearly separable (linear classifier achieves < 0.70)
        n_per_class = n_samples // num_classes
        total_n = n_per_class * num_classes

        x = np.zeros((total_n, input_dim), dtype=np.float64)
        y = np.zeros(total_n, dtype=np.int64)

        for j in range(num_classes):
            idx = range(j * n_per_class, (j + 1) * n_per_class)
            r = np.linspace(0.1, 1.0, n_per_class)
            # Spiral angle with angular separation per class
            t = (
                np.linspace(j * 4.0, (j + 1) * 4.0, n_per_class)
                + rng.randn(n_per_class) * 0.15
            )

            x[idx, 0] = r * np.sin(t)
            if input_dim > 1:
                x[idx, 1] = r * np.cos(t)
            if input_dim > 2:
                # Fill extra dimensions with low-variance noise
                x[idx, 2:] = rng.randn(n_per_class, input_dim - 2) * 0.05
            y[idx] = j

        # Shuffle points
        perm = rng.permutation(total_n)
        return x[perm], y[perm]

    # Generate training and evaluation sets
    x_train, y_train = generate_data(n_samples=600)
    x_eval, y_eval = generate_data(n_samples=200)

    # Custom weight initializer bound to seed
    def custom_init(in_d, out_d):
        std = np.sqrt(2.0 / in_d)
        W = rng.randn(in_d, out_d) * std
        b = np.zeros(out_d, dtype=np.float64)
        return W, b

    # Build 2-hidden-layer MLP: input_dim -> 64 -> 32 -> num_classes
    hidden_dim1 = 64
    hidden_dim2 = 32

    layers = [
        make_dense(input_dim, hidden_dim1, weight_init_fn=custom_init),
        make_activation("relu"),
        make_dense(hidden_dim1, hidden_dim2, weight_init_fn=custom_init),
        make_activation("relu"),
        make_dense(hidden_dim2, num_classes, weight_init_fn=custom_init),
    ]

    model = make_sequential(layers)
    loss_fn = make_loss("cross_entropy")
    optimizer = make_optimizer(model["params"], lr=0.05, kind="adam")

    # Train model
    train(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        x=x_train,
        y=y_train,
        epochs=150,
        batch_size=32,
        seed=seed,
    )

    # Evaluate accuracy on eval set
    eval_logits, _ = model["forward"](x_eval)
    preds = np.argmax(eval_logits, axis=1)
    acc = float(np.mean(preds == y_eval))

    metrics = {
        "accuracy": acc,
        "x": x_eval,
        "y": y_eval,
    }

    return model, metrics

# Step 13 - improve_generalization
import copy
import numpy as np


def improve_generalization(
    baseline_model_fn, x_train, y_train, x_val, y_val, seed=0
):
    """Improve held-out accuracy over an unregularized baseline.

    Inputs:
      baseline_model_fn: zero-arg callable -> fresh untrained sequential model
        (dict with 'forward', 'backward', 'params') matching the data dims.
      x_train, y_train: training features (N, D) and int labels (N,).
      x_val, y_val: validation features (N_val, D) and int labels (N_val,).
      seed: int for deterministic training.

    Returns:
      dict with keys:
        'val_accuracy': float accuracy of the improved model on x_val/y_val
        'baseline_val_accuracy': float val accuracy of plain unregularized SGD
        'predictions': np.ndarray shape (N_val,) int preds from improved model
        'model': the trained improved model
    """
    loss_fn = make_loss("cross_entropy")

    # -------------------------------------------------------------
    # 1. Baseline Model: Plain unregularized SGD prone to overfitting
    # -------------------------------------------------------------
    baseline_model = baseline_model_fn()
    base_opt = make_optimizer(baseline_model["params"], lr=0.05, kind="sgd")

    train(
        model=baseline_model,
        loss_fn=loss_fn,
        optimizer=base_opt,
        x=x_train,
        y=y_train,
        epochs=120,
        batch_size=16,
        seed=seed,
    )

    base_logits, _ = baseline_model["forward"](x_val)
    base_preds = np.argmax(base_logits, axis=1)
    baseline_val_acc = float(np.mean(base_preds == y_val))

    # -------------------------------------------------------------
    # 2. Improved Model: L2 weight decay + early stopping checkpointing
    # -------------------------------------------------------------
    improved_model = baseline_model_fn()
    improved_opt = make_optimizer(
        improved_model["params"], lr=0.01, kind="adam"
    )

    weight_decay = 1e-3
    n_samples = x_train.shape[0]
    batch_size = 32
    epochs = 150
    rng = np.random.RandomState(seed)

    best_val_acc = -1.0
    best_weights = None

    for epoch in range(epochs):
        perm = rng.permutation(n_samples)

        for start_idx in range(0, n_samples, batch_size):
            batch_idx = perm[start_idx : start_idx + batch_size]
            xb = x_train[batch_idx]
            yb = y_train[batch_idx]

            # Forward pass
            logits, caches = improved_model["forward"](xb)
            _, d_logits = loss_fn(logits, yb)

            # Backward pass
            _, grads = improved_model["backward"](d_logits, caches)

            # Apply L2 weight decay to 2D weight matrices (excluding 1D biases)
            for p_dict, g_dict in zip(improved_model["params"], grads):
                if "W" in p_dict and "W" in g_dict:
                    g_dict["W"] += weight_decay * p_dict["W"]

            improved_opt["step"](grads)

        # Track validation accuracy for checkpointing / early stopping
        val_logits, _ = improved_model["forward"](x_val)
        val_acc = float(np.mean(np.argmax(val_logits, axis=1) == y_val))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = [
                {k: v.copy() for k, v in layer_p.items()}
                for layer_p in improved_model["params"]
            ]

    # Restore best checkpoint weights
    if best_weights is not None:
        for target_p, src_p in zip(improved_model["params"], best_weights):
            for k, v in src_p.items():
                np.copyto(target_p[k], v)

    final_val_logits, _ = improved_model["forward"](x_val)
    predictions = np.argmax(final_val_logits, axis=1)
    val_accuracy = float(np.mean(predictions == y_val))

    return {
        "val_accuracy": val_accuracy,
        "baseline_val_accuracy": baseline_val_acc,
        "predictions": predictions,
        "model": improved_model,
    }

