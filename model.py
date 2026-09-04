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

# Step 2 - gradient_check (not yet solved)
# TODO: implement

# Step 3 - make_dense (not yet solved)
# TODO: implement

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

