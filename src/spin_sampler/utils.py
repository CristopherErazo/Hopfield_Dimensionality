import numpy as np
import os
import yaml
import jax.numpy as jnp
from numba import njit

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

def prob_plus(x):
    """
    Computes the probability p_plus = 1 / (1 + exp(-2x)) in a numerically stable way.

    Parameters:
    - x: The input value or array.

    Returns:
    - p_plus: The computed probability, same shape as x.
    """
    x = np.asarray(x)  # Ensure x is a NumPy array for vectorized operations
    ex = np.exp(-2 * np.abs(x))  # Compute exp(-2|x|) to ensure stability
    p_plus = 0.5 * (1 + np.sign(x) * (1 - ex) / (1 + ex))
    return p_plus

@njit
def prob_plus_numba(x):
    """
    Computes the probability p_plus = 1 / (1 + exp(-2x)) in a numerically stable way.

    Parameters:
    - x: The input value or array.

    Returns:
    - p_plus: The computed probability, same shape as x.
    """
    x = np.asarray(x)  # Ensure x is a NumPy array for vectorized operations
    ex = np.exp(-2 * np.abs(x))  # Compute exp(-2|x|) to ensure stability
    p_plus = 0.5 * (1 + np.sign(x) * (1 - ex) / (1 + ex))
    return p_plus

def prob_plus_jax(x):
    """
    Computes the probability p_plus = 1 / (1 + exp(-2x)) in a numerically stable way.

    Parameters:
    - x: The input value or array.

    Returns:
    - p_plus: The computed probability, same shape as x.
    """
    # x = np.asarray(x)  # Ensure x is a NumPy array for vectorized operations
    ex = jnp.exp(-2 * jnp.abs(x))  # Compute exp(-2|x|) to ensure stability
    p_plus = 0.5 * (1 + jnp.sign(x) * (1 - ex) / (1 + ex))
    return p_plus


class _NoOpPBar:
    """A no-operation progress bar that does nothing."""
    def __init__(self, *args, **kwargs):
        pass

    def update(self, count):
        pass

    def close(self):
        pass


def get_progress_bar(enabled, total):
    """
    Return a progress bar if enabled, otherwise return a no-op progress bar.

    Parameters:
    ----------
    - enabled: bool, whether to display the progress bar.
    - total: int, the total number of iterations.

    Returns:
    ----------
    - A progress bar object.
    """
    if enabled and tqdm is not None:
        return tqdm(total=total)
    return _NoOpPBar()



def read_config(file):
	filepath = os.path.abspath(f'{file}.yaml')
	print(filepath)
	with open(filepath,	'r') as stream:
		kwargs = yaml.safe_load(stream)
	return kwargs


