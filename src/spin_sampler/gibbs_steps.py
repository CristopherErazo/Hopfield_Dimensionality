import numpy as np
from spin_sampler.utils import prob_plus , prob_plus_jax
import jax.numpy as jnp
import jax.lax as lax
import jax
from jax import jit
from line_profiler import profile

# Data types
type_spins_np = np.int8
type_reals_np = np.float32



# NUMPY FUNCTIONS
@profile
def gibbs_step_single_chain(S, J, T, rnd_ord = True , key = None):
    """
    One full update of spin state using Gibbs sampling for 
    single chain with J.

    Parameters:
    -----------
    - S: Spin configuration shape (N,).
    - J: Coupling matrix (shape (N, N)).
    - T: Temperature.
    - rnd_ord: If True, update spins in random order.
    - key: Dummy variable for compatibility with JAX functions.

    Returns:
    --------
    - Updated spin configuration and dummy variable.
    """
    beta = 1 / T  
    N = len(S)

    # Define the update order
    idx = np.random.permutation(N) if rnd_ord else np.arange(N)

    # Update sequentially the spins
    for i in idx:
        h_i = S @ J[i]
        p_plus = prob_plus(beta * h_i)
        rand_vals = np.random.rand()
        S[i] = 1 if rand_vals < p_plus else -1
    return S , key

@profile
def gibbs_step_multi_chain(S, J, T, rnd_ord = True,  key = None):
    """
    One full update of spin state using Gibbs sampling for 
    multiple chains with same J.

    Parameters:
    -----------
    - S: Spin configuration shape (N_walkers,N).
    - J: Coupling matrix (shape (N, N)).
    - T: Temperature.
    - rnd_ord: If True, update spins in random order.
    - key: Dummy variable for compatibility with JAX functions.

    Returns:
    --------
    - Updated spin configuration and dummy variable.
    """
    beta = 1 / T  
    N_walkers , N = S.shape

    # Define the update order
    idx = np.random.permutation(N) if rnd_ord else np.arange(N)

    # Update sequentially the spins
    for i in idx:
        h_i = S @ J[i]
        p_plus = prob_plus(beta * h_i)
        rand_vals = np.random.rand(N_walkers)
        S[:, i] = np.where(rand_vals < p_plus, 1, -1)
    return S , key

@profile
def gibbs_step_multi_couplings(S, J, T, rnd_ord = True , key = None):
    """
    One full update of spin state using Gibbs sampling for 
    multiple chains with different J.

    Parameters:
    -----------
    - S: Spin configuration (shape (N_walkers, N)).
    - J: Coupling matrices (shape (N_walkers, N, N)).
    - T: Temperature.
    - rnd_ord: If True, update spins in random order.
    - key: Dummy variable for compatibility with JAX functions.

    Returns:
    --------
    - Updated spin configuration and dummy variable.
    """
    beta = 1 / T  
    N_walkers, N = S.shape

    # Define the update order
    idx = np.random.permutation(N) if rnd_ord else np.arange(N)

    # Update sequentially the spins
    for i in idx:
        h_i = np.sum(J[:, i] * S, axis=1)
        p_plus = prob_plus(beta * h_i)
        rand_vals = np.random.rand(N_walkers)
        S[:, i] = np.where(rand_vals < p_plus, 1, -1)
    return S  , key


# JAX FUNCTIONS



@jit
def gibbs_step_single_chain_jax(S , J , T , rnd_ord = False, key = None):
    """
    One full update of spin state using Gibbs sampling for 
    single chain with J.

    Parameters:
    -----------
    - S: Spin configuration shape (N,).
    - J: Coupling matrix (shape (N, N)).
    - T: Temperature.
    - rnd_ord: If True, update spins in random order.
    - key: PRNG key for randomness.

    Returns:
    --------
    - Updated spin configuration and new PRNG key.
    """
    N = S.shape[0]
    beta = 1 / T
    key, subkey = jax.random.split(key)
    idx = lax.cond(
        rnd_ord,
        lambda _: jax.random.permutation(subkey, N),  # If rnd_ord is True
        lambda _: jnp.arange(N),                     # If rnd_ord is False
        operand=None
    )

    def update_spin(i, val):
        S , key = val
        id = idx[i]
        key, subkey = jax.random.split(key)
        # Compute local field for spin i
        # h_i = S@J[id]
        h_i=jnp.dot(J[id],S)
        # Compute probability P(si = +1)
        p_plus = prob_plus_jax(beta * h_i)  #1 / (1 + jnp.exp(-beta_2 * h_i))
        # Update spin based on probabilities
        new_spin =  jax.random.choice(subkey,jnp.array([1, -1]),
                                    p = jnp.array([p_plus, 1-p_plus]))
        S = S.at[id].set(new_spin)
        return S , key

    S , key = lax.fori_loop(0, N, update_spin, (S , key))
    return S, key

@jit
def gibbs_step_multi_chain_jax(S , J , T , rnd_ord = False, key = None):
    """
    One full update of spin state using Gibbs sampling for 
    multiple chains with same J.

    Parameters:
    -----------
    - S: Spin configuration shape (N_walkers,N).
    - J: Coupling matrix (shape (N, N)).
    - T: Temperature.
    - rnd_ord: If True, update spins in random order.
    - key: PRNG key for randomness.

    Returns:
    --------
    - Updated spin configuration and new PRNG key.
    """

    N_walkers , N = S.shape
    beta = 1 / T
    key, subkey = jax.random.split(key)
    idx = lax.cond(
        rnd_ord,
        lambda _: jax.random.permutation(subkey, N),  # If rnd_ord is True
        lambda _: jnp.arange(N),                     # If rnd_ord is False
        operand=None
    )

    def update_spin(i, val):
        S , key = val
        id = idx[i]
        key, subkey = jax.random.split(key)
        # Compute local field for spin i
        h_i = S @ J[id] # shape = (N_walkers)
        # Compute probability P(si = +1)
        p_plus = prob_plus_jax(beta * h_i) #1 / (1 + jnp.exp(-beta_2 * h_i))
        # Generate random numbers for all samples
        rand_vals = jax.random.uniform(subkey, shape=(N_walkers,))
        # Update spins based on probabilities
        new_spin = jnp.where(rand_vals < p_plus, 1, -1)
        S = S.at[:,id].set(new_spin)
        return S , key

    S , key = lax.fori_loop(0, N, update_spin, (S , key))
    return S, key

# @profile
@jit
def gibbs_step_multi_couplings_jax(S , J , T , rnd_ord = False, key = None):
    """
    One full update of spin state using Gibbs sampling for 
    multiple chains with same J.

    Parameters:
    -----------
    - S: Spin configuration (shape (N_walkers, N)).
    - J: Coupling matrices (shape (N_walkers, N, N)).
    - T: Temperature.
    - rnd_ord: If True, update spins in random order.
    - key: PRNG key for randomness.

    Returns:
    --------
    - Updated spin configuration and new PRNG key.
    """
    N_walkers , N = S.shape
    beta = 1 / T

    key, subkey = jax.random.split(key)
    idx = lax.cond(
        rnd_ord,
        lambda _: jax.random.permutation(subkey, N),  # If rnd_ord is True
        lambda _: jnp.arange(N),                     # If rnd_ord is False
        operand=None
    )


    def update_spin(i, val):
        S , key = val
        id = idx[i]
        key, subkey = jax.random.split(key)
        # Compute local field for spin i
        h_i = jnp.sum(J[:,:,id] * S, axis=1) # shape = (N_walkers)
        # Compute probability P(si = +1)
        p_plus = prob_plus_jax(beta * h_i) #1 / (1 + jnp.exp(-beta_2 * h_i))
        # Generate random numbers for all samples
        rand_vals = jax.random.uniform(subkey, shape=(N_walkers,))
        # Update spins based on probabilities
        new_spin = jnp.where(rand_vals < p_plus, 1, -1)
        S = S.at[:,id].set(new_spin)
        return S , key

    S , key = lax.fori_loop(0, N, update_spin, (S , key))
    return S, key

