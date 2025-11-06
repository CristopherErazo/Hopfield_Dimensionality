import numpy as np
import jax.numpy as jnp
import jax

type_spins_jax = jnp.float32
type_reals_jax = jnp.float32

type_spins_np = np.float64
type_reals_np = np.float64

# Define different models

def define_hopfield_model(N , p , N_walkers = 1,mode = 'single_chain', backend = 'numpy', seed=None):
    """
    Defines the Hopfield model patterns and coupling matrix. 
    The patterns are random binary vectors of size N with values in {-1, 1}.
    The coupling matrix J is constructed using the Hebbian learning rule and the patterns
    as J = 1/N * patterns @ patterns.T  with the diagonal set to zero.

    Parameters:
    -----------
    - N: Number of spins or neurons.
    - p: Number of random patterns.
    - N_walkers: Number of chains to run in parallel.
    - mode: Sampling mode ('single_chain', 'multi_chain', 'multi_couplings').
    - backend: Backend to use ('numpy', 'numba', 'jax').
    - seed: Optional seed for random number generation (int) / mandatory for jax.

    Returns:
    ----------
    - J: Coupling matrix (shape (N, N) or (N_walkers,N,N)).   
    - patterns: Generated patterns (shape (N, p) or (N_walkers,N,p)).
    """
    if mode == 'multi_chain': N_walkers = 1

    if mode == 'multi_couplings' and N_walkers == 1:
        raise ValueError('The number of walkers is 1 only in single chain mode')
    if mode != 'multi_couplings' and N_walkers != 1:
        raise ValueError('The number of walkers must be > 1 only in multi couplings mode')
    if backend == 'jax' and seed == None:
        raise ValueError('The seed is mandatory for jax')
    
    # Initialize random seed/ key
    if seed is not None:
        if backend == 'jax': 
            key = jax.random.PRNGKey(seed)
            key, subkey = jax.random.split(key)
        else:
            np.random.seed(seed)
    
    # Define shape of patterns
    shape = (N_walkers,N,p) if mode == 'multi_couplings' else (N,p)



    if backend == 'jax': 
        patterns = jax.random.choice(subkey, jnp.array([-1, 1]), shape=shape).astype(type_reals_jax)
        if mode != 'multi_couplings':
            J = 1/N * (patterns @ patterns.T - p * jnp.eye(N))
        else:
            J = 1/N * jnp.stack([pat @ pat.T - p * jnp.eye(N) for pat in patterns])
    else:
        patterns = np.random.choice([-1, 1], size=shape).astype(type_reals_np)
        if mode != 'multi_couplings':
            J = 1/N * (patterns @ patterns.T - p*np.eye(N))
        else:
            J = 1/N * np.array( [pat @ pat.T - p*np.eye(N) for pat in patterns] )

    return J, patterns

def define_SK_model(N,N_walkers = 1,mode = 'single_chain', backend = 'numpy', seed=None):
    """
    Defines the Sherrington-Kirkpatrick model coupling matrix. 
    The coupling matrix J is symmetric and constructed with entries drawn from a 
    normal distribution with mean 0 and variance 1/N, and the diagonal set to zero.


    Parameters:
    -----------
    - N: Number of spins or neurons.
    - N_walkers: Number of chains to run in parallel.
    - mode: Sampling mode ('single_chain', 'multi_chain', 'multi_couplings').
    - backend: Backend to use ('numpy', 'numba', 'jax').
    - seed: Optional seed for random number generation (int) / mandatory for jax.

    Returns:
    ----------
    - J: Coupling matrix (shape (N, N) or (N_walkers,N,N)).   
    """

    if mode == 'multi_chain': N_walkers = 1

    if mode == 'multi_couplings' and N_walkers == 1:
        raise ValueError('The number of walkers is 1 only in single chain mode')
    if mode != 'multi_couplings' and N_walkers != 1:
        raise ValueError('The number of walkers must be > 1 only in multi couplings mode')
    if backend == 'jax' and seed == None:
        raise ValueError('The seed is mandatory for jax')
    
    # Initialize random seed/ key
    if seed is not None:
        if backend == 'jax': 
            key = jax.random.PRNGKey(seed)
            key, subkey = jax.random.split(key)
        else:
            np.random.seed(seed)
    
    # Define shape of patterns
    shape = (N_walkers,N,N) if mode == 'multi_couplings' else (N,N)

    if backend == 'jax':
        J = jax.random.normal(subkey, shape, dtype=type_reals_jax) / jnp.sqrt(N)
        J = (J + jnp.swapaxes(J, -1, -2)) / jnp.sqrt(2)  # Make symmetric with same variance
        if mode != 'multi_couplings':
            J = J - jnp.diag(jnp.diag(J))
        else:
            J = jnp.array([j - jnp.diag(jnp.diag(j)) for j in J])

    else:
        J = np.random.normal(0, 1/np.sqrt(N), size=shape).astype(type_reals_np)
        J = (J + np.swapaxes(J, -1, -2)) / np.sqrt(2)
        if mode != 'multi_couplings':
            J = J - np.diag(np.diag(J))
        else:
            J = np.array([j - np.diag(np.diag(j)) for j in J])

    return J


def define_random_model(N,N_walkers = 1,mode = 'single_chain', backend = 'numpy', seed=None):
    """
    Defines the a random coupling matrix at with zero mean gaussian entries
    with 1/N variance. Used for testing purposes.


    Parameters:
    -----------
    - N: Number of spins or neurons.
    - N_walkers: Number of chains to run in parallel.
    - mode: Sampling mode ('single_chain', 'multi_chain', 'multi_couplings').
    - backend: Backend to use ('numpy', 'numba', 'jax').
    - seed: Optional seed for random number generation (int) / mandatory for jax.

    Returns:
    ----------
    - J: Coupling matrix (shape (N, N) or (N_walkers,N,N)).   
    """

    if mode == 'multi_chain': N_walkers = 1

    if mode == 'multi_couplings' and N_walkers == 1:
        raise ValueError('The number of walkers is 1 only in single chain mode')
    if mode != 'multi_couplings' and N_walkers != 1:
        raise ValueError('The number of walkers must be > 1 only in multi couplings mode')
    if backend == 'jax' and seed == None:
        raise ValueError('The seed is mandatory for jax')
    
    # Initialize random seed/ key
    if seed is not None:
        if backend == 'jax': 
            key = jax.random.PRNGKey(seed)
            key, subkey = jax.random.split(key)
        else:
            np.random.seed(seed)
    
    # Define shape of patterns
    shape = (N_walkers,N,N) if mode == 'multi_couplings' else (N,N)

    if backend == 'jax':
        J = jax.random.normal(subkey, shape, dtype=type_reals_jax) / jnp.sqrt(N)
    else:
        J = np.random.normal(0, 1/np.sqrt(N), size=shape).astype(type_reals_np)

    return J


# Define different initializations

def initialize_spins(N,N_walkers = 1, mode = 'single_chain', backend = 'numpy', seed=None , config = 'random', ref_spin = None, m0 = None):
    """
    Initialize the spin states depending on the configuration.


    Parameters:
    -----------
    - N: Number of spins or neurons.
    - N_walkers: Number of chains to run in parallel.
    - mode: Sampling mode ('single_chain', 'multi_chain', 'multi_couplings').
    - backend: Backend to use ('numpy', 'numba', 'jax').
    - seed: Optional seed for random number generation (int) / mandatory for jax.
    - config: Initialization configuration ('random', 'magnetized').
    - ref_spin: Reference configuration for magnetized initialization (shape (N,) or (N_walkers,N)).
    - m0: Initial magnetization level for magnetized initialization (float in (0,1) or array of shape (N_walkers,)).

    Returns:
    ----------
    - S0: Initial spin configuration (shape (N) or (N_walkers,N)).   
    """

    if mode != 'single_chain' and N_walkers == 1:
        raise ValueError('The number of walkers is 1 only in single chain mode')
    if mode == 'single_chain' and N_walkers != 1:
        raise ValueError('The number of walkers must be > 1 only in multi couplings mode')
    if backend == 'jax' and seed == None:
        raise ValueError('The seed is mandatory for jax')
    
    shape = (N,) if mode == 'single_chain' else (N_walkers,N)   

    if config not in ['random', 'magnetized']:
        raise ValueError("config must be 'random' or 'magnetized'")
    if config == 'magnetized': 
        if ref_spin is None or m0 is None:
            raise ValueError("For 'magnetized' config, ref_spin and m0 must be provided")
        if ref_spin.shape != shape:
            raise ValueError(f"ref_spin shape must be {shape}")
        
        m0 = m0 if isinstance(m0, (list, np.ndarray)) else [m0]*N_walkers
        m0 = np.array(m0)
        if len(m0) != N_walkers:
            raise ValueError(f"m0 must be a float in (0,1) or an array of shape ({N_walkers},)")
        if np.any(m0 <= 0) or np.any(m0 >= 1):
            raise ValueError("All elements of m0 must be in the interval (0,1)")
        

    # Initialize random seed/ key
    if seed is not None:
        if backend == 'jax': 
            key = jax.random.PRNGKey(seed)
            key, subkey = jax.random.split(key)
        else:
            np.random.seed(seed)
    

    shape = (N_walkers,N)


    if backend == 'jax':
        S0 = jax.random.choice(subkey,jnp.array([1,-1]),shape=shape).astype(type_spins_jax)
    else:
        S0 = np.random.choice([-1, 1], size=shape).astype(type_spins_np)
    
    if config == 'magnetized':
        ref_spins = ref_spin[None,:].copy() if mode == 'single_chain' else ref_spin.copy()

        if backend == 'jax':
            indices = [jax.random.choice(subkey, N, shape=[int(m0[w] * N)], replace=False) for w in range(N_walkers)]
            for w in range(N_walkers):    
                S0 = S0.at[w,indices[w]].set(ref_spins[w,indices[w]])
        else:
            indices = [np.random.choice(N, size=int(m0[w] * N), replace=False) for w in range(N_walkers)]
            for w in range(N_walkers):    
                S0[w,indices[w]] = ref_spins[w,indices[w]]
    
    if mode == 'single_chain':
        S0 = S0[0]
    return S0


