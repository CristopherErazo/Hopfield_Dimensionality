import numpy as np
from spin_sampler.utils import prob_plus , get_progress_bar
from line_profiler import profile
from spin_sampler.dispatch import DISPATCHER
import jax.numpy as jnp
import jax.lax as lax
import jax
import warnings
from numba.core.errors import NumbaDebugInfoWarning

# Suppress NumbaDebugInfoWarning
warnings.filterwarnings("ignore", category=NumbaDebugInfoWarning)


class Sampler:
    '''
    Gibbs Sampler for spin system with different modes.

    Attributes:
    -----------
    - J (np.ndarray): Coupling matrix/matrices.
    - T (float): Temperature.
    - mode (str): Sampling mode ('single_chain', 'multi_chain', 'multi_couplings').
    - backend: Backend to use ('numpy', 'numba', 'jax').

    '''


    def __init__(self, J: np.ndarray, T: float, mode: str = 'single_chain',backend: str = 'numpy'):
        '''
        Initialize the Gibbs sampler.

        Parameters:
        -----------
        - J: Coupling matrix (shape (N, N) or (N_walkers, N, N)). Must be symmetric for each chain.
        - T: Temperature.
        - mode: Sampling mode ('single_chain', 'multi_chain', 'multi_couplings').
        - backend: Backend to use ('numpy', 'numba', 'jax').

        '''

        avail_modes = ['single_chain', 'multi_chain', 'multi_couplings']
        avail_backends = ['numpy', 'numba', 'jax']
              
        if not isinstance(T, (int, float)):
            raise TypeError('T must be a number.')
        if mode not in avail_modes:
            raise ValueError(f"Invalid mode. Choose from {avail_modes}.")
        if backend not in avail_backends:
            raise ValueError(f"Invalid backend. Choose from {avail_backends}.")
        
        # Check dimensions
        if mode == 'single_chain' and J.ndim != 2:
            raise ValueError("For 'single_chain', J must be 2D.")
        if mode == 'multi_chain' and J.ndim != 2:
            raise ValueError("For 'multi_chain', J must be 2D.")
        if mode == 'multi_couplings' and J.ndim != 3:
            raise ValueError("For 'multi_couplings', J must be 3D.")

        # Check symmetry
        if backend == 'jax':
            if not isinstance(J, jnp.ndarray):
                raise TypeError('J must be a jax array.')  
            if mode == 'single_chain' or mode == 'multi_chain':
                if not jnp.allclose(J, J.T, atol=1e-8):
                    raise ValueError('J must be symmetric.')
            if mode == 'multi_couplings':
                if not jnp.allclose(J, jnp.swapaxes(J, -1, -2), atol=1e-8):
                    raise ValueError('Each coupling matrix in J must be symmetric.')
        else:
            if not isinstance(J, np.ndarray):
                raise TypeError('J must be a numpy array.')  
            if mode == 'single_chain' or mode == 'multi_chain':
                if not np.allclose(J, J.T, atol=1e-8):
                    raise ValueError('J must be symmetric.')
            if mode == 'multi_couplings':
                if not np.allclose(J, np.swapaxes(J, -1, -2), atol=1e-8):
                    raise ValueError('Each coupling matrix in J must be symmetric.')
    
        self.J = J
        self.T = T
        self.mode = mode   
        self.backend = backend 
        self.gibbs_step = DISPATCHER[backend][mode]
        self.chain = []  # To store sampled states if needed
        self.spin_dtype = jnp.int8 if backend == 'jax' else np.int8
        self.float_dtype = jnp.float32 if backend == 'jax' else np.float64


    @profile
    def sample(self, initial_state, N_samples = 1, dt_samples = 1, rnd_ord=True, seed=None, store=False , progress = False):
        """
        Run Gibbs sampling as a generator.

        Parameters:
        -----------
        - initial_state: Initial spin configuration. If None, use the last state in chain
        - N_samples: Total number of steps to perform.
        - dt_samples: Save every 'dt_samples' steps to reduce correlation.
        - rnd_ord: If True, update spins in random order.
        - seed: Optional seed for random number generation (int) / mandatory for jax.
        - store: If True, store the sampled states in memory.
        - progress: If True, display a progress bar.

        Yields:
        ----------
        - The sampled state at each step (after thinning).
        """
    
        # Validate initial_state
        if self.backend == 'jax':
            if not isinstance(initial_state, jnp.ndarray):
                raise TypeError('initial state must be a jax array.') 
            if seed == None:
                raise ValueError('Must specify seed when using jax.')
        else: 
            if not isinstance(initial_state, np.ndarray):
                raise TypeError('initial_state must be a numpy array.')
            
        if self.mode == 'single_chain' and initial_state.ndim != 1:
            raise ValueError("For 'single_chain', initial_state must be 1D.")
        if self.mode in ['multi_chain', 'multi_couplings'] and initial_state.ndim != 2:
            raise ValueError(f"For '{self.mode}', initial_state must be 2D.")
        if self.mode == 'single_chain' and initial_state.shape[0] != self.J.shape[0]:
            raise ValueError("Initial state size does not match J dimensions.")
        if self.mode == 'multi_chain' and initial_state.shape[1] != self.J.shape[0]:
            raise ValueError("Initial state size does not match J dimensions.")
        if self.mode == 'multi_couplings' and (initial_state.shape[0] != self.J.shape[0] or initial_state.shape[1] != self.J.shape[1]):
            raise ValueError("Initial state size does not match J dimensions.")

        # Initialize random seed
        if seed is not None:
            if self.backend == 'jax': 
                key = jax.random.PRNGKey(seed)
            else:
                np.random.seed(seed)
                key = 42
        else:
            key = 42  # Dummy key for non-jax backends

        # Check if chain is empty, if not, print a warning
        if (not len(self.chain) == 0) and store:
            print(f"Warning: The chain is not empty, contains {len(self.chain)} elements. New samples will be appended to the existing chain.")

        S = initial_state.copy()
        pbar = get_progress_bar(progress, N_samples)

        # Main sampling loop
        for step in range(N_samples*dt_samples):
            S , key = self.step(S,rnd_ord=rnd_ord,key=key)  # Perform one Gibbs sampling step
            # Save or yield the state every `thin_by` steps
            if step % dt_samples == 0:
                if store:
                    self.chain.append((S.astype(self.spin_dtype)).copy())
                yield S.copy()
            pbar.update(1)
        pbar.close()
    


    def run_gibbs(self, initial_state,  N_samples = 1, dt_samples = 1, rnd_ord=True, seed=None, store=False , progress=False):
        """
        Iterate function 'sample' for 'N_samples' iterations and return the final state.

        Parameters
        ----------
        - initial_state: Initial spin configuration. If None, use the last state in chain 
        - N_samples: Total number of steps to perform.
        - dt_samples: Save every 'dt_samples' steps to reduce correlation.
        - rnd_ord: If True, update spins in random order.
        - seed: Optional seed for random number generation (int) / mandatory for jax.
        - store: If True, store the sampled states in memory.
        - progress: If True, display a progress bar.

        Returns
        -------
        np.ndarray
            The final sampled state after ``nsteps`` iterations.
        """

        if initial_state is None:
            if len(self.chain) == 0:
                raise ValueError("Cannot have `initial_state=None` if run_gibbs has never been called.")
            else:
                # Use last state in chain and remove it to avoid duplicates
                initial_state = self.chain.pop()
                initial_state = initial_state.astype(self.float_dtype)

        results = None
        for results in self.sample(initial_state, N_samples=N_samples,dt_samples=dt_samples,rnd_ord=rnd_ord,seed=seed,store=store,progress=progress):
            pass  # iterate through generator until last sample

        return results




    def step(self,S,rnd_ord=True,key=None):
        """
        Perform one Gibbs sampling step of the state S.

        Parameters:
        -----------
        - S: Current spin configuration.

        """
        S , key = self.gibbs_step(S, self.J, self.T,rnd_ord,key)
        return S , key
    
    def get_chain(self):
        """
        Get the stored chain of sampled states.

        Returns:
        ----------
        - Array of sampled states.
        """
        if self.mode == 'single_chain':
            if self.backend == 'jax':
                return jnp.array(self.chain)
            else:
                return np.array(self.chain)   
        else:
            if self.backend == 'jax':
                return jnp.swapaxes(jnp.array(self.chain),0,1)
            else:
                return np.swapaxes(np.array(self.chain),0,1)
    
    def reset_chain(self):
        """
        Reset the stored chain of sampled states.
        """
        self.chain = []
       

