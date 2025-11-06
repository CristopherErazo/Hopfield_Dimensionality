import jax.numpy as jnp
import numpy as np
import jax

def hamming(x, y):
    return jnp.sum(x != y)

def pairwise_hamming_vmap(S):
    """Compute pairwise Hamming distances using JAX vmap."""
    return jax.vmap(lambda x: jax.vmap(lambda y: hamming(x, y))(S))(S)

def pairwise_hamming_scan(S):
    """Compute pairwise Hamming distances using JAX scan."""
    N_samples , N = S.shape
    def scan_fn(carry, x):
        distances = jax.vmap(lambda y: hamming(x, y))(S)
        return carry, distances

    _, result = jax.lax.scan(scan_fn, None, S)
    result = result[~jnp.eye(N_samples,dtype=bool)].reshape(N_samples,-1)
    return result

def initial_gauss(r,Pemp): 
    '''
    Estimates the initial value of the parameters using the mean 
    and variance of the empirical distribution and the Gaussian
    model 
    Input: 
    r: array of Hamming distances
    Pemp: empirical distribution len(Pemp) = len(r)

    Return: 
    theta0 : the parameter found with Gaussian Approximation
    '''
    # Compute empirical mean and variance
    r0 = np.sum(r*Pemp)
    sig2 = np.sum(Pemp*(r-r0)**2)
    # Estimate the parameter with Gaussian aproximation
    th1 = 2-np.sqrt(2*r0/sig2)
    th0 = r0*(2-th1)
    theta0 = np.array([th0,th1])
    return theta0