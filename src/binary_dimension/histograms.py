import numpy as np
import jax.numpy as jnp
import warnings
from scipy.spatial.distance import pdist
from .utils import pairwise_hamming_scan


def compute_histogram(S): 
    '''
    Compute the histogram of Hamming distances of the data using a matrix product 
    of the data.

    Input: 
    S : shape(N_samples,N) is the data of the set of spins (Xs[i,j] = {-1,+1})
    
    Returns: 
    r : array with all the Hamming distances computed in the data
    P : array with the probability associated to each distance r
    '''
    if not isinstance(S, np.ndarray):
        raise ValueError("Input S must be a numpy array.")
    N_samples,N = S.shape
    dis = N*pdist(S,metric='hamming')
    r,P = np.unique(dis,return_counts=True)
    P = P/np.sum(P)
    return r.astype(int),P



def compute_histogram_jax(S): 
    """
    Compute the histogram of Hamming distances of the data using JAX
    Input:
    S : shape(N_samples,N) is the data of the set of spins (S[i,j] = {-1,+1})
    Returns:
    r : array with all the Hamming distances computed in the data
    P : array with the probability associated to each distance r
    """
    if not isinstance(S, jnp.ndarray):
        raise ValueError("Input S must be a jax.numpy array.")
    N_samples,N = S.shape
    dis = pairwise_hamming_scan(S)
    r , P = jnp.unique(dis,return_counts=True)
    P /= np.sum(P)
    return np.array(r).astype(int) , np.array(P)



def select_range(r, Pr, a_min=0.0, a_max=1.0):
    """
    Return the sub-range of r and corresponding normalized probabilities Pr
    whose cumulative probability lies within [a_min, a_max].

    Parameters
    ----------
    r : array-like, shape (M,)
        Sorted distances or bins.
    Pr : array-like, shape (M,)
        Corresponding (non-negative) probabilities or counts.
    a_min, a_max : float in [0, 1]
        Cumulative probability window. a_min < a_max required.

    Returns
    -------
    new_r, new_Pr : numpy.ndarray
        Sliced r and normalized Pr over the requested cumulative interval.
        If the interval is empty returns two empty 1D arrays.
    """
    r = np.asarray(r)
    Pr = np.asarray(Pr, dtype=float)

    if r.ndim != 1 or Pr.ndim != 1 or r.shape[0] != Pr.shape[0]:
        raise ValueError("r and Pr must be 1D arrays of the same length")

    # Clamp and validate a_min / a_max
    if a_min < 0.0:
        warnings.warn("a_min < 0, clamping to 0.0")
        a_min = 0.0
    if a_max > 1.0:
        warnings.warn("a_max > 1, clamping to 1.0")
        a_max = 1.0
    if not (a_min < a_max):
        raise ValueError("a_min must be strictly less than a_max")

    total = Pr.sum()
    if total == 0:
        raise ValueError("Pr sums to zero")

    # Fixed-point shortcut from original implementation
    if r.size > 0 and r[0] == 0 and int(np.argmax(Pr)) == 0:
        # ensure normalized
        return r, Pr / total

    # Normalized cumulative distribution used for index selection
    cPr = np.cumsum(Pr) / total

    # Find inclusive slice indices for the requested cumulative window
    ind_min = np.searchsorted(cPr, a_min, side="left")
    ind_max = np.searchsorted(cPr, a_max, side="right") - 1
    ind_max = min(ind_max, r.size - 1)

    if ind_min > ind_max:
        # Empty interval
        return np.empty(0, dtype=r.dtype), np.empty(0, dtype=Pr.dtype)

    new_r = r[ind_min : ind_max + 1]
    new_Pr = Pr[ind_min : ind_max + 1]
    new_Pr = new_Pr / new_Pr.sum()

    return new_r, new_Pr


def highest_prob_range(r, Pemp, a_tot = 1.0):
    """
    Select the smallest set of r-values whose total empirical probability
    (summing the largest Pemp values first) reaches at least `a_tot`.

    Returns
    -------
    rx : ndarray
        Selected r-values sorted in ascending order.
    Px : ndarray
        Corresponding probabilities normalized to sum to 1.

    Notes
    -----
    - If a_tot <= 0 returns empty arrays.
    - If a_tot >= 1 returns the full r and normalized Pemp.
    - Raises ValueError for input shape mismatches or zero total probability.
    """
    r = np.asarray(r)
    Pemp = np.asarray(Pemp, dtype=float)

    if r.shape != Pemp.shape:
        raise ValueError("r and Pemp must have the same shape")

    total = Pemp.sum()
    if total == 0:
        raise ValueError("Pemp sums to zero")

    # Trivial cases
    if a_tot <= 0:
        return np.empty(0, dtype=r.dtype), np.empty(0, dtype=Pemp.dtype)
    if a_tot >= 1:
        return r.copy(), Pemp / total

    # Sort indices by descending probability
    desc_idx = np.argsort(Pemp)[::-1]
    P_desc = Pemp[desc_idx]

    # Cumulative in descending-prob order and find minimal prefix reaching a_tot
    cdesc = np.cumsum(P_desc) / total
    k = np.searchsorted(cdesc, a_tot, side="left")
    # select up to and including index k
    sel_idx = desc_idx[: k + 1]

    # Extract and order by r ascending
    sel_r = r[sel_idx]
    sel_P = Pemp[sel_idx]
    order = np.argsort(sel_r)
    rx = sel_r[order]
    Px = sel_P[order]

    # Normalize selected probabilities
    Px = Px / Px.sum()

    return rx, Px