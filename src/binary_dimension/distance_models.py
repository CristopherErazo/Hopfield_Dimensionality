import numpy as np
from scipy.special import gammaln
from abc import ABC, abstractmethod

class DistanceModel(ABC):
    """Abstract interface for distance models d(r; theta)."""
    @abstractmethod
    def d(self, r, theta):
        pass

    @abstractmethod
    def grad(self, r, theta):
        """Return array shape (n, p) of ∂d/∂theta_j evaluated at each r."""
        pass

    @abstractmethod
    def hessian(self, r, theta):
        """Return array shape (n, p, p) of ∂²d/∂theta_j∂theta_k at each r."""
        pass

    def P_model(self, r, theta):
        """Return normalized model probability vector P(r | theta)."""
        dvals = self.d(r, theta)
        r_arr = np.asarray(r, dtype=float)
        logP = -dvals * np.log(2.0)
        logP += gammaln(dvals + 1.0)
        logP -= gammaln(dvals - r_arr + 1.0)
        logP -= gammaln(r_arr + 1.0)
        logP -= np.max(logP)   # stability
        P = np.exp(logP)
        return P / np.sum(P)       

    def outer_grad(self, r, theta):
        """Optional: return (n, p, p) outer products; default builds from grad()."""
        g = self.grad(r, theta)   # (n, p)
        return np.einsum('ni,nj->nij', g, g)

class LinearDistanceModel(DistanceModel):
    """d(r) = th0 + th1 * r"""

    def d(self, r, theta):
        th0, th1 = theta[0], theta[1]
        return th0 + th1 * np.asarray(r)

    def grad(self, r, theta=None):
        r = np.asarray(r)
        return np.column_stack((np.ones_like(r), r))
    
    def hessian(self, r, theta=None):
        n = len(r)
        return np.zeros((n, 2, 2))
    
class PolynomialDistanceModel(DistanceModel):
    """d(r) = th0 + th1 * r + th2 * r^2 + ... + th(p) * r^(p)"""

    def __init__(self, degree , N):
        self.degree = degree
        self.N = N

    def d(self, r, theta):
        r = np.asarray(r)
        if len(theta) != self.degree + 1:
            raise ValueError("Theta length must be degree + 1")
        dvals = np.zeros_like(r, dtype=float)
        for j, th in enumerate(theta):
            dvals += th * ((r/self.N) ** j)
        return dvals*self.N
    
    def grad(self, r, theta=None):
        r = np.asarray(r)
        p = self.degree + 1
        grads = np.column_stack([(r/self.N) ** j for j in range(p)])
        return self.N*grads

    def hessian(self, r, theta=None):
        n = len(r)
        p = len(theta) if theta is not None else 2
        return np.zeros((n, p, p))