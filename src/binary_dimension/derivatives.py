from scipy.special import psi
import numpy as np
from .distance_models import DistanceModel


class DKL_Computations:
    """
    Compute value, gradient and Hessian of D_KL(Pemp || P_model(theta)) for a given DistanceModel.
    """
    def __init__(self, distance_model):
        """
        Initialize the DKL computations with a distance model.
        Parameters
        ----------
        distance_model : DistanceModel
            An instance of DistanceModel to compute distances and probabilities.
        """
        self.model: DistanceModel = distance_model

        if not isinstance(self.model, DistanceModel):
            raise ValueError("distance_model must be an instance of DistanceModel.")


    def _S(self, r, theta, lmbda):
        """
        Compute auxiliary function S(r) = sum_{l=0}^{r-1} 1 / (d(r) - l)**lmbda for each entry in r.

        Parameters
        ----------
        r : array-like of non-negative integers
            Distances / bin indices.
        theta : array-like
            Parameters passed to d_model.
        lmbda : float
            Exponent.

        Returns
        -------
        ndarray, shape (len(r),)
            Values of the sum for each r. For r == 0 the value is 0.
        """

        d_vals = self.model.d(r, theta)

        r = np.asarray(r, dtype=int)
        d_vals = np.asarray(d_vals, dtype=float)

        # Fast analytic form for lmbda == 1 using digamma:
        if lmbda == 1:
            # sum_{l=0}^{r-1} 1/(d - l) = psi(d + 1) - psi(d - r + 1)
            return psi(d_vals + 1.0) - psi(d_vals - r + 1.0)

        # General lmbda: vectorized sum by broadcasting
        max_r = int(r[-1])
        if max_r == 0:
            return np.zeros_like(r, dtype=float)

        L = np.arange(max_r, dtype=float)                 # shape (max_r,)
        # shape (n, 1) - shape (1, max_r) -> (n, max_r)
        denom = d_vals[:, None] - L[None, :]
        mask = (L[None, :] < r[:, None])                  # True where l < r_i
        terms = np.zeros_like(denom, dtype=float)
        terms[mask] = denom[mask] ** (-lmbda)
        Svals = terms.sum(axis=1)
        return Svals


    def DKL(self,Pemp,Pmod): 
        '''
        Kullback-Leibler divergence DKL(Pemp || Pmod).

        Inputs: 
        Pemp : empirical distribution 
        Pmod : model 
        Both of them must have the same lenght. 

        Returns: 
        Dkl : value of the Kullback-Leibler divergence Dkl(Pemp||Pmod)
        '''
        idx = np.where(Pmod != 0)
        Pe = Pemp[idx]
        Pm = Pmod[idx]
        # Compute the relative entropy elementwise with the distributions 
        rel = Pe*np.log(Pe/Pm)
        # We will mask the invalid values in case we obtain Nan of Infinite values
        masked = np.ma.masked_invalid(rel).compressed()
        # Sum the values to obtain the Dkl
        Dkl = np.sum(masked)
        return Dkl

    def grad_DKL(self, r, Pemp, theta):
        """
        Compute the gradient of D_KL(Pemp || P_model(theta)) with respect to theta.
        Parameters
        ----------
        r : array-like, shape (n,)
            Distances / bin indices.
        Pemp : array-like, shape (n,)
            Empirical probability distribution.
        theta : array-like, shape (p,)
            Parameters at which to evaluate the gradient.
        Returns
        -------
        grad_Dkl : ndarray, shape (p,)
            Gradient of D_KL with respect to theta evaluated at the given parameters.
        """
        Pmod = self.model.P_model(r,theta)
        Svals = self._S(r, theta,1)
        g_d = self.model.grad(r, theta)   # (n, p)
        aux = (Pemp - Pmod) * (np.log(2.0) - Svals)    # (n,)
        grad_r = aux[:, None] * g_d                      # (n, p)
        return np.sum(grad_r, axis=0)


    def hessian_DKL(self, r, Pemp, theta, return_inv=False):
            Pmod = self.model.P_model(r, theta)                       # (n,)
            grad_d = self.model.grad(r, theta)                  # (n,p)
            outer_d = self.model.outer_grad(r, theta)          # (n,p,p)
            hessian_d = self.model.hessian(r, theta)              # (n,p,p)
            # Auxiliary S variables
            S1 = self._S(r, theta,lmbda=1)    # (n,)
            S2 = self._S(r, theta,lmbda=2)    # (n,)
            # Auxiliary 'scalar' variables
            aux_logZ = Pmod * (S1 - np.log(2.0))          # (n,)
            aux_outer = Pmod * (np.log(2.0) - S1)**2 + (Pemp - Pmod) * S2   # (n,)
            aux_hess = (Pemp - Pmod) * (np.log(2.0) - S1)          # (n,)
            # Compute gradient of logZ
            grad_logZ = np.sum(aux_logZ[:,None] * grad_d , axis=0)        # (p,)
            # Compute Hessian
            hessian = aux_outer[:,None,None] * outer_d
            hessian += aux_hess[:,None,None] * hessian_d
            hessian = np.sum(hessian, axis=0)        # (p,p)
            hessian -= np.outer(grad_logZ, grad_logZ)    # (p,p)

            if return_inv:
                # Compute the inverse
                det = np.linalg.det(hessian)
                if det == 0: 
                    inv_hess = np.identity(len(theta))
                else: 
                    inv_hess = np.linalg.inv(hessian)
                return hessian,inv_hess
            else:
                return hessian