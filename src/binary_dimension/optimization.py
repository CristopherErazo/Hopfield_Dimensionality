import numpy as np
from .derivatives import DKL_Computations
from .distance_models import DistanceModel


class DKL_Optimizer:

    def __init__(self, rx, Px, distance_model):
        """
        Initialize the DKL optimizer with empirical data and DKL computations.
        Parameters
        ----------
        rx : array-like
            Distances / bin indices.
        Px : array-like
            Empirical probabilities corresponding to rx.
        distance_model : DistanceModel
            An instance of DistanceModel to compute distances and probabilities.
        """
        self.dis_model : DistanceModel = distance_model
        self.dkl_comp = DKL_Computations(self.dis_model)
        self.rx = np.asarray(rx)
        self.Px = np.asarray(Px, dtype=float)
        

        if not isinstance(self.dis_model, DistanceModel):
            raise ValueError("distance_model must be an instance of DistanceModel.")
        
    def single_step(self, theta, lr = 1.0):
        """
        Perform a single optimization step using Newton's method.
        Parameters
        ----------
        theta : array-like
            Current parameter estimates.
        lr : float
            Learning rate for the update step.
        Returns
        -------
        theta_new : ndarray
            Updated parameter estimates after one optimization step.
        DKL_value : float
            Value of D_KL at the current parameters.
        """
        theta = np.asarray(theta, dtype=float)
        # Compute gradient and inverse Hessian
        gradDKL = self.dkl_comp.grad_DKL(self.rx, self.Px, theta)
        inv_hessDKL = self.dkl_comp.hessian_DKL(self.rx, self.Px, theta,return_inv=True)[1]
        # Compute Newton's velocity
        velocity = - inv_hessDKL @ gradDKL
        # Update parameters using Newton's method
        theta_new = theta + lr * velocity
        return theta_new
    

    def optimize(self, theta_init, error = 1e-7, max_iter = 20, lr = 1.0):
        """
        Optimize parameters to minimize D_KL using iterative Newton's method.

        Parameters
        ----------
        theta_init : array-like
            Initial parameter estimates.
        error : float
            Convergence threshold for parameter updates.
        max_iter : int
            Maximum number of iterations to perform.
        lr : float
            Learning rate for the update steps.
            
        Returns
        -------
        theta_opt : ndarray
            Optimized parameter estimates.
        log_DKL : float
            Value of D_KL at the optimized parameters.
        """
        theta = theta_init.copy()
        for iteration in range(max_iter):
            theta_new = self.single_step(theta, lr=lr)
            # Check for convergence
            if np.linalg.norm(theta_new - theta) < error:
                break
            theta = theta_new
            Pmod = self.dis_model.P_model(self.rx, theta)
            log_DKL = np.log(self.dkl_comp.DKL(self.Px,Pmod))
            
        theta_opt = theta_new
        Pmod = self.dis_model.P_model(self.rx, theta_opt)
        log_DKL = np.log(self.dkl_comp.DKL(self.Px,Pmod))
        return theta_opt, log_DKL , iteration + 1
    



    