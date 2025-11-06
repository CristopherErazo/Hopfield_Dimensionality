import numpy as np
import time
from scipy.special import gammaln
from scipy.special import polygamma
from scipy.special import psi
from scipy.spatial.distance import pdist
import jax.numpy as jnp
import jax
from scipy.signal import find_peaks

# Activation function
def phi(X,name = 'tanh'):
    '''
    Compute the postactivations of X elementwise
    '''
    if name == 'tanh':
        z = np.tanh(X)
    if name == 'relu':
        z = X * (X > 0)
    return z

def white_data(X):
    '''
    Removes the mean and variace from the data

    Input: 
    X : shape = (N_samples,N)
    Output: 
    X_white = (X-av)/st where
    av: shape(N) -> mean of each column
    st: shape(N) -> std of each column
    '''
    #Remove the mean and variance from the data
    av = np.mean(X,axis = 0)
    st = np.std(X,axis = 0)
    X_white = (X - av)/st
    return X_white,av,st

def center_data(X):
    '''
    Removes the mean from the data

    Input: 
    X : shape = (N_samples,N)
    Output: 
    X_cent = (X-av) where
    av: shape(N) -> mean of each column
    '''
    #Remove the mean and variance from the data
    av = np.mean(X,axis = 0)
    X_cent = (X - av)
    return X_cent,av


def dim_PCA(X,name ='tanh',Dspin = None, percent=0.95, Eig = False):
    '''
    Computes the PCA estimate of the ID for the pre(X) and post(P) activations 

    Input: 
    X: shape(N_samples,N) -> data
    name='tanh' : defines the activation function
    Dspin : if not None - check the % of covariance explained at Dspin
    Percent = 0.95: Used to compute the number of principal components necessary 
    Eig = False : if True returns also the eigenvalues of X and Phi
    
    Output: 
    Dim_X : PCA dimension for X estimated with participation ratio
    Dim_P : PCA dimension for P = phi(X,name) estimated with participation ratio
    frX/N : fraction of covariance of X explained with Dim_X
    frP/N : fraction of covariance of Phi explained with Dim_P
    frS_X/N : fraction of covariance of X explained with Dspin (zero if Dspin = None)
    frS_P/N : fraction of covariance of Phi explained with Dspin (zero if Dspin = None)
    Dim_X_percent : PCA dimension for X estimated with percentage of variance explained
    Dim_P_percent : PCA dimension for P estimated with percentage of variance explained

    If Eig = True: 
        Eig_X : eigenvalues of X covariance in descending order
        Eig_P : eigencalues of Phi covariance in descending order
    
    
    '''
    #Compute the rate phi(X)
    P = phi(X,name)
    #White the data
    X_white,_ = center_data(X)
    P_white,_ = center_data(P)
    N_samples,N = X.shape
    #print(N_samples)
    #Compute the covariance and diagonalize
    Cov_X = (1/N_samples)*np.matmul(X_white.transpose(),X_white)
    Eig_X,_ = np.linalg.eigh(Cov_X)
    Cov_P = (1/N_samples)*np.matmul(P_white.transpose(),P_white)
    #print(Cov_X.shape,Cov_P.shape)
    Eig_P,_ = np.linalg.eigh(Cov_P)
    Eig_X= Eig_X[::-1]
    Cum_X = np.cumsum(Eig_X)
    Cum_X = Cum_X/Cum_X[-1]
    Eig_P = Eig_P[::-1]
    Cum_P = np.cumsum(Eig_P)
    Cum_P = Cum_P/Cum_P[-1]
    # print(Eig_X)
    # print(Cum_X)
    #Compute the dimension as the participation ratio
    sX2 = np.sum(Eig_X**2)
    sP2 = np.sum(Eig_P**2)
    if sX2*sP2 > 0:
        Dim_X = (np.sum(Eig_X))**2/(sX2)
        Dim_P = (np.sum(Eig_P))**2/(sP2)
    else:
        Dim_X , Dim_P = 0. , 0.


    # Compute the percentage explained
    dX = int(Dim_X)
    frX = Cum_X[dX-1] + (Dim_X - dX)*(Cum_X[dX]-Cum_X[dX-1])

    dP = int(Dim_P)
    frP = Cum_P[dP-1] + (Dim_P - dP)*(Cum_P[dP]-Cum_P[dP-1])
    frS_X , frS_P = 0. , 0.
    if Dspin is not None: 
        dS = int(Dspin)
        frS_X = Cum_X[dS-1] + (Dspin - dS)*(Cum_X[dS]-Cum_X[dS-1])
        frS_P = Cum_P[dS-1] + (Dspin - dS)*(Cum_P[dS]-Cum_P[dS-1])
    # Compute the dimension with percentage of variance
    idx = np.where(Cum_X>percent)[0][0]
    Dim_X_percent = idx + (percent-Cum_X[idx-1])/(Cum_X[idx]-Cum_X[idx-1])
    idx = np.where(Cum_P>percent)[0][0]
    Dim_P_percent = idx + (percent-Cum_P[idx-1])/(Cum_P[idx]-Cum_P[idx-1])
    if Eig == False: 
        return Dim_X,Dim_P,frX,frP,frS_X,frS_P,Dim_X_percent,Dim_P_percent
    else: 
        return Dim_X,Dim_P,frX,frP,frS_X,frS_P,Dim_X_percent,Dim_P_percent,Eig_X,Eig_P




#################################################
## BINARY INTRINSIC DIMENSION
#################################################


def compute_histogram(Xs): 
    '''
    Compute the histogram of Hamming distances of the data using a matrix product

    Input: 
    Xs : shape(N_samples,N) is the data of the set of spins (Xs[i,j] = {-1,+1})
    
    Returns: 
    r : array with all the Hamming distances computed in the data
    P : array with the probability associated to each distance r
    '''
    # N_samples,N = Xs.shape
    # dots = np.matmul(Xs,Xs.transpose())
    # dots = dots[~np.eye(N_samples,dtype=bool)].reshape(N_samples,-1)
    # dot,P = np.unique(dots,return_counts=True)
    # P = P/np.sum(P)
    # r = 0.5*(N-dot)
    # return r[::-1].astype(int),P[::-1]

    N_samples,N = Xs.shape

    dis = N*pdist(Xs,metric='hamming')
    r,P = np.unique(dis,return_counts=True)
    P = P/np.sum(P)
    return r.astype(int),P


def hamming(x, y):
    return jnp.sum(x != y)

def pairwise_hamming_vmap(S):
    # S: shape (N_samples, N)
    # vmap over the first argument, then the second
    return jax.vmap(lambda x: jax.vmap(lambda y: hamming(x, y))(S))(S)

def pairwise_hamming_scan(S):
    N_samples , N = S.shape
    def scan_fn(carry, x):
        distances = jax.vmap(lambda y: hamming(x, y))(S)
        return carry, distances

    _, result = jax.lax.scan(scan_fn, None, S)
    result = result[~jnp.eye(N_samples,dtype=bool)].reshape(N_samples,-1)
    return result

def compute_histogram_jax(S): 
    N_samples,N = S.shape
    dis = pairwise_hamming_scan(S)
    r , P = jnp.unique(dis,return_counts=True)
    P /= np.sum(P)#N_samples*(N_samples-1)
    return np.array(r).astype(int) , np.array(P)


def select_range(r,Pr,a_min,a_max): 
    '''
    Change the interval of the distribution to a personalized one given by (a_min,a_max)

    Inputs: 
    r : array of Hamming distances
    Pr : array of corresponding probabilities
    a_min : the values for which Prob{r<rmin} < a_min will be cut
    a_max : the values for which Prob{r<rmax} > a_max will be cut

    Return: 
    new_r : array of Hamming distances between rmin and rmax
    new_Pr : corresponding probabilities normalized in the new interval

    If a_min = 0 and a_max = 1 the returned variables will be the same as the inputs
    '''

    
    # Simple checks
    if a_min >= a_max: 
        print('Error a_min > a_max, setting range (0,1)')
        a_min = 0.0
        a_max = 1.0
    if a_min < 0 : 
        print('Error a_min < 0, setting = 0')
        a_min = 0.0
    if a_max > 1 : 
        print('Error a_max > 1, setting = 1')
        a_max = 1.0 

    if r[0] == 0 and np.argmax(Pr)== 0 : #Fixed point alert
        new_r = r
        new_Pr = Pr
    else:

        # Compute the cumulative distribution
        cPr = np.cumsum(Pr)
        # Obtain the min and max indexes that defines the new interval
        ind_min = np.where(cPr >= a_min)[0][0]
        if cPr[0] >= a_max: 
            ind_max = len(r)-1
        else: 
            ind_max = np.where(cPr <= a_max)[0][-1]
        # Evaluate the new variables in the reduced interval
        new_r = r[ind_min:ind_max]
        new_Pr = Pr[ind_min:ind_max]
        # Normalize the new probability distribution
        new_Pr = new_Pr / np.sum(new_Pr)
    return new_r,new_Pr

def select_range_prob(r,Pr,beta): 
    id_mod = np.where(Pr == np.max(Pr))[0][0]
    mod_r = r[id_mod]
    cum = np.cumsum(Pr)
    p0 = cum[id_mod]
    p_up = p0 + beta*(p0)
    # p_down = p0 - beta*(1-p0)
    p_down = p0*(1-beta)
    idx_min= np.where(cum >= p_down)[0][0]
    idx_max= np.where(cum <= p_up)[0][-1]
    rx = r[idx_min:idx_max]
    Px = Pr[idx_min:idx_max]
    Px = Px/np.sum(Px)
    return rx,Px

def select_range_symmetric(r,Pr,N,alpha): 
    '''
    Select range as N/2(1 +/- alpha) with alpha = (0,1) 
    '''
    rmin = 0.5*N*(1-alpha)
    rmax = 0.5*N*(1+alpha)
    if rmax >= r[-1]: 
        rmax = r[-1]
    if rmin <= r[0]: 
        rmin = r[0]
    if rmin >= r[-1]: 
        rmin = r[0]
        rmax = r[-1]
    
    idx_min= np.where(r >= rmin)[0][0]
    idx_max= np.where(r <= rmax)[0][-1]
    rx = r[idx_min:idx_max]
    Px = Pr[idx_min:idx_max]
    Px = Px/np.sum(Px)
    return rx,Px


def highest_prob_range(r,Pemp,alpha): 
    '''
    Select range with highest probability accounting for 
    probability = alpha.
    '''
    # Sort the distribution by highest probability and compute the cumulative
    Pord = np.sort(Pemp)[::-1]
    idx = np.argsort(Pemp)[::-1]
    Cord = np.cumsum(Pord)
    # Compute the index that reaches the threshold of probability
    wh = np.where(Cord > alpha)[0]
    if len(wh) == 0 : 
        idn = -1
    else: 
        idn = np.where(Cord > alpha)[0][0]
    # Evaluate the range and probability until that limit
    rx_dis = r[idx[:idn]]
    Px_dis = Pemp[idx[:idn]]
    # Order the distribution in ascending order of the r-values
    rx = np.sort(rx_dis)
    id_or = np.argsort(rx_dis)
    Px = Px_dis[id_or]
    Px = Px/np.sum(Px)
    return rx,Px
    
##################################################
## CODE WITH NUMPY
# DERIVATIVES IMPLEMENTED EXPLICITLY 
#################################################

def d(r,theta):
    '''Evaluate the function d(r) = th0 + th1*r'''
    th0 = theta[0]
    th1 = theta[1]
    return th0 + th1*r

def grad_d(r,theta): 
    '''
    Evaluate the gradient of d(r) with respect to theta in the model d(r) = th0 + th1 * r
    Therefore the gradient is grad d(r) = (1,r)

    Inputs: 
    r : array of Hamming distances
    theta : parameters [th0,th1]

    Output: 
    grad_d : gradient of d(r) with respect to theta -> shape = (len(r) , 2)
    '''
    # Create the gradient with ones in the first component and the values of r in the second.
    grad_d = np.array([np.ones(shape=r.shape),r]).transpose()
    return grad_d

def P_model(r,theta): 
    '''
    Compute the model P(r|theta) in the range given by r.
    
    Inputs: 
    r : array of Hamming distances
    theta :  = [th0 , th1] parameters of the model

    Returns:
    P_mod : Theoretical distriburion evaluated with the given parameters 
    len(P_mod) = len(r)
    '''
    # First compute the logarithm of P with the log(Gamma) function
    logP = - d(r,theta)*np.log(2)+ gammaln(d(r,theta) + 1) - gammaln(d(r,theta)-r + 1) - gammaln(r + 1)
    # Remove mean 
    # logP = logP - np.mean(logP)
    # Clip the values
    # logP = np.clip(logP,-20,20)
    # Take the exponential of the log
    P_mod = np.exp(logP)
    # Normalize the distribution
    P_mod = P_mod/np.sum(P_mod)
    return P_mod



    
def DKL(Pemp,Pmod): 
    '''
    Compute the Kullback-Leibler divergence between the empirical distribution and the model. 

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

def S(r,theta,alpha): 
    '''
    Evaluate the auxiliar function S(r) = S^{alpha}_{theta}(r) = Sum_{l=0}^{r-1} 1/(d(r)-l)^{alpha} with d(r) = th0 + th1 * r

    Inputs: 
    r: array of Hamming distances
    theta: parameters [th0,th1]
    alpha: exponent

    Returns: 
    S : function evaluated at each r (len(S)=len(r))
    '''
    if alpha == 1:
        Dr = d(r,theta)
        S = psi(-Dr) - psi(r-Dr)
    else: 
        # S = polygamma(1,-Dr) - polygamma(1,r-Dr)
        # Initialize an empty list
        S = []
        # If the minimum distance is 0 then S = 0
        # print(r)
        if r[0] == 0:
            R = r[1:]
            S.append(0)
        else: 
            R = r
        # Compute S(r) for each r
        for rit in R: 
            L = np.arange(rit)
            Dr = d(rit,theta)
            S.append(np.sum(1/(Dr-L)**alpha))
        # Convert S to array
        S = np.array(S)
    return S


def Grad_DKL(r,Pemp,Pmod,theta): 
    '''
    Compute the gradient of the Dkl and evaluate it in the given parameters theta = [th0,th1]
    grad_Dkl = [dDkl/dth0,dDkl/dth1]

    Inputs: 
    r : array of Hamming distances 
    Pemp : empirical distribution 
    Pmod : model 
    theta: parameters [th0,th1]
    
    Returns: grad_Dkl shape = (len(theta)=2)
    '''
    # Compute the auxiliar variable S(r)
    S_alpha = S(r,theta,1)
    # Compute the gradient of d(r)
    G_d = grad_d(r,theta)
    # Evaluate the expression inside of the sum for each r before multiplying for grad_d = (1,r)
    aux = (Pemp-Pmod)*(np.log(2) - S_alpha)
    # Compute each element of the sum before computing the gradient
    # We need to multiply elementwise the aux variable with the gradiend G_d
    # and to do that we need to reshape because up to this point we have: 
    # aux.shape=(len(r),) and G_d.shape = (len(r),2) 
    grad_r = aux.reshape(len(r),1)*G_d
    # Once we multiply columnwise we have grad_r.shape = (len(r),2)
    # and we sum over the first axis to obtain the gradient with grad_Dkl.shape = (2)
    grad_Dkl = np.sum(grad_r,axis=0)
    return grad_Dkl



def Grad2(r):#,G,theta): 
    '''
    Compute the out product of the gradient of d(r) with respect to the parameters 
    
    Input: 
    G : grad_d(r) = (1,r) with shape = (len(r),2)
    theta : parameters [th0,th1]

    Output: 
    G2 = G (*) G.transpose = [[1,r][r,r²]] with shape  = (len(r),2,2)
    '''
    # Initialize an empty list
    # G2 = []
    # # Run over the gradient at each r 
    # for i in range(G.shape[0]):
    #     # Reshape to turn it into a column
    #     Gi = G[i].reshape(len(theta),1)
    #     # Multiply the column by the row to obtain a matrix and save
    #     G2.append( np.matmul( Gi,Gi.transpose() ) )
    # Convert it into an array
    # G2 = np.array(G2)
    G2 = np.array([[np.ones(len(r)),r],[r,r**2]]).transpose(2,0,1)
    return G2
    

def Hessian_DKL(r,Pemp,Pmod,theta): 
    '''
    Compute the Hessian and inverse Hessian of the Dkl evaluated at the given parameters th0,th1
    Hess_Dkl = [[d²Dkl/dth0²,d²Dkl/dth0dth1],[d²Dkl/dth1dth0,d²Dkl/dth1²]]

    Inputs: 
    r : array of Hamming distances 
    Pemp : empirical distribution 
    Pmod : model 
    theta: parameters [th0,th1]
    
    Returns: 
    Hess : Hessian
    Inv_Hess : Inverse Hessian

    Both are matrices (2x2) in this case (d(r) = th0 + th1 * r)
    ''' 
 
    # Variables needed
    G = grad_d(r,theta) 
    # Gradient of d(r) = G = (1,r) -- G.shape(len(r),2)
    G2 = Grad2(r)
    # Outer product of G = G2 = G (*) G.transpose = ((1,r),(r,r²)) -- G2.shape(len(r),2,2)
    S1 = S(r,theta,1)    # Auxiliar variable S1(r) -- alpha = 1 -- len(S1) = len(r)
    S2 = S(r,theta,2)    # Auxiliar variable S2(r) -- alpha = 2 -- len(S2) = len(r)
    # Compute the first term in the sum
    
    # Compute the term inside the sum without the matrix part 
    aux = Pmod*((np.log(2)-S1)**2) + (Pemp-Pmod)*S2
    # Right now aux has shape=len(r) and we need to multiply
    # slice by slice to the matrix G2 to have the first term
    # of the Hessian at each component r
    H1_r = aux.reshape(len(r),1,1)*G2 # We reshape to multiply slice by slice H1_r.shape = (len(r),2,2)
    # We sum over all the terms in r (axis=0) and we obtain a sigle matrix
    H1 = np.sum(H1_r,axis=0)  # H1.shape = (2,2)
    # The shapes are (2,2) because there are only 2 parameters, this might change in general
    
    # Compute the second term with the grad_logZ

    # Compute the term inside the sum without the matrix part 
    aux = Pmod*(S1-np.log(2))
    # Right now aux has shape=len(r) and we need to multiply
    # slice by slice to the vector G to have the grad_logZ at each r
    g_logZ_r = aux.reshape(len(r),1)*G # We reshape to multiply column by column g_logZ_r.shape = (len(r),2)
    # We sum over all the terms in r (axis=0) and we obtain a sigle vector
    g_logZ = np.sum(g_logZ_r,axis=0).reshape(len(theta),1) # g_logZ.shape = (2,1)
    # We compute the outer product to obtain a matrix
    logZ2 = np.matmul(g_logZ,g_logZ.transpose()) # logZ2.shape = (2,2)

    # The part proportional to the second derivatives of d(r) is zero in this case

    # Sum of the parts
    Hess = H1 - logZ2
    # Compute the inverse
    det = np.linalg.det(Hess)
    if det == 0: 
        InvHess = np.identity(len(theta))
    else: 
        InvHess = np.linalg.inv(Hess)
    return Hess,InvHess


#################################
### OPTIMIZATION
#################################

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


def initial(r,Pemp,it): 
    '''
    Estimates a good initial condition for the parameters theta = [th0,th1]
    sampling at random values of th1 between 0 and 2 and setting th0 = N(1-th1/2)
    Selects the set of parameters that minimizes the DKL

    Input: 
    r: array of Hamming distances
    Pemp: empirical distribution len(Pemp) = len(r)
    it: number of iterations performed to pick the best one as initial condition

    Return: 
    theta0 : the parameter that minimizes the DKL among the random sample
    to be used as the initial condition in the optimization process. 
    '''
    # Make th1 random between 0 and 2
    TH1 = 2*np.random.random(it)
    # Compute empirical average of r
    av_r = np.sum(r*Pemp)
    # Empty list to save logDKL
    logD = []
    for th1 in TH1: 
        # Compute th0 and create the parameter theta0
        th0 = (2-th1)*av_r
        theta0 = np.array([th0,th1])
        # Evaluate the model with the given parameters
        Pmod = P_model(r,theta0)
        # Compute and save the logDKL of the given realization of parameters
        logD.append(np.log(DKL(Pemp,Pmod)))
    # Select the index in which we find the minimum logDKL
    ind = logD.index(min(logD))
    # Create the optimal initial condition and return theta0
    th1 = TH1[ind]
    th0 = (2-th1)*av_r
    theta0 = np.array([th0,th1])
    return theta0

def grad_descent_lr(r,Pemp,theta0,eps,lr): 
    '''
    This algorithm performs the gradient descent to fit the model P(r|theta) with the data
    using a fixed learning rate. 

    Inputs: 
    r: array of Hamming distances
    Pemp: empirical distribution len(Pemp) = len(r)
    theta0: initial condition for the parameters (must be selected appropriately)
    eps: desired precision - stop criteria when |theta(t+1) - theta(t)| < eps
    lr: learning rate (fixed)

    Returns: 
    th : array of all the parameters in each iteration -> shape = (Nit,2)
    Div : array of all the DKL in each iteration -> shape = (Nit)
    Nit : total number of steps performed to achieve the desired precision
    '''
    # Initializations
    Pmod = P_model(r,theta0)
    theta = theta0
    dkl = DKL(Pemp,Pmod)
    # Save the parameters and value of DKL over the iterations
    th = [theta]
    Div = [dkl]
    # Initialize the error and counter    
    err  = 100
    Nit = 0    
    while err > eps: 
        # Update the counter
        Nit = Nit + 1
        # Compute the gradient
        G = Grad_DKL(r,Pemp,Pmod,theta)
        # print(G)
        # Compute the velocity vector
        v = lr * G / dkl
        # Update the parameter and error
        theta = theta - v
        err = np.sum(v**2)
        # Compute the updated model and save the parameters and DKL
        Pmod = P_model(r,theta)
        th.append(theta)
        dkl = DKL(Pemp,Pmod)
        Div.append(dkl)
        if Nit > 2000: #Avoid extremely long computations
            print('No convergence! Nit>2000')
            break       
    return np.array(th),np.array(Div),Nit    


def grad_descent_hessian(r,Pemp,theta0,eps,lr): 
    '''
    This algorithm performs the gradient descent to fit the model P(r|theta) with the data
    using the inverse Hessian as adaptive lerning rate.

    Inputs: 
    r: array of Hamming distances
    Pemp: empirical distribution len(Pemp) = len(r)
    theta0: initial condition for the parameters (must be selected appropriately)
    eps: desired precision - stop criteria when |theta(t+1) - theta(t)| < eps

    Returns: 
    th : array of all the parameters in each iteration -> shape = (Nit,2)
    Div : array of all the DKL in each iteration -> shape = (Nit)
    Nit : total number of steps performed to achieve the desired precision
    '''
    # Initializations
    Pmod = P_model(r,theta0)S
    theta = theta0
    # Save the parameters and value of DKL over the iterations
    th = [theta]
    Div = [DKL(Pemp,Pmod)]
    # Initialize the error and counter    
    err  = 100
    Nit = 0    
    while err > eps: 
        # Update the counter
        Nit = Nit + 1
        # Compute the gradient
        G = Grad_DKL(r,Pemp,Pmod,theta)
        # Compute the inverse Hessian
        _,invH = Hessian_DKL(r,Pemp,Pmod,theta)
        # Compute the velocity vector
        v = lr*np.matmul(invH,G)
        # Update the parameter and error
        theta = theta - v
        err = np.sum(v**2)
        # Compute the updated model and save the parameters and DKL
        Pmod = P_model(r,theta)
        th.append(theta)
        Div.append(DKL(Pemp,Pmod))
        if Nit > 2000: #Avoid extremely long computations
            print('No convergence! Nit>2000')
            break       
    return np.array(th),np.array(Div),Nit   



def evaluate_log_dkl(r,Pemp,min0 , max0, min1 , max1 , N0, N1):
    th0 = np.linspace(min0 , max0, N0)
    th1 = np.linspace(min1 , max1, N1)
    logD = np.zeros(shape=(N0,N1))
    for i in range(N0): 
        for j in range(N1): 
            theta = [th0[i],th1[j]]
            Pmod = P_model(r,theta)
            Dij = DKL(Pemp,Pmod)
            logD[i,j] = np.log(Dij)
    return th0,th1,logD



def Minimize_Dkl(r,Pemp,N,eps=1e-7,show=False,lr=1):
    '''
    Optimizes the DKL to find the best parameters. 

    Input: 
    r: full array of Hamming distances
    Pemp: corresponding empirical probabilities
    eps = 1e-7 : defines the precision of the optimization
    show = False: If true it prints the optimal theta, logDkl and N_steps

    Returns: 
    theta : optimal parameters
    log : value of log(Dkl)
    N_steps : number of optimization steps performed
    '''

    it = 25
    cond = False
    nchecks = 0
    # The conditions ensures that the Optimization algorithm does not pick a 
    # extrange solution or overfloat, in case of missmatch the optimization is 
    # repeated with different conditons. 

    # The first initial condition is with the gaussian approximation
    theta0 = initial_gauss(r,Pemp)

    th, Dkl, Nit = grad_descent_hessian(r,Pemp,theta0,eps,lr)

    theta = th[-1]
    DK = Dkl[-1]
    # print(theta,DK)
    cond = DK >  0 and theta[0]>0 and theta[0]<N and theta[1]>0 and theta[1]<2

    while cond == False: 
        # If the gaussian approximation fails we select
        # random initial conditions with some constrain (see initial())
        theta0 = initial(r,Pemp,it)
        th, Dkl, Nit = grad_descent_hessian(r,Pemp,theta0,eps,lr)
        theta = th[-1]
        DK = Dkl[-1]
        # log = np.log(Dkl[-1])
        # The condition ensures logDKL < 0, th0 = (0,N) and th1 = (0,2)
        cond = DK >  0 and theta[0]>0 and theta[0]<N and theta[1]>0 and theta[1]<2
        nchecks += 1
        if nchecks > 10: 
            print(f'To many checks no fit possible... output theta=[-N,-2] - logdkl = 10 - Nit = -1')
            return [-N,-2],10.,-1    #return this values just to avoid problems and keep track of bad fits
    # Print if show = True
    if show == True: 
        print(f'Optimal theta = ({theta[0]:.4},{theta[1]:.4})  logDkl = {np.log(DK):.4}   N_steps = {Nit}')
    # The first component of the parameter is the estimate of the dimension.
    return theta,np.log(DK),Nit



def fixed_point_flags(r,P,threshold=0.75): 
    is_fixed_point = False
    fit_possible = True
    if r[np.argmax(P)] < 5 : is_fixed_point = True
    if r[0] == 0 and P[0]> threshold: fit_possible=False
    return is_fixed_point , fit_possible

def range_check_multi_modal(r,P,prominance = 0.05): 
    peaks = find_peaks(P, prominence=prominance * max(P))[0]
    if len(peaks) == 1: # No multi modal
        return r.copy() , P.copy()
    else:
        # print(r)
        # print(P)
        idx = np.where(r >= np.mean(r[peaks][:2]))[0][0]-1
        rx = r.copy()[:idx]
        Px = P.copy()[:idx]
        Px /= np.sum(Px)
        return rx , Px


def select_and_fit(r,P,N,show=True,threshold=0.75,prominance=0.05,beta=0.95,lr=1,a_min=1e-3,a_max=0.2,range='automatic'):

    is_fixed_point , fit_possible = fixed_point_flags(r,P,threshold=threshold)
 
    rx , Px = r.copy() , P.copy()
    if is_fixed_point: #Potential fixed point
        if fit_possible: #Try to fit
            theta , logkl , Nit = Minimize_Dkl(r,P,N,show=show,lr=lr)
            x_param = [*theta , logkl,Nit]
        if not fit_possible or Nit < 0: #if fit not possible or fit not converge -> fix x_params
            #print('here1')
            x_param = [0. , 0. , -30,-1]

    else: # No fixed point found
        # ra , Pa = bid.highest_prob_range(r,P,0.9) # Remove noise
        if range == 'automatic':
            ra , Pa = range_check_multi_modal(r,P,prominance = prominance) # select range
            rx , Px = select_range_prob(ra,Pa,beta)
        if range == 'alphas':
            rx , Px = select_range(r,P,a_min,a_max)
            
        theta , logkl , Nit = Minimize_Dkl(rx,Px,N,show=show,lr=lr)
        x_param = [*theta , logkl,Nit]

    if x_param[-1] < 0:
        #print('here2')
        x_param = [np.nan,np.nan,np.nan,np.nan]
    return np.array(x_param) , rx , Px





##################################################
########### Computation of the dimensions ########
##################################################




def compute_dimensions(X,alpha=0.75,percent=0.95): 
    '''
    Compute the dimension estimates using PCA and BID estimators
    X --> data matrix shape(Ns,N) with Ns samples 
    alpha = 0.25 --> defines the symmetric interval to evaluate the model for Hamming Distances
                    [r_min,r_max]=N/2[1-alpha,1+alpha]
    percent = 0.95 --> desired percentage of covariance to be explained in order to compute PCA dimension estimate
    Return: 
    dim_pca = [Dim_X,Dim_P,Dim_X_per,Dim_P_per]
        * Dim_X , Dim_P --> pca dimension estimates with participation ratio
        * Dim_X_per , Dim_P_per --> pca dimension estimates with percentage of variance = percent
    dim_bid = [theta[0],theta[1],logKL,Nsteps]
        * theta[0] = Dspin --> bid estimate, zero'th order parameter in model d(r) = theta_0 + theta_1*r
        * theta[1] --> first order parameter in model d(r) = theta_0 + theta_1*r
        * logKL --> Kullback Leibler divergence at the optimal parameter after optimization
        * Nsteps --> Number of optimization steps until convergence
    frac = [frX,frP,frS_X,frS_P]
        * frX --> Fraction of X variance explained with Dim_X principal components
        * frP --> Fraction of P variance explained with Dim_P principal components
        * frS_X --> Fraction of X variance explained with Dspin principal components
        * frS_P --> Fraction of P variance explained with Dspin principal components
    eigen = [Eig_X,Eig_P] 
        * Eigenvalue spectrum for both X and P covariances
    '''
    N = X.shape[1]
    Xs = np.sign(X - np.mean(X,axis=0))
    r, Pemp = compute_histogram(Xs)
    if Pemp[0] >= 0.3:  #Fixed point
        print(f'Fixed Point')
        theta = [0.,0.]
        logKL = 0.
        Nsteps = -2
        Dim_X,Dim_P,frX,frP,frS_X,frS_P,Dim_X_per,Dim_P_per,Eig_X,Eig_P = 0.,0.,1.,1.,1.,1.,0.,0.,np.zeros(N),np.zeros(N)
    else: 
        rx , Px = highest_prob_range(r,Pemp,alpha)
        theta,logKL,Nsteps = Minimize_Dkl(rx,Px,N,show=False)
        Dspin = theta[0]
        Dim_X,Dim_P,frX,frP,frS_X,frS_P,Dim_X_per,Dim_P_per,Eig_X,Eig_P = dim_PCA(X,Dspin = Dspin,percent=percent,Eig=True)

    dim_pca = [Dim_X,Dim_P,Dim_X_per,Dim_P_per]
    dim_bid = [theta[0],theta[1],logKL,Nsteps]
    frac = [frX,frP,frS_X,frS_P]
    eigen = [Eig_X,Eig_P]
    return dim_pca , dim_bid , frac , eigen