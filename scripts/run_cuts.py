import argparse, time
from spin_sampler import Sampler , define_hopfield_model , initialize_spins
from binary_dimension import compute_histogram_jax , select_range
from binary_dimension.utils import initial_gauss
from binary_dimension.optimization import DKL_Optimizer
from binary_dimension.distance_models import LinearDistanceModel 

from configurations import make_params_dict, make_data_paths
import jax
import jax.numpy as jnp
import os
import multiprocessing
import numpy as np


def order_params_jax(patterns, S):
    '''
    Compute the order parameters: magnetization and spin-glass parameter
    '''
    _, N = S.shape
    N, _ = patterns.shape
    av_s = jnp.mean(S, axis=0)
    q = 1 / N * jnp.sum(av_s ** 2)
    m = 1 / N * av_s @ patterns
    m = jnp.sort(jnp.abs(m))[::-1]

    return m, q

def fixed_point_flags(r,P,threshold=0.75): 
    is_fixed_point = False
    fit_possible = True
    if r[np.argmax(P)] < 5 : is_fixed_point = True
    if r[0] == 0 and P[0]> threshold: fit_possible=False
    return is_fixed_point , fit_possible



def main():
    print(f'-------Starting job on host: {os.uname().nodename}-----')
    print("Backend:", jax.default_backend())
    print("Devices:", jax.devices())
    print("OMP_NUM_THREADS =", os.environ.get("OMP_NUM_THREADS"))
    print("XLA_FLAGS =", os.environ.get("XLA_FLAGS"))
    print("CPU count (visible to process):", multiprocessing.cpu_count())

    # Define parameters of the system
    parser = argparse.ArgumentParser(description='Test the spin sampler.')
    parser.add_argument('--N', type=int, default=100, help='Number of spins')
    parser.add_argument('--T', type=float, default=1.0, help='Temperature')
    parser.add_argument('--alpha', type=float, default=0.04, help='Load parameter (only for Hopfield model)')
    parser.add_argument('--N_walkers', type=int, default=1, help='Number of parallel chains')
    parser.add_argument('--mode', type=str, default='single_chain', help="Sampling mode: 'single_chain', 'multi_chain' or 'multi_couplings'")
    parser.add_argument('--backend', type=str, default='jax', help="Backend: 'numpy', 'numba' or 'jax'")
    parser.add_argument('--N_samples', type=int, default=500, help='Number of samples to draw')
    parser.add_argument('--dt_samples', type=int, default=1, help='Number of Monte Carlo steps between samples (reduce time correlation)')
    parser.add_argument('--progress', type=str, default='True', help='Show a progress bar (tqdm required)')
    parser.add_argument('--h', type=float, default=0.9, help='Initial magnetization along the reference pattern')
    parser.add_argument('--burnin', type=int, default=800, help='Number of burn-in samples to discard')
    parser.add_argument('--iteration', type=int, default=0, help='Iteration index for repeated runs')

    args = parser.parse_args()
    print(args)
    N = args.N
    T = args.T
    alpha = args.alpha
    N_walkers = args.N_walkers
    mode = args.mode
    backend = args.backend
    N_samples = args.N_samples
    dt_samples = args.dt_samples
    progress = args.progress
    h = args.h
    burnin = args.burnin
    progress = args.progress == 'True'
    seed = int(time.time())
    rnd_ord = True
    it = args.iteration
    seed += it # Different seed for each iteration

    

    t0 = time.time()
    # Define the couplings of Hopfield model and initialize spins
    p = int(alpha * N)
    J , patterns = define_hopfield_model(N,p,N_walkers,mode,backend,seed)
    print(f'Defined Hopfield model with seed = {seed}, N={N}, p={p}, alpha={alpha}')
    # print(f'{J.shape = }, {patterns.shape = }.')
    initial_state = initialize_spins(N,N_walkers,mode,backend,seed=seed,config='magnetized',ref_spin=patterns[:,0],m0=h)

    # Create a sampler instance and sample
    sampler = Sampler(J, T, mode,backend)
    # Run burnin period
    final = sampler.run_gibbs(initial_state,burnin,1,seed=seed,progress=progress)
    # Run sampling
    sampler.run_gibbs(final,N_samples,dt_samples,seed=seed,store=True,progress=progress);
    S = sampler.get_chain()
    sampler.reset_chain() # Clean up the sampler
    print(f'Chain shape: {S.shape}') # (N_samples,N)

    # Compute hopfield order parameters
    M, q = order_params_jax(patterns, S)
    m = M[0]
    print(f'Order parameters: m={m}, q={q}')
    
    # Compute Histogram of Hamming distances and dimensionality
    r , P = compute_histogram_jax(S)

    mean_r = jnp.sum(r*P)
    mean_r2 = jnp.sum(r**2*P)
    var_r = mean_r2 - mean_r**2
    
    # x = 1-2r/N

    mean_x = 1-2*mean_r/N
    var_x = 4*var_r/N**2

    is_fixed_point , fit_possible = fixed_point_flags(r,P)
 
    if is_fixed_point: #Potential fixed point
        if fit_possible: #Try to fit
            theta_init = initial_gauss(r,P)
            distance_model = LinearDistanceModel()
            Dkl_optimizer = DKL_Optimizer(r,P, distance_model)
            theta_opt , log_DKL , Nit = Dkl_optimizer.optimize(theta_init)
            
        else: #No fit possible, set fix point values
            theta_opt = np.array([0. , 0.])
            log_DKL = -50
            Nit = 1
    else: #Not a fixed point, perform cut and fit only on the left region

        rx , Px = select_range(r,P,0,0.3)
        theta_init = initial_gauss(r,P)
        distance_model = LinearDistanceModel()
        Dkl_optimizer = DKL_Optimizer(rx,Px, distance_model)
        theta_opt , log_DKL , Nit = Dkl_optimizer.optimize(theta_init)

    # Results and save
    results = [m,q,*theta_opt,log_DKL,Nit,mean_x,var_x]

    # Parameters to save
    names_fixed = ['N_samples','burnin','h','alpha']
    names_variable = ['N','T']
    params = make_params_dict(names_fixed,names_variable)
    file_path , filename , dir_path = make_data_paths('summary', experiment_name= 'fss', params=params,base_dir='./data',ext='txt')

    with open(file_path,'a') as f:
        f.write(' '.join(map(str,results))+'\n')

    names_fixed = ['N_samples','burnin','h','alpha']
    names_variable = ['N','T','it']
    params = make_params_dict(names_fixed,names_variable)
    file_path , filename , dir_path = make_data_paths('configurations', experiment_name= 'fss_matrices', params=params,base_dir='./data',ext='txt')

    # Save the Spin configurations and patterns
    jnp.savez(file_path, S=S, patterns=patterns)


    dt = time.time() - t0
    print(f'------TIME TAKEN  = {dt/60 :.5} min = {dt/3600 :.3} hours-------')


if __name__ == "__main__":
    main()
