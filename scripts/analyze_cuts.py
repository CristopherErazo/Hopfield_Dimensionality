import argparse
import numpy as np
import jax
from spin_sampler import Sampler , define_hopfield_model , initialize_spins
from binary_dimension import compute_histogram_jax
from configurations import make_params_dict, make_data_paths
import os



if __name__ == "__main__":
    # Define parameters of the system
    parser = argparse.ArgumentParser(description='Test the spin sampler.')
    parser.add_argument('--N', type=int, default=1024, help='Number of spins')
    parser.add_argument('--N_samples', type=int, default=1500, help='Number of samples to draw')
    parser.add_argument('--h', type=float, default=0.9, help='Initial magnetization along the reference pattern')
    parser.add_argument('--N_iterations', type=int, default=30, help='Number of iterations')
    parser.add_argument('--run_over', type=str, default='T', help='Parameter to run over: T or alpha')
    parser.add_argument('--burnin', type=int, default=1500, help='Number of burn-in steps')
    args = parser.parse_args()
    print(args)
    N = args.N
    T = args.T
    alpha = args.alpha
    N_samples = args.N_samples
    h = args.h
    run_over = args.run_over
    N_iteration = args.N_iterations

    T_values = np.linspace(0.1,1.6,31)
    alpha_values = np.linspace(0.01,0.16,31)

    if run_over == 'T':
        variable = T_values
        alpha = 0.04
    elif run_over == 'alpha':
        variable = alpha_values
        T = 0.3
    # Parameters to save
    names_fixed = ['N','N_samples','burnin','h']
    
    for iteration in range(N_iteration):
        for var in variable:
            if run_over == 'T':
                T = var
            elif run_over == 'alpha':
                alpha = var
            print(f'Running for T={T}, alpha={alpha}, iteration={iteration}')
            
            names_variable = ['T','alpha','iteration']

            params = make_params_dict(names_fixed,names_variable)
            file_path , _ , _ = make_data_paths('spins', experiment_name= 'cut_in_parameters', params=params,base_dir='./data',ext=None)

            S = jax.numpy.load(file_path + '.npy')
            print(f'Loaded samples from {file_path}.npy with shape {S.shape}')
            
            # Compute Histogram of Hamming distances
            r , P = compute_histogram_jax(S)
