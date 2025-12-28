import argparse, time
from spin_sampler import Sampler , define_hopfield_model , initialize_spins
from binary_dimension import compute_histogram_jax
from configurations import make_params_dict, make_data_paths
import jax
import os
import multiprocessing


if __name__ == "__main__":
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
    # print(args)
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
    iteration = args.iteration
    seed += iteration  # Different seed for each iteration

    # Parameters to save
    names_fixed = ['N','N_samples','burnin','h']
    names_variable = ['T','alpha','iteration']

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
    # print(f'Chain shape: {S.shape}') # (N_samples,N)

    params = make_params_dict(names_fixed,names_variable)
    file_path , _ , _ = make_data_paths('spins', experiment_name= 'cut_in_parameters', params=params,base_dir='./data',ext=None)
    jax.numpy.savetxt(file_path, S)        


    dt = time.time() - t0
    print(f'------TIME TAKEN  = {dt/60 :.5} min = {dt/3600 :.3} hours-------')
