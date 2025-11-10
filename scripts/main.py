import argparse
# import numpy as np
# import jax.numpy as jnp
from spin_sampler import Sampler , define_hopfield_model , initialize_spins
from binary_dimension import compute_histogram_jax, select_range
from binary_dimension.utils import initial_gauss
from binary_dimension.optimization import DKL_Optimizer
from binary_dimension.distance_models import LinearDistanceModel

if __name__ == "__main__":
    # Define parameters of the system
    parser = argparse.ArgumentParser(description='Test the spin sampler.')
    parser.add_argument('--N', type=int, default=100, help='Number of spins')
    parser.add_argument('--T', type=float, default=1.0, help='Temperature')
    parser.add_argument('--alpha', type=float, default=0.04, help='Load parameter (only for Hopfield model)')
    parser.add_argument('--N_walkers', type=int, default=1, help='Number of parallel chains')
    parser.add_argument('--mode', type=str, default='single_chain', help="Sampling mode: 'single_chain', 'multi_chain' or 'multi_couplings'")
    parser.add_argument('--backend', type=str, default='jax', help="Backend: 'numpy', 'numba' or 'jax'")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (mandatory for jax backend)')
    parser.add_argument('--N_samples', type=int, default=500, help='Number of samples to draw')
    parser.add_argument('--dt_samples', type=int, default=1, help='Number of Monte Carlo steps between samples (reduce time correlation)')
    parser.add_argument('--rnd_ord', type=bool, default=True, help='Randomize the order of spin updates (True recommended)')
    parser.add_argument('--store', type=bool, default=True, help='Store the samples')
    parser.add_argument('--progress', type=bool, default=True, help='Show a progress bar (tqdm required)')
    parser.add_argument('--a_min', type=float, default=0.0, help='Minimum cumulative probability for range selection')
    parser.add_argument('--a_max', type=float, default=1.0, help='Maximum cumulative probability for range selection')
    parser.add_argument('--a_tot', type=float, default=0.9, help='Total cumulative probability for highest probability range')

    args = parser.parse_args()

    N = args.N
    T = args.T
    alpha = args.alpha
    N_walkers = args.N_walkers
    mode = args.mode
    backend = args.backend
    seed = args.seed
    N_samples = args.N_samples
    dt_samples = args.dt_samples
    rnd_ord = args.rnd_ord
    store = args.store
    progress = args.progress
    a_min = args.a_min
    a_max = args.a_max
    a_tot = args.a_tot



    # Define the couplings of Hopfield model and initialize spins
    p = int(alpha * N)
    J , patterns = define_hopfield_model(N,p,N_walkers,mode,backend,seed)
    initial_state = initialize_spins(N,N_walkers,mode,backend,seed)

    # Create a sampler instance and sample
    sampler = Sampler(J, T, mode,backend)
    sampler.run_gibbs(initial_state,N_samples,dt_samples,rnd_ord,seed,store,progress);
    S = sampler.get_chain()
    sampler.reset_chain() # Clean up the sampler
    print(f'Chain shape: {S.shape}') # (N_walkers,N_samples,N)

    # Compute histogram of Hamming distances using JAX 
    r, P = compute_histogram_jax(S)
    print(f'Hamming distances r: {r.shape}, probabilities P: {P.shape}')

    # Select a sub-range of distances
    rx , Px = select_range(r,P,a_min,a_max)
    print(f'Selected range r: {rx.shape}, probabilities P: {Px.shape}')

    theta_init = initial_gauss(rx,Px)
    print(f'Initial parameters theta: {theta_init}')

    distance_model = LinearDistanceModel()
    Dkl_optimizer = DKL_Optimizer(rx,Px, distance_model)
    theta_opt , log_DKL , Nit = Dkl_optimizer.optimize(theta_init)
    print(f'Optimized parameters theta: {theta_opt}')
    print(f'Log D_KL at optimum: {log_DKL}')
    print(f'Number of iterations: {Nit}')