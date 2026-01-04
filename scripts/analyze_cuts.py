import argparse
import numpy as np
from configurations import make_params_dict ,load_data , save_data
from scipy.stats import zscore


if __name__ == "__main__":
    # Define parameters of the system
    parser = argparse.ArgumentParser(description='Test the spin sampler.')
    parser.add_argument('--N', type=int, default=1024, help='Number of spins')
    parser.add_argument('--N_samples', type=int, default=2500, help='Number of samples to draw')
    parser.add_argument('--N_iterations', type=int, default=40, help='Number of iterations')
    parser.add_argument('--burnin', type=int, default=2000, help='Number of burn-in steps')
    parser.add_argument('--perc', type=float, default=80.0, help='Percentile for Dkl cut')
    parser.add_argument('--z_th', type=float, default=2.0, help='Z-score threshold for bid cut')

    args = parser.parse_args()
    print(args)
    N = args.N
    N_samples = args.N_samples
    N_iteration = args.N_iterations
    burnin = args.burnin
    perc = args.perc
    z_th = args.z_th

    h_values = [0.0,0.9]
    T_values = np.linspace(0.1,1.6,31)
    alpha_values = np.linspace(0.01,0.16,31)


    # Parameters to load

    names_fixed = ['N','N_samples','burnin','h']
    names_variable = ['T','alpha']

    results = {
        'means' : np.zeros((len(h_values),len(T_values),len(alpha_values),6)),
        'stds' : np.zeros((len(h_values),len(T_values),len(alpha_values),6)),
        'counts' : np.zeros((len(h_values),len(T_values),len(alpha_values)))
    }
  
    for ih , h in enumerate(h_values):
        for iT , T in enumerate(T_values):
            for ia , alpha in enumerate(alpha_values):
                params = make_params_dict(names_fixed,names_variable)
                x = load_data('results', experiment_name= 'mapping', params=params,base_dir='./data',ext='txt')
                print(f'\nRunning iteration for h={h}, T={T}, alpha={alpha}. Data shape: {x.shape}') # (N_iterations,6 parameters)

                # Filter out all iterations where any of the parameters is NaN
                x_valid = x[~np.isnan(x).any(axis=1)]

                # Filtering based on Dkl values
                Dkl = x_valid[:,4]
                print(f'Max Dkl before filtering: {np.max(Dkl)}, total valid points: {len(Dkl)}')
                Dkl_percentile = np.percentile(Dkl, perc, axis=None)  # 80th percentile across all values
                mask_Dkl = Dkl <= Dkl_percentile
                x_filtered = x_valid[mask_Dkl]
                print(f'Max Dkl after filtering: {np.max(x_filtered[:,4])}, total filtered points: {len(x_filtered)}')

                 # Check if there is variation across realizations (w) and filter out outliers based on z-score
                th0 = x_filtered[:,2]/N
                std_bid = np.std(th0)
                if std_bid > 1e-3:
                    z_bid = np.abs(zscore(th0))
                    mask_bid = (z_bid < z_th) # Keep only those within threshold
                    x_final = x_filtered[mask_bid]
                else:
                    x_final = x_filtered
                
                print(f'Total points after z-score filtering: {len(x_final)}')

                means = np.mean(x_final, axis=0) #shape (6,)
                stds = np.std(x_final, axis=0)  #shape (6,)

                results['means'][ih,iT,ia,:] = means
                results['stds'][ih,iT,ia,:] = stds
                results['counts'][ih,iT,ia] = x_final.shape[0]
    
    # Save results
    names = ['N','N_samples','burnin','h','z_th','perc']
    params = make_params_dict(names)
    save_data(results,'results',experiment_name='mapping',params=params)

