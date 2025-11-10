from configurations import make_params_dict, save_data
from configurations.data import load_histogram_data
from binary_dimension import DKL_Optimizer, LinearDistanceModel, initial_gauss, select_range
from scipy.stats import zscore
import numpy as np
import argparse


def analize_histograms(N, data_hist, a_min, a_max, z_threshold = 2, kl_perc = 80):
    """
    Analyze and optimize histograms to extract binary interaction distance (BID) parameters.

    Parameters
    ----------
    N : int
        System size.
    data_hist : list of (2, N) numpy arrays
        List of histogram data, each containing r and P arrays.
    a_min : float
        Minimum value for range selection.
    a_max : float
        Maximum value for range selection.
    z_threshold : float
        Z-score threshold for filtering outliers based on BID/N.
    kl_perc : float
        Percentile threshold for filtering based on KL divergence.

    Returns
    -------
    bid_pars : ndarray
        Optimized BID parameters for each histogram. Size (Neff, 3) where Neff is the number of effective histograms after filtering.
    data_hist : list of (2, N) numpy arrays
        Filtered list of histogram data after optimization.
    sigmas : ndarray
        Standard deviations of the optimized models.

    """

    # Number of histograms to process and distance model
    Nw = len(data_hist)
    distance_model = LinearDistanceModel()
    
    # Optimize all histograms
    bid_pars = []
    for w in range(Nw): 
        # Extract r and P from histogram data and select range
        r , P = data_hist[w]
        rx , Px = select_range(r,P,a_min=a_min,a_max=a_max)
        # Create DKL optimizer and optimize with gaussian initial condition
        KL_optimizer = DKL_Optimizer(rx,Px,distance_model)
        theta_init = initial_gauss(rx,Px)
        theta_opt , log_DKL , Nit = KL_optimizer.optimize(theta_init,max_iter=100)
        bid_pars.append([*theta_opt,log_DKL])
    bid_pars = np.array(bid_pars)

    # Filter based on KL values
    kl = bid_pars[:,2]
    
    # Filter out NaNs first
    mask_not_nan = ~np.isnan(kl)
    bid_pars = bid_pars[mask_not_nan]
    data_hist = [data_hist[i] for i in range(len(data_hist)) if mask_not_nan[i]]
    kl = kl[mask_not_nan]
    print(f'Maximum KL before filtering: {np.max(bid_pars[:,2])}')

    # Now filter based on kl percentile
    kl_percentile = np.percentile(kl, kl_perc, axis=None)  # 80th percentile across all values
    mask_kl = (kl <= kl_percentile)
    bid_pars = bid_pars[mask_kl]
    data_hist = [data_hist[i] for i in range(len(data_hist)) if mask_kl[i]]

    # Now test with bid/N
    th0 = bid_pars[:,0]/N    
    
    # Check if there is variation across realizations (w) and filter out outliers based on z-score
    std_bid = np.std(th0, axis=0)
    if np.any(std_bid > 1e-3):
        z_bid = np.abs(zscore(th0, axis=0, nan_policy='omit'))
        mask_bid = (z_bid < z_threshold) # Keep only those within threshold
        bid_pars = bid_pars[mask_bid]
        data_hist = [data_hist[i] for i in range(len(data_hist)) if mask_bid[i]]
    print(f'Maximum KL after filtering: {np.max(bid_pars[:,2])}')


    # Compute the standard deviations of the optimized models
    # for each remaining histogram after filtering
    r_full = np.arange(N+1)
    x_full = 1-2*r_full/N
    Neff = len(data_hist)
    sigmas = []
    for w in range(Neff):
        Pmod = distance_model.P_model(r_full,bid_pars[w,:2])
        avx = np.sum(x_full*Pmod)
        stx = np.sqrt( np.sum(x_full**2 *Pmod) - avx**2)
        sigmas.append(stx)
    sigmas = np.array(sigmas)

    return bid_pars, data_hist , sigmas


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process histogram scaling data.')
    parser.add_argument('--a_min', type=float, default=0.0, help='Minimum value for range selection.')
    parser.add_argument('--a_max', type=float, default=0.3, help='Maximum value for range selection.')
    parser.add_argument('--z_threshold', type=float, default=2.0, help='Z-score threshold for filtering outliers based on BID/N.')
    parser.add_argument('--kl_perc', type=float, default=80.0, help='Percentile threshold for filtering based on KL divergence.')
    args = parser.parse_args()

    # Parse arguments
    a_min = args.a_min
    a_max = args.a_max
    z_threshold = args.z_threshold
    kl_perc = args.kl_perc 

    print('Starting histogram scaling processing with parameters:')
    print(f'Parameters: a_min={a_min}, a_max={a_max}, z_threshold={z_threshold}, kl_perc={kl_perc}')

    # Parameters fixed
    N_samples = 10000
    dt_samples = 1
    burnin = 800
    alpha = 0.04
    h = 0.9
    rnd_ord = True
    mode = 'single_chain'

    # Parameters Lists
    # temps = [0.6,0.9,1.2,1.5]
    # Ns = [1024,2048,4096]
    temps = np.linspace(0.4,1.6,25)
    Ns = np.logspace(9,15,7,base=2).astype(int)

    # Parameters to save
    names_fixed = ['N_samples','dt_samples','burnin','alpha','h','rnd_ord','mode']
    names_variable = ['T','N']
    name_fixed_save = ['N_samples','dt_samples','burnin','alpha','h','rnd_ord','mode','a_min','a_max','z_threshold','kl_perc']

    for T in temps:
        for N in Ns:
            print('-----------\n',f'Processing N={N}, T={T}...')
            params = make_params_dict(names_fixed,names_variable)
            data_hist = load_histogram_data('histograms', experiment_name= 'scaling_exponents', params=params,base_dir='./data',ext='txt')
            bid_pars, data_hist_filtered , sigmas = analize_histograms(N, data_hist, a_min, a_max, z_threshold, kl_perc)
            print(f'len(Original histograms): {len(data_hist)} | len(Filtered histograms): {len(data_hist_filtered)}')
            # Save the filtered results
            params_save = make_params_dict(name_fixed_save,names_variable)
            save_data(data_hist_filtered,'histograms',experiment_name= 'scaling_exponents_filtered', params=params_save,base_dir='./data',show=False)
            dic = {'bid_pars': bid_pars , 'sigmas': sigmas}
            save_data(dic,'bid_parms',experiment_name= 'scaling_exponents_filtered', params=params_save,base_dir='./data',show=False)