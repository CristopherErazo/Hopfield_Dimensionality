# from SGD.utils import make_params_dict , make_data_paths , download_cluster_data
from configurations import make_params_dict, make_data_paths, download_cluster_data
import numpy as np
import os

# Parameters fixed
N_samples = 10000
dt_samples = 1
burnin = 800
alpha = 0.04
h = 0.9
rnd_ord = True
mode = 'single_chain'

# Parameters Lists
temps = [0.56, 0.575, 0.59, 1.19, 1.21]
# Ns = [1024,2048,4096]
# temps = np.linspace(0.4,1.6,25)
Ns = np.logspace(9,15,7,base=2).astype(int)#[:-1]

# Parameters to save
names_fixed = ['N_samples','dt_samples','burnin','alpha','h','rnd_ord','mode']
names_variable = ['T','N']

for T in temps:
    for N in Ns:
        p = int(alpha*N)
        params = make_params_dict(names_fixed,names_variable)
        _ , filename , path_local = make_data_paths('histograms', experiment_name= 'scaling_exponents', params=params,base_dir='./data',ext='txt',normalize=False)
        path_cluster = os.path.join('Hopfield_Dimensionality/',path_local)

        download_cluster_data('ulysses',path_cluster,path_local,filename,filename,show=True)
