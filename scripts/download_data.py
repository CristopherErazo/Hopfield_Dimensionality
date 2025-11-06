# from SGD.utils import make_params_dict , make_data_paths , download_cluster_data
from configurations import make_params_dict, make_data_paths, download_cluster_data
import numpy as np

# Parameters fixed
N_samples = 3500
dt_samples = 1
burnin = 800
alpha = 0.04
h = 0.0
rnd_ord = True
mode = 'multi_chain'

# Parameters Lists
temps = [0.6,0.9,1.2,1.5]
Ns = [1024,2048,4096]
# temps = np.linspace(0.4,1.6,25)
# Ns = np.logspace(9,15,7,base=2).astype(int)

# Parameters to save
names_fixed = ['N_samples','dt_samples','burnin','alpha','h','rnd_ord','mode']
names_variable = ['T','N']

for T in temps:
    for N in Ns:
        p = int(alpha*N)
        path_cluster =f'Hopfield_BID/New_Data1/Hist/N_{N}/Ns_{N_samples}_dts_{dt_samples}/'
        filename_cluster = f'multi_T{T:.4}_alpha{alpha:.4}_p{p}_h{h:.4}_burnin{burnin}_rnd-ord{rnd_ord}.txt'

        params = make_params_dict(names_fixed,names_variable)
        _ , filename_local , path_local = make_data_paths('histograms', experiment_name= 'scaling_exponents', params=params,base_dir='./data',ext='txt')
        
        download_cluster_data('ulysses',path_cluster,path_local,filename_cluster,filename_local,show=True)
    
        # print(path_cluster)
        # print(filename_cluster)
        # print(path_local)
        # print(filename_local)
                # download_cluster_data('ulysses',path_cluster,path_local,filename_cluster,filename_local,show=False)
    
    
    # path_cluster =f'~/Hopfield_BID/New_Data1/Hist/N_{N}/Ns_{N_samples}_dts_{dt_samples}/'
    # os.makedirs(path_hist,exist_ok=True)
    # filename = f'multi_T{T:.4}_alpha{alpha:.4}_p{p}_h{h:.4}_burnin{burnin}_rnd-ord{rnd_ord}.txt'