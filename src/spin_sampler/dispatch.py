from spin_sampler.gibbs_steps import gibbs_step_single_chain, gibbs_step_multi_chain, gibbs_step_multi_couplings
from spin_sampler.gibbs_steps import gibbs_step_single_chain_jax, gibbs_step_multi_chain_jax, gibbs_step_multi_couplings_jax
from spin_sampler import compiled_gibbs 

# Define the dispatcher
DISPATCHER = {
    'numpy': {
        'single_chain': gibbs_step_single_chain,
        'multi_chain': gibbs_step_multi_chain,
        'multi_couplings': gibbs_step_multi_couplings,
    },
    'numba': {
        'single_chain': compiled_gibbs.gibbs_step_single_chain_numba,
        'multi_chain': compiled_gibbs.gibbs_step_multi_chain_numba,
        'multi_couplings': compiled_gibbs.gibbs_step_multi_couplings_numba,
    },
    'jax': {
        'single_chain': gibbs_step_single_chain_jax,
        'multi_chain': gibbs_step_multi_chain_jax,
        'multi_couplings': gibbs_step_multi_couplings_jax,
    }
}

