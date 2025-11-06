import pytest
import numpy as np
from spin_sampler.spin_systems import *


def test_hopfield_model():
    N , p  = 10 , 3
    with pytest.raises(ValueError):
        define_hopfield_model(N, p, N_walkers=1, mode='multi_couplings')

    for mode in ['single_chain']:
        with pytest.raises(ValueError):
            define_hopfield_model(N, p, N_walkers=2, mode=mode)
    
    with pytest.raises(ValueError):
        define_hopfield_model(N, p, backend='jax', seed=None)



def test_SK_model():
    N = 10
    with pytest.raises(ValueError):
        define_SK_model(N, N_walkers=1, mode='multi_couplings')

    for mode in ['single_chain' ]:
        with pytest.raises(ValueError):
            define_SK_model(N, N_walkers=2, mode=mode)
    
    with pytest.raises(ValueError):
        define_SK_model(N, backend='jax', seed=None)     


def test_random_model():
    N = 10
    with pytest.raises(ValueError):
        define_random_model(N, N_walkers=1, mode='multi_couplings')

    for mode in ['single_chain' ]:
        with pytest.raises(ValueError):
            define_random_model(N, N_walkers=2, mode=mode)
    
    with pytest.raises(ValueError):
        define_random_model(N, backend='jax', seed=None)     


def test_spin_initialization():
    N = 10
    N_walkers = 3
    m0 = 0.5

    for mode in ['multi_chain','multi_couplings']:
        with pytest.raises(ValueError):
            initialize_spins(N, N_walkers=1, mode=mode)

    with pytest.raises(ValueError):
        initialize_spins(N, N_walkers=2, mode='single_chain')
   
    with pytest.raises(ValueError):
        initialize_spins(N, backend='jax', seed=None)  

    for mode in ['single_chain', 'multi_chain', 'multi_couplings']:
        for backend in ['numpy', 'numba', 'jax']:
            for config in ['random', 'magnetized']:
                
                Nw = 1 if mode == 'single_chain' else N_walkers
                expected_shape = (N,) if mode == 'single_chain' else (Nw,N)

                ref_spin = jnp.ones(shape=expected_shape).astype(type_spins_jax) if backend == 'jax' else np.ones(shape=expected_shape).astype(type_spins_np)
                spins = initialize_spins(N, N_walkers=Nw, mode=mode, backend=backend, seed=42,config=config,ref_spin=ref_spin,m0=m0)
                
                assert spins.shape == expected_shape
                if backend == 'jax':
                    assert isinstance(spins, jnp.ndarray)
                else:
                    assert isinstance(spins, np.ndarray)

    for mode in ['single_chain', 'multi_chain', 'multi_couplings']:
        for backend in ['numpy', 'numba', 'jax']:
                Nw = 1 if mode == 'single_chain' else N_walkers
                expected_shape = (N,) if mode == 'single_chain' else (Nw,N)
                ref_spin = jnp.ones(shape=expected_shape).astype(type_spins_jax) if backend == 'jax' else np.ones(shape=expected_shape).astype(type_spins_np)
                
                with pytest.raises(ValueError):
                    initialize_spins(N, N_walkers=Nw, mode=mode, backend=backend, seed=42,config='magnetized',ref_spin=ref_spin,m0=None)

                with pytest.raises(ValueError):
                    initialize_spins(N, N_walkers=Nw, mode=mode, backend=backend, seed=42,config='magnetized',ref_spin=None,m0=m0)

                other_shape = (N+1,) if mode == 'single_chain' else (Nw+1,N)
                ref_spin = jnp.ones(shape=other_shape).astype(type_spins_jax) if backend == 'jax' else np.ones(shape=other_shape).astype(type_spins_np)
                with pytest.raises(ValueError):
                    initialize_spins(N, N_walkers=Nw, mode=mode, backend=backend, seed=42,config='magnetized',ref_spin=ref_spin,m0=m0)

    m0 = [0.5]*(N_walkers+1)                         
    for mode in ['single_chain', 'multi_chain', 'multi_couplings']:
        for backend in ['numpy', 'numba', 'jax']:
                Nw = 1 if mode == 'single_chain' else N_walkers
                expected_shape = (N,) if mode == 'single_chain' else (Nw,N)
                ref_spin = jnp.ones(shape=expected_shape).astype(type_spins_jax) if backend == 'jax' else np.ones(shape=expected_shape).astype(type_spins_np)
                
                m0 = [0.5]*(N_walkers+1)    
                with pytest.raises(ValueError):
                    initialize_spins(N, N_walkers=Nw, mode=mode, backend=backend, seed=42,config='magnetized',ref_spin=ref_spin,m0=m0)

                m0 = [0.5]*(N_walkers)
                m0[0] = 1.1
                with pytest.raises(ValueError):
                    initialize_spins(N, N_walkers=Nw, mode=mode, backend=backend, seed=42,config='magnetized',ref_spin=ref_spin,m0=m0)

