import pytest
import numpy as np
import jax.numpy as jnp
import jax
from spin_sampler.sampling import Sampler  
from spin_sampler.spin_systems import *



def test_sampler_initialization():
    N = 10
    N_walkers = 3
    J = define_SK_model(N)
    T = 1.0

    with pytest.raises(TypeError):
        Sampler(J, "invalid")  # T must be a number

    # Test valid initializations
    for backend in ['numpy', 'numba', 'jax']:
        for mode in ['single_chain', 'multi_chain', 'multi_couplings']:
            J = define_SK_model(N, N_walkers=N_walkers if mode == 'multi_couplings' else 1, mode=mode, backend=backend, seed=42)
            sampler = Sampler(J,T, mode=mode, backend=backend)
            assert sampler.mode == mode
            assert sampler.T == T
            assert callable(sampler.gibbs_step)
            if backend == 'jax':                    
                assert isinstance(sampler.J, jnp.ndarray)
                with pytest.raises(TypeError):
                    Sampler(np.array(J), T, mode=mode, backend=backend)
            else:
                assert isinstance(sampler.J, np.ndarray)
                with pytest.raises(TypeError):
                    Sampler(jnp.array(J), T, mode=mode, backend=backend)

    
    # Test invalid mode
    for backend in ['numpy', 'numba', 'jax']:
        for inv_mode in ['invalid_mode', 123, None]:
            with pytest.raises(ValueError):
                Sampler(J, T, mode=inv_mode, backend=backend)

    # Test invalid backend
    for inv_backend in ['invalid_backend', 123, None]:
        for mode in ['single_chain', 'multi_chain', 'multi_couplings']:
            with pytest.raises(ValueError):
                Sampler(J, T, mode=mode, backend=inv_backend)

    # Test invalid dimensions of J
    for backend in ['numpy', 'numba', 'jax']:
        for mode in ['single_chain', 'multi_chain', 'multi_couplings']:
            Nw = 1 if mode == 'multi_couplings' else N_walkers
            mod = 'single_chain' if mode == 'multi_couplings' else 'multi_couplings'
            J = define_SK_model(N, N_walkers=Nw, mode=mod, backend=backend, seed=42)
            with pytest.raises(ValueError):
                Sampler(J, T, mode=mode,backend=backend) 



def test_symmetric_matrices():
    N = 10
    N_walkers = 3
    T = 1.0
    for backend in ['numpy', 'numba', 'jax']:
        for mode in ['single_chain', 'multi_chain', 'multi_couplings']:
            Nw = N_walkers if mode == 'multi_couplings' else 1
            J_assym = define_random_model(N,N_walkers=Nw, mode=mode, backend=backend, seed=42)
            # Test asymmetric J
            with pytest.raises(ValueError):
                Sampler(J_assym, T, mode=mode, backend=backend)


def test_gibbs_step_functionality():
    N = 10
    N_walkers = 3
    T = 2.0 
    for backend in ['numpy', 'numba', 'jax']:
        key = jax.random.PRNGKey(42) if backend == 'jax' else 1
        for mode in ['single_chain', 'multi_chain', 'multi_couplings']:

            Nw = N_walkers if mode == 'multi_couplings' else 1
            J = define_SK_model(N, N_walkers=Nw, mode=mode, backend=backend, seed=42)
            sampler = Sampler(J, T, mode=mode, backend=backend)

            Nw = 1 if mode == 'single_chain' else N_walkers
            S_in = initialize_spins(N,N_walkers=Nw,mode=mode,backend=backend,seed=42)
            print(S_in.shape,J.shape)
            
            S_out , key = sampler.gibbs_step(S_in.copy(), J, T,True,key)
            
            # State should change
            if backend == 'jax':
                assert not jnp.array_equal(S_in,S_out)
            else:
                assert not np.array_equal(S_in,S_out) 



def test_sample_method(): 
    N = 10
    N_walkers = 3
    T = 2.0     
    N_samples = 5
    dt_samples = 2

    wrong_shape = {
        'single_chain' : (N_walkers,N),
        'multi_chain': (N,),
        'multi_couplings': (N,)
    }
    wrong_shape1 = {
        'single_chain' : (N+1,),
        'multi_chain': (N_walkers,N+1),
        'multi_couplings': (N_walkers,N+1)
    }
    for backend in ['numpy', 'numba', 'jax']:
            key = jax.random.PRNGKey(42) if backend == 'jax' else 1
            for mode in ['single_chain', 'multi_chain', 'multi_couplings']:
                
                Nw = N_walkers if mode == 'multi_couplings' else 1
                J = define_SK_model(N, N_walkers=Nw, mode=mode, backend=backend, seed=42)
                sampler = Sampler(J, T, mode=mode, backend=backend)

                Nw = 1 if mode == 'single_chain' else N_walkers
                S_in = initialize_spins(N,N_walkers=Nw,mode=mode,backend=backend,seed=42)
   
                with pytest.raises(TypeError):
                    list(sampler.sample('initial_state', N_samples,dt_samples,seed=42))

                if backend == 'jax':
                    with pytest.raises(ValueError):
                        list(sampler.sample(S_in, N_samples,dt_samples,seed=None))

                # Incorrect
                shape = wrong_shape[mode]
                S_in = jnp.ones(shape=shape) if backend=='jax' else np.ones(shape=shape)
                with pytest.raises(ValueError):
                    list(sampler.sample(S_in, N_samples,dt_samples,seed=42))

                shape = wrong_shape1[mode]
                S_in = jnp.ones(shape=shape) if backend=='jax' else np.ones(shape=shape)
                with pytest.raises(ValueError):
                    list(sampler.sample(S_in, N_samples,dt_samples,seed=42))
   

def test_run_gibbs_returns_final_state():
    N = 10
    N_walkers = 3
    T = 2.0     
    N_samples = 5
    dt_samples = 2
    for backend in ['numpy', 'numba', 'jax']:
            key = jax.random.PRNGKey(42) if backend == 'jax' else 1
            for mode in ['single_chain', 'multi_chain', 'multi_couplings']:
                
                Nw = N_walkers if mode == 'multi_couplings' else 1
                J = define_SK_model(N, N_walkers=Nw, mode=mode, backend=backend, seed=42)
                sampler = Sampler(J, T, mode=mode, backend=backend)

                Nw = 1 if mode == 'single_chain' else N_walkers
                S_in = initialize_spins(N,N_walkers=Nw,mode=mode,backend=backend,seed=42)
                S_out = sampler.run_gibbs(S_in,N_samples=N_samples,dt_samples=dt_samples,rnd_ord=True,seed=42,store=True)

                assert S_in.shape == S_out.shape
                assert len(sampler.chain) == N_samples
                assert S_in.dtype == S_out.dtype
                if backend == 'jax':
                    assert isinstance(S_out,jnp.ndarray)
                else:
                    assert isinstance(S_out,np.ndarray)
    
    

def test_run_gibbs_with_none_initial_state():
    N = 10
    N_walkers = 3
    T = 2.0     
    N_samples = 5
    dt_samples = 2
    for backend in ['numpy', 'numba', 'jax']:
            key = jax.random.PRNGKey(42) if backend == 'jax' else 1
            for mode in ['single_chain', 'multi_chain', 'multi_couplings']:
                
                Nw = N_walkers if mode == 'multi_couplings' else 1
                J = define_SK_model(N, N_walkers=Nw, mode=mode, backend=backend, seed=42)
                sampler = Sampler(J, T, mode=mode, backend=backend)
                
                with pytest.raises(ValueError):
                    sampler.run_gibbs(None,N_samples=N_samples,dt_samples=dt_samples)

                Nw = 1 if mode == 'single_chain' else N_walkers
                S_in = initialize_spins(N,N_walkers=Nw,mode=mode,backend=backend,seed=42)
                sampler.run_gibbs(S_in,N_samples=N_samples,dt_samples=dt_samples,rnd_ord=True,seed=42,store=True);
                sampler.run_gibbs(None,N_samples=N_samples,dt_samples=dt_samples,rnd_ord=True,seed=42,store=True);

                assert len(sampler.chain) == 2*N_samples - 1
