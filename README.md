

# Hopfield Dimensionality

Compact tools for sampling Hopfield spin systems and estimating the
Binary Intrinsic Dimension (BID) from binary spin data.

Two main components (under `src/`):

- `spin_sampler` — Gibbs sampling for Hopfield / spin-glass models.
- `binary_dimension` — histogram, distance models, derivatives, and
  optimization code to estimate BID.


## Installation

To install the Spin Sampler library :

1. Clone the repository:
    ```bash
    git clone https://github.com/CristopherErazo/devtools_scicomp_project_2025.git <folder_name>

    ```

2. Navigate to the folder directory, create and activate a virtual environment (with `python 3.9` preferably) and install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Precompile the `numba` module. 

    ```bash
    python src/spin_sampler/compile_gibbs.py
    ```
    This creates a file (`.so` in linux or `.pyd` in Windows) that contains the precompiled version of the `numba` functions and can be called as a module. If running on Windows you might need to install [MSVC Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first, selecting *Desktop development with C++* during installation.

4. Install the package:
    ```bash
    pip install .
    ```

## Quick examples

Run the sampler and get samples:

```python
from spin_sampler import Sampler, define_hopfield_model, initialize_spins
J, _ = define_hopfield_model(1000, 40)
init = initialize_spins(1000)
sampler = Sampler(J, T=1.0)
sampler.run_gibbs(init, N_samples=1000, dt_samples=1, store=True)
S = sampler.get_chain()
```

Fit BID models (see `notebooks/template.ipynb` for a full demo):

```python
from binary_dimension import compute_histogram
from binary_dimension.utils import initial_gauss
from binary_dimension.distance_models import LinearDistanceModel
from binary_dimension.optimization import DKL_Optimizer

r, P = compute_histogram(S)
theta0 = initial_gauss(r, P)
model = LinearDistanceModel()
opt = DKL_Optimizer(r, P, model)
theta_opt, dkl_hist, nit = opt.optimize(theta0)
```

## Layout (high level)

- `src/spin_sampler/` — sampling code and backends.
- `src/binary_dimension/` — BID tools: `BID.py`, `distance_models.py`,
  `derivatives.py`, `optimization.py`, `utils.py`.
- `notebooks/` — example notebooks (template.ipynb).


## Citation & License

If you use this code in research, cite: *The dimensionality of the Hopfield
model* (preprint/manuscript). See `LICENSE` for terms.

---

