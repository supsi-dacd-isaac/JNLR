<img src="docs/logo.webp" alt="JNLR Logo" width="120" align="left">

### JNLR
JAX-based non-linear reconciliation and learning

<br clear="left">

**J-NLR** is a Python library for non-linear reconciliation, learning, and geometric analysis on constraint manifolds. Built on [JAX](https://github.com/google/jax), it leverages automatic differentiation and GPU/TPU acceleration to efficiently project predicted values onto surfaces defined by implicit constraints $f(z) = 0$.

[![PyPI version](https://img.shields.io/pypi/v/JNLR)](https://pypi.org/project/JNLR/)
[![Python version](https://img.shields.io/pypi/pyversions/JNLR)](https://pypi.org/project/JNLR/)
[![License](https://img.shields.io/pypi/l/JNLR)](https://github.com/a-paulus/JNLR/blob/main/LICENSE)
[![codecov](https://codecov.io/gh/supsi-dacd-isaac/JNLR/graph/badge.svg?branch=main)](https://codecov.io/gh/supsi-dacd-isaac/JNLR)

📚 **[Documentation](https://supsi-dacd-isaac.github.io/JNLR/)** - Full API reference, examples, and interactive notebooks

## Minimal projection example
 [▶ Minimal projection example](https://supsi-dacd-isaac.github.io/JNLR/examples/projection_minimal_example/)
```python
import numpy as np
from jnlr.reconcile import make_solver

# generate gaussian samples in 3d
# and reproject them onto the constraint
n_samples = 1000
X = np.random.randn(n_samples, 3)

# define a constraint as an implicit function.
# Each component of the function (1 in this case) returns
# how far a point is from the surface of the constraint
def f_implicit(v):
    z1, z2, z3 = v
    return z1**2 + z2**2 - z3**2

solver = make_solver(f_implicit, n_iterations=30)
X_proj = solver(X)
incoherence_pre = np.mean(np.abs([f_implicit(x_i) for x_i in X]))
incoherence_post = np.mean(np.abs([f_implicit(x_i) for x_i in X_proj]))
print("mean abs f before projection: {:0.2e}".format(incoherence_pre))
print("mean abs f after projection: {:0.2e}".format(incoherence_post))
```
response: 
```
mean abs f before projection: 1.84e+00
mean abs f after projection: 1.91e-09
```

## Key Features

- **Non-linear Reconciliation** — Multiple solvers (Augmented Lagrangian, curvature-aware Newton, vanilla projections) for projecting forecasts onto constraint manifolds defined by $f(z)=0$, each respecting a user-defined weighting matrix $W$.
  [▶ Interactive example](https://supsi-dacd-isaac.github.io/JNLR/examples/projection_hypersurfaces/)

- **SHOULD Analysis** — Curvature-based methods to determine *when* reconciliation is beneficial. Analyse the local curvature of the constraint surface and the distribution of prediction errors to verify whether RMSE is guaranteed to reduce before applying any correction.
  [▶ Interactive example](https://supsi-dacd-isaac.github.io/JNLR/examples/should/)

- **Manifold Sampling** — Sample from explicit (graph) or implicit manifolds using volume-weighted random sampling, Latin hypercube designs, or Langevin dynamics constrained to the surface — useful for Monte-Carlo estimation and training-set augmentation.
  [▶ Interactive example](https://supsi-dacd-isaac.github.io/JNLR/examples/samplers/)

- **Mesh Generation** — Create triangulated meshes from explicit parameterisations for visualisation and exact geodesic computation.
  [▶ Interactive example](https://supsi-dacd-isaac.github.io/JNLR/examples/meshes/)

- **Geodesics** — Compute geodesic distances and shortest paths on manifolds via the exact Mitchell–Mount–Papadimitriou (MMP) algorithm or fast graph-based approximations. Includes the *pointcloud geodesic distance*, a probabilistic score for distributional shifts on curved spaces.
  [▶ Interactive example](https://supsi-dacd-isaac.github.io/JNLR/examples/compute_geodesics/)

- **Visualization** — Interactive 3D rendering of manifolds, projections, sampling distributions, and geodesic paths with [Plotly](https://plotly.com/python/). All example notebooks include fully interactive, rotatable plots — [explore them in the docs](https://supsi-dacd-isaac.github.io/JNLR/).

- **JAX-native** — Fully JIT-compiled and vectorized (`vmap`) for high-performance batch processing. Automatic differentiation provides exact Jacobians and Hessians without finite-difference approximations.


## Citation

If you use **JNLR** in academic work, please cite the associated paper:

Lorenzo Nespoli, Anubhab Biswas, Roberto Rocchetta, and Vasco Medici.  
"Nonlinear reconciliation: Error reduction theorems."  
Transactions on Machine Learning Research (TMLR), 2026.  
OpenReview: https://openreview.net/forum?id=dXRWuogm3J

### BibTeX

```bibtex
@article{nespoli2026nonlinear_reconciliation,
  title   = {Nonlinear reconciliation: Error reduction theorems},
  author  = {Nespoli, Lorenzo and Biswas, Anubhab and Rocchetta, Roberto and Medici, Vasco},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=dXRWuogm3J},
  note    = {Accepted by TMLR}
}
```

## Acknowledgements

This work has been funded by the Swiss State Secretariat for Education, Research and Innovation (SERI) under the Swiss contribution to the Horizon Europe projects DR-RISE (Horizon Europe, Grant Agreement No. 101104154) and REEFLEX (Horizon Europe, Grant Agreement No. 101096192).
