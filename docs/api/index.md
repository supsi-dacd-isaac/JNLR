# API Reference

Welcome to the JNLR API documentation. Select a module from the navigation to explore its functions and classes.

## Modules

| Module | Description |
|--------|-------------|
| [**Reconcile**](reconcile.md) | Non-linear reconciliation solvers including Augmented Lagrangian and curvature-aware Newton methods |
| [**Should**](should.md) | SHOULD analysis for determining when reconciliation is beneficial based on curvature |
| [**Stats**](stats.md) | Statistical utilities for error analysis and metrics |
| [**Curvature Utils**](curvature_utils.md) | Low-level curvature computation utilities |

## Quick Example

```python
import jax.numpy as jnp
from jnlr import reconcile, should

# Define your constraint function
def constraint(z):
    return z[0] + z[1] + z[2] - 1.0

# Your predictions
predictions = jnp.array([0.4, 0.3, 0.4])

# Reconcile to satisfy the constraint
reconciled = reconcile.project(predictions, constraint)
```
