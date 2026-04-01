"""Keep JAX numerics predictable across tests.

`jnlr.geodesics.shooting` enables float64 at import; other tests use float32 inputs
with solvers built from `jnp.eye`, which would otherwise mismatch inside `lax.scan`.

On Linux + Python 3.13, `jnp.eye(n)` defaults to float64 unless `dtype=` is set; tests
pass explicit `dtype=jnp.float32` on `W` and reset x64 here as a safety net.
"""

import jax
import pytest


def pytest_configure(config):
    jax.config.update("jax_enable_x64", False)


@pytest.fixture(autouse=True)
def _jax_float32_mode():
    jax.config.update("jax_enable_x64", False)
    yield
