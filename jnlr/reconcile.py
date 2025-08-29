import jax.numpy as jnp
import jax
from jax import lax
from typing import Callable, Optional


def solve_nlp_reconciliation_fixed_iter(f:Callable, z_hat: jnp.ndarray,
                                        n_constraints:int=1, W: Optional[jnp.ndarray] = None,
                                        n_iterations_learning: int = 10) -> tuple:
    r"""
    Solve the constrained optimization problem using a fixed number of iterations of the Newton-Lagrange method.
    $$
    \text{arg}\min_{z} \tfrac{1}{2} (z - \hat{z})^T W (z - \hat{z}) \\
    $$
    $$
    \text{s.t. } f(z) = 0
    $$


    Args:
        f:  Function representing the constraints.
        z_hat:  Initial guess for the optimization variable.
        n_constraints:  Number of constraints.
        W:  Weight matrix.
        n_iterations_learning:  Number of iterations for the learning process.

    Returns:
        z_final:  The projected z after the fixed number of iterations.

    """

    jac_f_compiled = jax.jit(lambda x: jnp.atleast_2d(jax.jacobian(f, argnums=0)(x)))

    z = z_hat.copy()
    lam = jnp.zeros(n_constraints)

    # Pre-compute 2*W
    W2 = 2 * W

    def body_fn(i, state):
        Z, lam = state

        # Compute mismatch
        f_val = f(Z)

        # Compute updates using Newton-Lagrange method
        J = jac_f_compiled(Z)
        top = jnp.concatenate([W2, J.T], axis=1)
        bottom = jnp.concatenate([J, jnp.zeros((n_constraints, n_constraints))], axis=1)
        LHS = jnp.concatenate([top, bottom], axis=0)
        RHS = jnp.concatenate([-W2 @ (Z - z_hat) - J.T @ lam, -jnp.atleast_1d(f_val)], axis=0)

        # Add small regularization for numerical stability
        # LHS = LHS + jnp.eye(LHS.shape[0]) * 1e-8

        sol = jnp.linalg.solve(LHS, RHS)
        dZ = sol[:-n_constraints]
        dlambda = sol[-n_constraints:]

        Z_new = Z + dZ
        lam_new = lam + dlambda

        return (Z_new, lam_new)

    # Fixed iteration loop
    z_final, lam_final = lax.fori_loop(0, n_iterations_learning, body_fn, (z, lam))


    return z_final

def make_solver(f, W, n_constraints: int, n_iterations_learning: int = 10):
    """
    Create a JIT-compiled solver function for the constrained optimization problem.


    Args:
        f: Function representing the constraints.
        W: Weight matrix.
        n_constraints:  Number of constraints.
        n_iterations_learning:  Number of iterations for the learning process.

    Returns:
        A JIT-compiled function that takes z_hat as input and returns the projected z.
    """

    W2 = 2 * W
    jac_f = jax.jacobian(f, argnums=0)

    @jax.jit
    def solve_single(z_hat: jnp.ndarray) -> jnp.ndarray:
        z = z_hat.copy()
        lam = jnp.zeros(n_constraints)

        def body_fn(i, state):
            Z, lam = state
            f_val = f(Z)
            J = jnp.atleast_2d(jac_f(Z))

            top = jnp.concatenate([W2, J.T], axis=1)
            bottom = jnp.concatenate([J, jnp.zeros((n_constraints, n_constraints))], axis=1)
            LHS = jnp.concatenate([top, bottom], axis=0)
            RHS = jnp.concatenate([-W2 @ (Z - z_hat) - J.T @ lam, -jnp.atleast_1d(f_val)], axis=0)

            sol = jnp.linalg.solve(LHS, RHS)
            dZ = sol[:-n_constraints]
            dlambda = sol[-n_constraints:]
            return (Z + dZ, lam + dlambda)

        z_final, _ = lax.fori_loop(0, n_iterations_learning, body_fn, (z, lam))
        return z_final

    return solve_single


