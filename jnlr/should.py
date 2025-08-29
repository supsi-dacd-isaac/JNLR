from jax import numpy as jnp
from functools import partial
import jax
from curvature_utils import min_tangent_eigenvalue, curvature_along_projection, min_tangent_eigenvalue_vv
from jnlr.reconcile import make_solver
from stats import clopper_pearson_intervals

@partial(jax.jit, static_argnames=('f', 'grad_f', 'hessian_f', 'vmapped_solver'))
def constant_sign_curvature(f, grad_f, hessian_f, vmapped_solver, z_hat: jnp.ndarray) -> jnp.ndarray:
    r"""
    Check if forecasting RMSE is guaranteed to reduce based on the curvature condition, for a hypersurface defined by f(z) = 0 having constant sign curvature.

    Args:
        f: Constraint function.
        grad_f: Function returning the gradient vector of shape (n,).
        hessian_f: Function returning the Hessian matrix of shape (n, n).
        vmapped_solver: Function to project points onto the constraint surface.
        z_hat: Forecasted points of shape (batch_size, n).


    Returns:
        np.ndarray: Boolean array indicating if the curvature condition is satisfied for each point in z_hat


    Notes:
        Theorem 1: For a hypersurface defined by f(z) = 0, forecasting RMSE is guaranteed to reduce if:

        $$\lambda_{min}(H_{restricted}(\tilde{z})) * f(\hat{z}) > 0$$

        where
        * \( \tilde{z} \) is the projection of the forecasted point, $\hat{z}$, onto the surface defined by $f(z) = 0.$
    """

    z_tilde = vmapped_solver(z_hat)
    lambda_min = jax.vmap(min_tangent_eigenvalue, in_axes=(None, None, 0))(grad_f,  hessian_f, z_tilde)
    f_val = jax.vmap(f)(z_hat)
    return lambda_min * f_val > 0



@partial(jax.jit, static_argnames=('jacobian_F', 'hessians_F', 'vmapped_solver'))
def vector_valued_convex(f, jacobian_F, hessians_F, vmapped_solver, z_hat: jnp.ndarray) -> jnp.ndarray:
    r"""
    Generalization of scalar curvature condition to vector-valued case.

    Args:
        f: Constraint function.
        jacobian_F: Function returning $(m, n)$ Jacobian matrix $DF(z)$.
        hessians_F: Function returning $(m, n, n)$ Hessians of each
        vmapped_solver: Function to project points onto the constraint surface.
        z_hat: Forecasted points of shape $(batch_size, n)$.
    Returns:
        np.ndarray: Boolean array indicating if the curvature condition is satisfied for each point in $\hat{z}$.

    Notes:
        Check if lambda_min(H_combined) >= 0, where:
        H_combined = $sum_i (\delta_\pi[i] / ||∇f_i||) * H_{tan_i}$
    where:
        - $\delta_\pi = \tilde{z} - \hat{z}$
        - $H_{tan_i} = E^T H_i E$  (Hessian projected to tangent space)

    """

    z_tilde = vmapped_solver(z_hat)
    lambdas = jax.vmap(min_tangent_eigenvalue_vv, in_axes=(None, None, None, 0, 0))(f, jacobian_F,  hessians_F, z_tilde, z_hat)
    lambda_min = lambdas[:, 0]  # Take the minimum eigenvalue from the pair (min, max)
    ratios = lambdas[:, 1]
    accepted = lambda_min > 0
    return accepted, ratios



@partial(jax.jit, static_argnames=('vmapped_solver'))
def p_reduction(vmapped_solver, z_hat: jnp.ndarray, z_hat_samples: jnp.ndarray, alpha=0.05):
    r"""
    Estimate the probability of RMSE reduction using projected bootstrap samples.

    Args:
        vmapped_solver: Function to project points onto the constraint surface.
        z_hat: Forecasted points of shape `(batch_size, n)`.
        z_hat_samples: Bootstrap samples of shape `(batch_size, num_samples, n)`.
        alpha: Significance level for confidence intervals (not used by this function).

    Returns:
        np.ndarray: Estimated probability of RMSE reduction for each point in `z_hat`.

    Notes:
        **Theorem 3.** For a hypersurface defined by $f(z) = 0$,
        the probability of forecasting RMSE reduction can be estimated as

        $$\frac{1}{N} \sum_{i=1}^{N}
        \mathbf{1}\!\left\{\,\tilde{\delta}_i^\top \delta_{\pi}
        > - \frac{\lVert \delta_{\pi} \rVert^2}{2} \right\}.$$


        where

        * \( \mathbf{1} \) is the indicator function;
        * \( \tilde{\delta}_i = \tilde{y}_i - \tilde{y} \);
        * \( \tilde{y}_i \) is the projection of the \(i\)-th sample onto the surface \( f(z) = 0 \);
        * \( \tilde{y} \) is the projection of \( \hat{y} \) onto the surface \( f(z) = 0 \);
        * \( \delta_{\pi} = \tilde{y} - \hat{y} \).
    """

    z_tilde = vmapped_solver(z_hat)
    z_tilde_samples = vmapped_solver(z_hat_samples.reshape(-1, z_hat.shape[-1])).reshape(z_hat_samples.shape)
    delta_pi = z_tilde - z_hat

    delta_tilde_samples = z_tilde_samples - z_tilde[:, None, :]
    norm_delta_pi = jnp.linalg.norm(delta_pi, axis=-1, keepdims=True)
    scalar_product = jnp.sum(delta_tilde_samples * delta_pi[:, None, :], axis=-1)
    threshold = norm_delta_pi ** 2 / 2
    condition = scalar_product > - threshold

    # cutoff
    delta_tilde_hat_samples = z_tilde_samples - z_hat[:, None, :]
    post_rec_dist = jnp.sum(delta_tilde_hat_samples**2, axis=-1)
    bootstrap_errs = z_hat_samples - z_hat[:, None, :]
    pre_rec_dist = jnp.sum(bootstrap_errs ** 2, axis=-1)
    max_pre_rec_dist = jnp.max(pre_rec_dist, axis=-1, keepdims=True)
    condition = jnp.where(post_rec_dist < max_pre_rec_dist, condition, jnp.nan)

    return jnp.nanmean(condition, axis=-1)

def p_reduction_and_intervals(vmapped_solver, z_hat: jnp.ndarray, z_hat_samples: jnp.ndarray, alpha=0.02):
    r"""
    Estimate the probability of RMSE reduction using projected bootstrap samples,
    along with Clopper-Pearson confidence intervals.
    Args:
        vmapped_solver: Function to project points onto the constraint surface.
        z_hat: Forecasted points of shape `(batch_size, n)`.
        z_hat_samples: Bootstrap samples of shape `(batch_size, num_samples, n)`.
        alpha: Significance level for confidence intervals.
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - Estimated probability of RMSE reduction for each point in `z_hat`.
            - Clopper-Pearson confidence intervals for each estimate.
    """

    z_tilde = vmapped_solver(z_hat)
    z_tilde_samples = vmapped_solver(z_hat_samples.reshape(-1, z_hat.shape[-1])).reshape(z_hat_samples.shape)
    delta_pi = z_tilde - z_hat

    delta_tilde_samples = z_tilde_samples - z_tilde[:, None, :]
    norm_delta_pi = jnp.linalg.norm(delta_pi, axis=-1, keepdims=True)
    scalar_product = jnp.sum(delta_tilde_samples * delta_pi[:, None, :], axis=-1)
    threshold = norm_delta_pi ** 2 / 2
    condition = scalar_product > - threshold

    # cutoff
    delta_tilde_hat_samples = z_tilde_samples - z_hat[:, None, :]
    post_rec_dist = jnp.sum(delta_tilde_hat_samples ** 2, axis=-1)
    bootstrap_errs = z_hat_samples - z_hat[:, None, :]
    pre_rec_dist = jnp.sum(bootstrap_errs ** 2, axis=-1)
    max_pre_rec_dist = jnp.max(pre_rec_dist, axis=-1, keepdims=True)
    condition = jnp.where(post_rec_dist < max_pre_rec_dist, condition, jnp.nan)

    # compute clopper-pearson intervals
    intervals = jax.vmap(clopper_pearson_intervals, in_axes=(0, None, None))(jnp.nansum(condition, axis=-1),
                                                                             condition.shape[-1], alpha)
    return jnp.nanmean(condition, axis=-1), intervals, delta_pi

