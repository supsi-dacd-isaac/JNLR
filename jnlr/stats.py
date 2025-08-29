import jax.numpy as jnp
from jax import lax
from jax.scipy.special import betainc
import jax

@jax.jit
def beta_ppf_approx(q, a, b, tol=1e-7, max_iter=100):

    def body_fn(state):
        low, high, i = state
        mid = (low + high) / 2
        cdf_mid = betainc(a, b, mid)
        low = jnp.where(cdf_mid < q, mid, low)
        high = jnp.where(cdf_mid >= q, mid, high)
        return (low, high, i + 1)

    def cond_fn(state):
        low, high, i = state
        return jnp.logical_and(jnp.any(high - low > tol), i < max_iter)

    init = (jnp.zeros_like(q), jnp.ones_like(q), 0)
    low, high, _ = lax.while_loop(cond_fn, body_fn, init)
    return (low + high) / 2

@jax.jit
def clopper_pearson_intervals(k:int, n:int, alpha:float=0.05):
    """
    Compute Clopper-Pearson confidence intervals for a binomial proportion.
    Args:
        k: number of successes
        n: number of trials
        alpha: significance level (default 0.05 for 95% CI)

    Returns:
        (lower_bound, upper_bound): tuple of lower and upper bounds of the confidence interval

    """
    k = jnp.asarray(k, dtype=jnp.float32)
    n = jnp.asarray(n, dtype=jnp.float32)

    def case_k0(_):
        upper = beta_ppf_approx(1 - alpha / 2, 1.0, n - k + 1)
        return 0.0, upper

    def case_kn(_):
        lower = beta_ppf_approx(alpha / 2, k + 1, 1.0)
        return lower, 1.0

    def case_else(_):
        lower = beta_ppf_approx(alpha / 2, k, n - k + 1)
        upper = beta_ppf_approx(1 - alpha / 2, k + 1, n - k)
        return lower, upper

    return lax.cond(
        k == 0,
        case_k0,
        lambda _: lax.cond(k == n, case_kn, case_else, None),
        operand=None
    )