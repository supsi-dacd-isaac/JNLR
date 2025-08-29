import jax.numpy as jnp
import jax
from functools import partial

@jax.jit
def tangent_space_basis(nu: jnp.ndarray):
    r"""Return an orthonormal basis for the tangent space orthogonal to $nu ∈ R^n$"""
    n = nu.shape[0]
    eye = jnp.eye(n)
    A = jnp.concatenate([nu[None, :], eye], axis=0)
    Q, _ = jnp.linalg.qr(A.T)
    return Q[:, 1:]  # Drop the component along nu

def tangent_space_basis_vv(jacobian: jnp.ndarray):
    Q, _ = jnp.linalg.qr(jacobian.T, mode='complete')
    return Q[:, jacobian.shape[0]:]


@partial(jax.jit, static_argnames=('grad_f', 'hessian_f'))
def min_tangent_eigenvalue(grad_f, hessian_f, z: jnp.ndarray) -> jnp.ndarray:
    g = grad_f(z)
    norm = jnp.linalg.norm(g)

    def compute():
        nu = g / norm
        E = tangent_space_basis(nu)  # (n, n-1)
        H = hessian_f(z)
        H_restricted = E.T @ H @ E
        eigvals = jnp.linalg.eigvalsh(H_restricted)
        return jnp.min(eigvals)

    return jax.lax.cond(norm > 1e-6, compute, lambda: jnp.nan)


# ---------- NEW: solve for λ ----------
def solve_lagrange_multipliers(J: jnp.ndarray,
                               delta_pi: jnp.ndarray,
                               reg=1e-10):
    r"""
    Solve
    $$
    J J^T λ = -2 J δ_π
    $$
    for λ  (m×m system).
    A small Tikhonov term reg·I protects against rank‑deficiency.


    Args:
        J: Jacobian matrix of shape (m, n)
        delta_pi: Difference vector of shape (n,)
        reg: Regularization parameter

    Returns:
        λ: Lagrange multipliers of shape (m,)

    """

    JJt = J @ J.T
    JJt_reg = JJt + reg * jnp.eye(JJt.shape[0], dtype=JJt.dtype)
    rhs = -2.0 * (J @ delta_pi)       # shape (m,)
    return jnp.linalg.solve(JJt_reg, rhs)   # λ ∈ R^m


@partial(jax.jit, static_argnames=('jacobian_f', 'hessian_f'))
def min_tangent_eigenvalue_vv(f, jacobian_f, hessian_f,  z_tilde: jnp.ndarray, z_hat: jnp.ndarray, eps=1e-6) -> jnp.ndarray:
    r"""
    Generalization of min_tangent_eigenvalue to vector-valued constraint functions $F: R^n -> R^m$.
    Computes the minimum eigenvalue of the combined Hessian projected to the tangent space.

    Args:
        f: Constraint function
        jacobian_f: Function returning (m, n) Jacobian matrix DF(z)
        hessian_f: Function returning (m, n, n) Hessians of each component function f_i
        z_tilde: Point at which to compute the curvature (projection of z_hat onto constraint surface)
        z_hat: Original point before projection
        eps: Small constant to avoid division by zero
    Returns:
        A pair (min_eigenvalue, ratio) where:
            - min_eigenvalue: Minimum eigenvalue of the combined Hessian projected to the tangent space
            - ratio: Ratio used for assessing the curvature condition
    """
    delta_pi = z_tilde - z_hat
    J = jacobian_f(z_tilde)       # (m, n)
    grad_norms = jnp.linalg.norm(J, axis=1)
    valid_mask = jnp.all(grad_norms > 1e-6)

    # ---------- original sufficient condition (safe) ----------
    def compute_delta():
        E = tangent_space_basis_vv(J)
        Hs = hessian_f(z_tilde)
        H_tan_sum = jnp.zeros((E.shape[1], E.shape[1]), dtype=z_tilde.dtype)
        delta_proj = J @ delta_pi  # (m,)
        kappa_max = 0.0
        for i in range(Hs.shape[0]):
            H_proj = E.T @ Hs[i] @ E
            weight = -delta_proj[i] / (grad_norms[i] + 1e-8)
            H_tan_sum += weight * H_proj
            lam_abs = jnp.max(jnp.abs(jnp.linalg.eigvalsh(H_proj)))
            kappa_i = lam_abs / (grad_norms[i] + eps)
            kappa_max = jnp.maximum(kappa_max, kappa_i)

        lam_min = jnp.linalg.eigvalsh(H_tan_sum)[0]

        threshold = kappa_max * jnp.linalg.norm(delta_pi)
        our_threshold = jnp.abs(lam_min) * jnp.linalg.norm(delta_pi)
        return jnp.array([lam_min>threshold, threshold])

    # ---------- equivalent λ‑based implementation ----------
    def compute_lambda():
        E = tangent_space_basis_vv(J)
        Hs = hessian_f(z_tilde)

        lam = solve_lagrange_multipliers(J, delta_pi)  # (m,)
        mixed_vec = (J @ J.T) @ lam  # (m,)

        H_tan_sum = jnp.zeros((E.shape[1], E.shape[1]), dtype=z_tilde.dtype)
        for i in range(Hs.shape[0]):
            H_proj = E.T @ Hs[i] @ E
            weight = 0.5 * mixed_vec[i] / (grad_norms[i] + 1e-8)
            H_tan_sum += weight * H_proj

        eigvals = jnp.linalg.eigvalsh(H_tan_sum)
        lam_max = jnp.max(-eigvals)
        ratio = 1.0 / (jnp.abs(lam_max) + eps) / (jnp.linalg.norm(delta_pi) + eps)
        return jnp.array([eigvals[0], ratio])

    # choose either implementation; both are equivalent and safe
    return jax.lax.cond(valid_mask, compute_delta, lambda: jnp.ones(2) * jnp.nan)


@partial(jax.jit, static_argnames=('hessian_f'))
def min_eigenvalue(hessian_f, z: jnp.ndarray) -> jnp.ndarray:
    r"""
    Compute the minimum eigenvalue of the Hessian of $f$ at $z$.


    Args:
        hessian_f: Function returning the Hessian matrix of shape (n, n)
        z: Point at which to compute the Hessian
    Returns:
        Minimum eigenvalue of the Hessian

    """
    H = hessian_f(z)
    eigvals = jnp.linalg.eigvalsh(H)
    return jnp.min(eigvals)


@partial(jax.jit, static_argnames=('hessian_f'))
def curvature_along_projection(hessian_f, z_tilde: jnp.ndarray, delta_tilde: jnp.ndarray) -> jnp.ndarray:

    def compute():
        return delta_tilde.T @ hessian_f(z_tilde) @ delta_tilde

    return compute()
