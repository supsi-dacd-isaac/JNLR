import jax
import jax.numpy as jnp
from jax import jacfwd

from typing import Optional
from jnlr.utils.function_utils import infer_io_shapes

def sqrtdet_G(J):
    # J: (..., D, d); G = J^T J
    G = jnp.swapaxes(J, -1, -2) @ J     # (..., d, d)
    sign, logdet = jnp.linalg.slogdet(G)
    return jnp.exp(0.5 * logdet)        # (...,)

def gumbel_top_k_indices(key, logits, k):
    # Weighted sampling w/o replacement ~ Plackett–Luce
    g = jax.random.gumbel(key, logits.shape)
    z = logits + g
    # take top-k indices (unique by construction)
    idx = jnp.argpartition(z, -k)[-k:]
    # optional: sort them by z descending
    idx = idx[jnp.argsort(z[idx])[::-1]]
    return idx

def _make_candidates(key, bounds, M):
    d = bounds.shape[0]
    u01 = jax.random.uniform(key, (M, d))
    U = bounds[:, 0] + u01 * (bounds[:, 1] - bounds[:, 0])
    return U


def _preprocess_phi(phi0, d_input=None, key=None, bounds:Optional[jnp.ndarray]=None):
    if key is None:
        key = jax.random.PRNGKey(0)

    phi = lambda u: jnp.hstack([u, phi0(u)])

    d_input, d_tot = infer_io_shapes(phi,d_input=d_input)
    d_input = d_input[0]
    d_tot = d_tot[0]

    if bounds is None:
        bounds = jnp.array([[-1.0, 1.0]] * int(d_input))
    else:
        bounds = jnp.asarray(bounds)
    phi_v  = jax.vmap(phi)
    return bounds, d_input, d_tot, phi, phi_v, key


def volume_expl(phi0, key=None, bounds:Optional[jnp.ndarray]=None, N=1000,
                             oversample=6, roi_R=None, min_pool=8, d_input=None):
    """
    Unique samples (no duplicates) ~ volume measure induced by y=phi(u) over
    the param box, optionally restricted to ||y|| <= roi_R.
    Grows candidate pool until enough valid points exist.
    """

    bounds, d, d_tot, phi, phi_v, key = _preprocess_phi(phi0, d_input=d_input, key=key, bounds=bounds)

    Jphi = jacfwd(phi)
    Jphi_v = jax.vmap(Jphi)

    M = max(int(N * oversample), N + min_pool)
    key_pool = key
    Ys = jnp.empty((0, d_tot))
    Us = jnp.empty((0, bounds.shape[0]))
    Ws = jnp.empty((0,))

    # Keep drawing candidates until we have ≥ N valid
    total_valid = 0
    while total_valid < N:
        key_pool, kU = jax.random.split(key_pool)
        U = _make_candidates(kU, bounds, M)            # (M, d)
        Y = phi_v(U)                                   # (M, d_tot)
        J = Jphi_v(U)                                  # (M, d_tot, d)
        w = sqrtdet_G(J)                               # (M,)

        if roi_R is not None:
            inside = (jnp.linalg.norm(Y, axis=1) <= roi_R)
            w = jnp.where(inside, w, 0.0)

        # Keep only positive-weight candidates
        valid = w > 0.0
        Uv = U[valid]; Yv = Y[valid]; Wv = w[valid]

        # Accumulate
        Us = jnp.concatenate([Us, Uv], axis=0)
        Ys = jnp.concatenate([Ys, Yv], axis=0)
        Ws = jnp.concatenate([Ws, Wv], axis=0)
        total_valid = int(Ws.shape[0])

        # If still short, bump pool size (geometric growth)
        if total_valid < N:
            M = max(M * 2, N - total_valid + min_pool)

    # Weighted sampling WITHOUT replacement on the valid pool
    key_pick = jax.random.split(key_pool, 1)[0]
    logits = jnp.log(jnp.clip(Ws, 1e-30))              # finite for valid only
    # (no need to mask here: all Ws > 0)
    idx = gumbel_top_k_indices(key_pick, logits, N)
    return Ys[idx], Us[idx]


def latin_hypercube(bounds, n_samples=100):
    d = bounds.shape[0]
    # Generate the intervals
    intervals = jnp.linspace(0, 1, n_samples + 1)
    # Randomly sample within each interval
    u = jax.random.uniform(jax.random.PRNGKey(0), (n_samples, d))
    points = intervals[:-1, None] + u * (intervals[1:, None] - intervals[:-1, None])
    # Shuffle the points in each dimension
    for i in range(d):
        points = points.at[:, i].set(jax.random.permutation(jax.random.PRNGKey(i + 1), points[:, i]))
    # Scale points to the bounds
    scaled_points = bounds[:, 0] + points * (bounds[:, 1] - bounds[:, 0])
    return scaled_points


def randinput_expl(phi0, key=None, bounds:Optional[jnp.ndarray]=None, N=1000,
                             oversample=6, roi_R=None, min_pool=8, d_input=None):
    bounds, d, d_tot, phi, phi_v, key = _preprocess_phi(phi0, d_input=d_input, key=key, bounds=bounds)
    x_rand = latin_hypercube(bounds, n_samples=N)
    return phi_v(x_rand)


import jax.scipy as jsp
from functools import partial


# Build tangent projection ops at point x for F(x)=0
def build_tangent_ops_chol(x, JF):
    J = jnp.atleast_2d(JF(x))                         # (c, D)
    A = J @ J.T                       # (c, c) SPD if rank(J)=c
    L = jnp.linalg.cholesky(A)        # factor once

    def proj_vec(v):                  # v: (D,)
        lam = jsp.linalg.cho_solve((L, True), J @ v)    # (c,)
        return v - J.T @ lam

    def proj_mat(V):                  # V: (D, k)
        lam = jsp.linalg.cho_solve((L, True), J @ V)    # (c, k)
        return V - J.T @ lam

    return proj_vec, proj_mat

def build_tangent_ops_qr(x, JF):
    J = jnp.atleast_2d(JF(x))                         # (c, D)
    # Reduced QR of J^T gives an orthonormal basis of the normal space
    Q, R = jsp.linalg.qr(J.T, mode="economic")   # Q: (D, r), r = rank(J)
    def proj_vec(v): return v - Q @ (Q.T @ v)
    def proj_mat(V): return V - Q @ (Q.T @ V)
    return proj_vec, proj_mat


def newton_project(x, F, JF, iters=6, method="chol"):
    def step(x,_):
        J = jnp.atleast_2d(JF(x)); r = jnp.atleast_1d(F(x))                       # (c,D), (c,)
        if method == "qr":
            # Solve (J J^T) lam = r via QR of J^T
            Q, R = jsp.linalg.qr(J.T, mode="economic")
            lam = jsp.linalg.solve_triangular(R.T, Q.T @ r, lower=True)
        else:
            A = J @ J.T
            lam = jsp.linalg.solve(A, r, assume_a="pos")
        return x - J.T @ lam, None
    x, _ = jax.lax.scan(step, x, None, length=iters)
    return x

def _clip_to_ball(x0, y, F, JF, R, iters=14):
    def inside(z):
        return jnp.linalg.norm(z) <= R

    def bisect(_y):
        lo = jnp.array(0.0, dtype=y.dtype)
        hi = jnp.array(1.0, dtype=y.dtype)

        def body_fun(_, carry):
            lo, hi = carry
            mid = 0.5 * (lo + hi)
            z = newton_project(x0 + mid * (y - x0), F, JF)
            cond = inside(z)
            lo_new = jnp.where(cond, mid, lo)
            hi_new = jnp.where(cond, hi,  mid)
            return (lo_new, hi_new)

        lo, hi = jax.lax.fori_loop(0, iters, body_fun, (lo, hi))
        return newton_project(x0 + lo * (y - x0), F, JF)

    # Pass operand `y` and make both branches accept it
    return jax.lax.cond(inside(y), lambda yy: yy, bisect, y)

@partial(jax.jit, static_argnames=("F","JF","proj_method","newton_iters"))
def roi_langevin_step(key, x, F, JF, *, sigma=1e-2, lam=0.0, kappa=0.0, R=jnp.inf,
                      newton_iters=6, proj_method="chol"):
    # Build projector ops at current x (reuse within this step)
    build = build_tangent_ops_qr if proj_method == "qr" else build_tangent_ops_chol
    proj_vec, proj_mat = build(x, JF)

    xi = jax.random.normal(key, x.shape)     # ambient noise
    inward = proj_vec(-x)                    # tangent "towards origin"
    v_rand = proj_vec(xi)                    # random tangent direction
    v_dir  = v_rand + kappa * inward
    drift  = -lam * inward                   # soft quadratic well

    y = newton_project(x + sigma * (v_dir + drift), F, JF,
                       iters=newton_iters, method=proj_method)
    y = _clip_to_ball(x, y, F, JF, R)
    return y

# ---------- Convenience: draw a chain ----------
def langevin_implicit(phi, *, n=2000, burn=200, thin=1,
                     sigma=5e-2, lam=0.0, kappa=0.5, R=jnp.inf, newton_iters=6,key=None, x0=None):

    """Return n samples after burn-in/thinning from M ∩ B(0,R)."""

    J_phi = jax.jacfwd(phi)

    d_tot, d_out = infer_io_shapes(phi)
    d_tot = d_tot[0]

    if x0 is None:
        x0 = jnp.zeros((d_tot,), dtype=jnp.float32)
        x0 = x0.at[:d_tot].set(jax.random.uniform(jax.random.PRNGKey(0), (d_tot,), minval=-1.0, maxval=1.0))
        # Project to the manifold
        x0 = newton_project(x0, phi, J_phi, iters=newton_iters)

    print(x0)

    if key is None:
        key = jax.random.PRNGKey(0)

    total = burn + n * thin
    keys  = jax.random.split(key, total)

    def one(x, k):
        y = roi_langevin_step(k, x, F=phi, JF=J_phi, sigma=sigma, lam=lam, kappa=kappa, R=R, newton_iters=newton_iters)
        return y, y
    xf, path = jax.lax.scan(one, x0, keys)
    kept = path[burn::thin] # shape (n, D)
    return kept


from jnlr.utils.manifolds import f_ackley as phi_expl
from jnlr.utils.function_utils import f_impl

phi = f_impl(phi_expl)
samples = langevin_implicit(phi, n=500, burn=100, thin=2, sigma=0.1, lam=0.01, kappa=0.1, R=3.0)
