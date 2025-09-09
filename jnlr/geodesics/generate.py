import jax
import jax.numpy as jnp
import diffrax as dfx
from typing import Optional
from jnlr.utils.log_utils import configure_logging
from jnlr.utils.function_utils import infer_io_shapes, f_impl
from jnlr.utils.manifolds import f_paraboloid


logger = configure_logging(__name__)

def trapezoid_integral(y, x):
    dx = x[1:] - x[:-1]
    return jnp.sum(0.5 * (y[1:] + y[:-1]) * dx)

def trapezoid_cumulative(y, x):
    dx = x[1:] - x[:-1]
    inc = 0.5 * (y[1:] + y[:-1]) * dx
    return jnp.concatenate([jnp.array([0.0]), jnp.cumsum(inc)])

def project_velocity_to_tangent(J, x, v, eps=1e-10):
    """
    Tangent projection for codimension c:
    v_tan = v - J^T (JJ^T)^{-1} (J v).
    """
    J_x   = J(x)                   # (c, D)
    rhs = J_x @ v                              # (c,)
    JJt = J_x @ J_x.T + eps * jnp.eye(J_x.shape[0])
    y   = jnp.linalg.solve(JJt, rhs)         # (c,)
    return v - J_x.T @ y                       # (D,)

def vHv_all(f_vec, x, v):
    def single_fi(fi):
        return jax.jvp(lambda x: jnp.dot(jax.grad(fi)(x), v), (x,), (v,))[1]
    return jax.vmap(lambda i: single_fi(lambda z: f_vec(z)[i]))(jnp.arange(f_vec(x).shape[0]))

def make_geodesic_rhs_soa(f_vec):
    """
    Create RHS function for geodesic ODE on the manifold defined by f_vec(x)=0.
    Uses second-order approximation (SOA) for acceleration.
    Args:
        f_vec: (vector-valued) function defining the manifold

    Returns:

    """
    J = jax.jacfwd(f_vec)

    @jax.jit
    def rhs(t, y, args=None):
        D = y.shape[0] // 2
        x, v = y[:D], y[D:]                          # (n,)
        Jx = J(x)                                    # (m,n)

        # b_i = v^T H_i v
        b = vHv_all(f_vec, x, v)
        G = Jx @ Jx.T + 1e-10*jnp.eye(Jx.shape[0])   # (m,m)
        lam = jnp.linalg.solve(G, b)                 # (m,)
        a = -(Jx.T @ lam)                            # (n,)
        return jnp.concatenate([v, a])
    return rhs


def integrate_geodesic_implicit_generic(
    F, x0, v0, t1=1.0, n_steps=400, project_init=True, project_after=True):
    """
    SECOND ORDER APPROXIMATION
    Integrate a geodesic on {x : F(x)=0}, F: R^D -> R^c (vector-valued).
    Inputs:
      x0 : (D,)  initial point (projected to surface if project_init=True)
      v0 : (D,)  initial velocity (projected to tangent if project_init=True)
    Returns:
      ts      : (n_steps,)
      X       : (n_steps, D) points on the manifold
      V       : (n_steps, D) velocities (ambient)
      L_total : scalar geodesic length
      L_cum   : (n_steps,) cumulative length
    """
    J = jax.jacfwd(F)                 # (c, D)
    if project_init:
        x0 = project(x0)
        v0 = project_velocity_to_tangent(J, x0, v0)

    y0 = jnp.concatenate([x0, v0])
    ts = jnp.linspace(0.0, t1, n_steps)

    rhs = make_geodesic_rhs_soa(F)
    term   = dfx.ODETerm(rhs)
    solver = dfx.Tsit5()
    sol = dfx.diffeqsolve(
        term, solver,
        t0=0.0, t1=t1, dt0=t1/n_steps,
        y0=y0, saveat=dfx.SaveAt(ts=ts),
        adjoint=dfx.DirectAdjoint(),
    )
    D = x0.shape[0]
    Y = sol.ys
    X, V = Y[:, :D], Y[:, D:]

    if project_after:
        # project points back to F=0
        X = project(X)
        # reproject velocities to tangent at the corrected points
        D = X.shape[-1]
        packed = jnp.concatenate([X, V], axis=-1)     # (n_steps, 2D)
        V = jax.vmap(lambda xv: project_velocity_to_tangent(J, xv[:D], xv[D:]))(packed)

    speeds  = jnp.linalg.norm(V, axis=-1)    # ||x'(t)|| in ambient → geodesic speed
    L_total = trapezoid_integral(speeds, ts)
    L_cum   = trapezoid_cumulative(speeds, ts)
    return ts, X, V, L_total, L_cum


def generate_geodesics(
    f_explicit: Optional = None,
    f_implicit: Optional = None,
    p0: Optional[jnp.ndarray] = None,
    v0: Optional[jnp.ndarray] = None,
    sampling_method:str = 'latin_hypercube',
    n_geodesics: int = 100,
    method: str = 'int_soa',
) -> None:
    """
    Generate geodesics on the manifold defined by f_impl(x)=0.
    Args:
        f_explicit: explicit function defining the manifold.
        p0: Initial point on the manifold (if None, a default point is used).
        v0: Initial velocity (if None, a default velocity is used).
        method: Method to use for geodesic integration ('int_soa' for second-order approximation).

    Returns:
        ts: Time steps of the geodesic.
        X: Points on the manifold along the geodesic.
        V: Velocities along the geodesic.
        L_total: Total length of the geodesic.
        L_cum: Cumulative length along the geodesic.
    """
    logger.info("Generating geodesic...")

    if f_implicit is None and f_explicit is None:
        raise ValueError("Either f_implicit or f_explicit must be provided.")
    if f_explicit is not None and f_implicit is not None:
        logger.warning("Both f_explicit and f_implicit provided; using f_implicit.")

    f_implicit = f_impl(f_explicit) if f_implicit is None else f_implicit

    # Default initial point and velocity if not provided
    input_shape, output_shape = infer_io_shapes(f_implicit)
    logger.info(f"Inferred input shape: {input_shape}, output shape: {output_shape}")


    # if p0 and v0 are provided, use them directly, otherwise sample them using the specified method
    if p0 is not None and v0 is not None:
        x0 = p0
        v0 = v0
    else:
        if sampling_method == 'latin_hypercube':
            from jnlr.utils.sampling import latin_hypercube_sampling
            x0 = latin_hypercube_sampling(n_geodesics, input_shape[0], low=-1.0, high=1.0)
            v0 = latin_hypercube_sampling(n_geodesics, input_shape[0], low=-1.0, high=1.0)
        elif sampling_method == 'random':
            key = jax.random.PRNGKey(0)
            x0 = jax.random.uniform(key, (n_geodesics, input_shape[0]), minval=-1.0, maxval=1.0)
            v0 = jax.random.uniform(key, (n_geodesics, input_shape[0]), minval=-1.0, maxval=1.0)
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")

        # Project initial points and velocities
        x0 = jax.vmap(lambda x: project_point_to_surface(f_implicit, x))(x0)
        v0 = jax.vmap(lambda xv: project_velocity_to_tangent(f_implicit, xv[0], xv[1]))(jnp.concatenate([x0, v0], axis=-1))


    return ts, X, V, L_total, L_cum




if __name__ == "__main__":
    # Example usage with the paraboloid function
    generate_geodesics(f_paraboloid)