from typing import Callable, Tuple
import jax
import jax.numpy as jnp
import diffrax

# Enable float64 for better conditioning
from jax import config as _jax_config
_jax_config.update("jax_enable_x64", True)

# 2. Compute metric and Christoffel symbols
def compute_metric(f, u):
    jac_f = jax.jacobian(f)
    J = jac_f(u)
    g = J.T @ J
    return g



def compute_christoffel(f, u):
    g = compute_metric(f, u)
    g_inv = jnp.linalg.inv(g)
    def g_ij(i, j):
        return lambda u: compute_metric(f, u)[i, j]
    dg = {}
    for i in range(2):
        for j in range(2):
            dg[(i,j)] = jax.jacfwd(g_ij(i,j))(u)
    Gamma = jnp.zeros((2,2,2))
    for k in range(2):
        for i in range(2):
            for j in range(2):
                term = 0.0
                for l in range(2):
                    term += g_inv[k,l] * (dg[(j,l)][i] + dg[(i,l)][j] - dg[(i,j)][l])
                Gamma = Gamma.at[k,i,j].set(0.5 * term)
    return Gamma

# 3. Define geodesic ODE

def geodesic_rhs(t, state, args):
    f = args
    u, du = state[:2], state[2:]
    Gamma = compute_christoffel(f, u)
    d2u = jnp.zeros(2)
    for k in range(2):
        for i in range(2):
            for j in range(2):
                d2u = d2u.at[k].add(-Gamma[k,i,j] * du[i] * du[j])
    return jnp.concatenate([du, d2u])



# JIT compile geodesic RHS
geodesic_rhs = jax.jit(geodesic_rhs, static_argnums=(2,))


# 4. Geodesic integrator
def integrate_geodesic(f, u0, v0, t_max=1.0, n_steps=100):
    state0 = jnp.concatenate([u0, v0])
    term = diffrax.ODETerm(geodesic_rhs)
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, t_max, n_steps))
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0,
        t1=t_max,
        dt0=t_max/n_steps,
        y0=state0,
        args=f,
        saveat=saveat,
        adjoint=diffrax.DirectAdjoint()
    )
    return sol.ys

import optax
def shooting_loss(v0, f, u0, u1, t_max=1.0, n_steps=20):
    sol = integrate_geodesic(f, u0, v0, t_max, n_steps)
    final_u = sol[-1, :2]
    return jnp.sum((final_u - u1)**2)


shooting_loss = jax.jit(shooting_loss, static_argnames=("f", "n_steps"))

loss_grad_fn = jax.value_and_grad(shooting_loss, argnums=0)


def optimize_shooting_optax(f, u0, u1, t_max=1.0, n_steps=20, iters=100, lr=0.01):
    v0 = jnp.ones_like(u0)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(v0)

    @jax.jit
    def step(v0, opt_state):
        loss, grad = loss_grad_fn(v0, f, u0, u1, t_max, n_steps)
        updates, opt_state = optimizer.update(grad, opt_state)
        v0 = optax.apply_updates(v0, updates)
        return v0, opt_state, loss

    for _ in range(iters):
        v0, opt_state, loss = step(v0, opt_state)
    return v0


def shooting_distance(f, u0, u1, t_max=1.0, n_steps=20):
    """
    Computes geodesic distance and path between u0 and u1 on the manifold defined by f(u,v).

    Returns:
      distance: scalar geodesic distance
      path: (n_steps, dim) trajectory along geodesic
    """
    v0_opt = optimize_shooting_optax(f, u0, u1, t_max=t_max, n_steps=n_steps, iters=300, lr=0.05)  # use few steps for optimization
    path = integrate_geodesic(f, u0, v0_opt, t_max=t_max, n_steps=n_steps)  # fine solve
    du = path[1:, :2] - path[:-1, :2]  # finite differences # TODO: make generic to d dimensions
    distances = []
    for i in range(du.shape[0]):
        u = path[i, :2] # TODO: make generic to d dimensions
        g = compute_metric(f, u)
        dsq = du[i].T @ g @ du[i]
        # Clamp to avoid negative rounding and NaNs in sqrt
        dsq = jnp.maximum(dsq, 0.0)
        distances.append(jnp.sqrt(dsq))
    total_distance = jnp.sum(jnp.stack(distances))
    embedded_path = jax.vmap(lambda uv: f(uv))(path[:, :2])
    return embedded_path, total_distance
