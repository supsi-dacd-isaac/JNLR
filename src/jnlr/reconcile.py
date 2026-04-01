import jax.numpy as jnp
import jax
from jax import lax
import optax
from jnlr.utils.function_utils import infer_io_shapes


def make_solver_alm_optax(
    f,
    w:jnp.ndarray = None,
    # --- ALM (outer) ---
    n_iterations: int = 30,
    tol_feas: float = 1e-8,
    rho0: float = 0.9,
    rho_mult: float = 10.0,
    rho_increase_thresh: float = 0.25,  # bump rho if ||f|| > thresh*tol_feas

    # --- LBFGS (inner) ---
    max_inner: int = 100,
    tol_grad: float = 1e-6,
    tol_step: float = 1e-10,
    lbfgs_learning_rate=None,  # let line search pick step by default
    lbfgs_memory_size: int = 10,
    ls_max_steps: int = 25,  # zoom line search budget

    # numerics
    eps_chol: float = 1e-12,
    return_history: bool = False,
    vmapped: bool = True,
):
    r"""
    Returns: proj(zhat_batch) -> z_proj_batch
    Projects onto ${z : f(z)=0}$ in metric W using ALM + Optax L-BFGS (zoom line search).
    """

    # --- Whitening W = L^T L ---
    if w is None:
        input_d, output_d = infer_io_shapes(f)
        W = jnp.eye(input_d[0])
    else:
        W = jnp.asarray(w)

    W = jnp.asarray(W)
    W = 0.5 * (W + W.T)
    n = W.shape[0]
    L = jnp.linalg.cholesky(W + eps_chol * jnp.eye(n))
    Linv = jnp.linalg.solve(L, jnp.eye(n))

    # Build the (static) optimizer once; the objective is provided per-outer-iter
    if lbfgs_learning_rate is None:
        # legacy: use zoom (slower)
        linesearch = optax.scale_by_zoom_linesearch(max_linesearch_steps=ls_max_steps)
        solver = optax.lbfgs(
            learning_rate=None,
            memory_size=lbfgs_memory_size,
            scale_init_precond=True,
            linesearch=linesearch,
        )
    else:
        # FAST path: fixed step, NO line search
        solver = optax.lbfgs(
            learning_rate=lbfgs_learning_rate,  # e.g., 1.0
            memory_size=lbfgs_memory_size,
            scale_init_precond=True,
            linesearch=None,  # <- critical
        )
    def feas_norm_y(y):
        z = Linv @ y
        return jnp.linalg.norm(jnp.atleast_1d(f(z)))

    def inner_minimize(y_init, yhat, lam, rho):
        r"""
        Minimize $$L(y; lam, rho) = 0.5||y - yhat||^2 + lam^T c + 0.5*rho*||c||^2$$
        with Optax L-BFGS + zoom LS.
        """
        # Objective (captures yhat, lam, rho)
        def L_value(y):
            z = Linv @ y
            c = jnp.atleast_1d(f(z))
            d = y - yhat
            return 0.5 * (d @ d) + jnp.dot(lam, c) + 0.5 * rho * (c @ c)

        # Initialize optimizer state
        y = y_init
        opt_state = solver.init(y)

        def cond_inner(state):
            k, y, opt_state, step, gnorm = state
            return (k < max_inner) & ((gnorm > tol_grad) | (step > tol_step))

        def body_inner(state):
            k, y, opt_state, _, _ = state
            # Provide value & grad explicitly; pass value_fn for the linesearch
            value, grad = jax.value_and_grad(L_value)(y)
            updates, opt_state = solver.update(
                grad, opt_state, y, value=value, grad=grad, value_fn=L_value
            )
            y_new = optax.apply_updates(y, updates)
            step = jnp.linalg.norm(optax.tree_utils.tree_norm(updates))
            gnorm = jnp.linalg.norm(grad, ord=jnp.inf)
            return (k + 1, y_new, opt_state, step, gnorm)

        k0 = jnp.array(0)
        step0 = jnp.array(jnp.inf)
        gnorm0 = jnp.array(jnp.inf)
        _, y_fin, _, _, _ = lax.while_loop(cond_inner, body_inner, (k0, y, opt_state, step0, gnorm0))
        return y_fin


    def solve_single(zhat):
        yhat = L @ zhat
        y = yhat
        # initialize lambda without Python int casts (JAX-safe)
        lam = jnp.zeros_like(jnp.atleast_1d(f(Linv @ y)))
        rho = jnp.array(rho0)

        if return_history:
            # fixed-length scan; freeze updates after convergence
            def body(carry, _):
                k, y, lam, rho = carry
                active = feas_norm_y(y) > tol_feas

                y_new = lax.cond(
                    active,
                    lambda _: inner_minimize(y, yhat, lam, rho),
                    lambda _: y,
                    operand=None,
                )
                z_new = Linv @ y_new
                c_new = jnp.atleast_1d(f(z_new))
                lam_prop = lam + rho * c_new
                rho_prop = jnp.where(
                    jnp.linalg.norm(c_new) > rho_increase_thresh * tol_feas,
                    rho * rho_mult,
                    rho,
                )

                y_next = jnp.where(active, y_new, y)
                lam_next = jnp.where(active, lam_prop, lam)
                rho_next = jnp.where(active, rho_prop, rho)
                return (k + 1, y_next, lam_next, rho_next), (Linv @ y_next)

            (_, _, _, _), z_hist = lax.scan(body, (0, y, lam, rho), xs=None, length=n_iterations)
            return z_hist  # (T, n)
        else:
            # original early-stop loop; return only the final iterate
            def cond_outer(state):
                k, y, lam, rho = state
                return (k < n_iterations) & (feas_norm_y(y) > tol_feas)

            def body_outer(state):
                k, y, lam, rho = state
                y_new = inner_minimize(y, yhat, lam, rho)
                z_new = Linv @ y_new
                c_new = jnp.atleast_1d(f(z_new))
                lam_new = lam + rho * c_new
                rho_new = jnp.where(
                    jnp.linalg.norm(c_new) > rho_increase_thresh * tol_feas,
                    rho * rho_mult,
                    rho,
                )
                return (k + 1, y_new, lam_new, rho_new)

            _, y_opt, _, _ = lax.while_loop(cond_outer, body_outer, (0, y, lam, rho))
            return Linv @ y_opt



    if vmapped:
        return jax.jit(jax.vmap(solve_single))
    else:
        return jax.jit(solve_single)



def make_solver(f,
                w:jnp.ndarray = None,
                n_iterations: int = 50,
                damping: float = 1e-5,          # a bit larger for f32; reduce if using x64
                beta: float = 0.5,              # backtracking factor
                c_armijo: float = 1e-4,         # sufficient decrease on ||f||
                max_bt: int = 12,
                return_history: bool = False,
                vmapped: bool = True):
    r"""
    Create a v-mapped and JIT-compiled solver function for the constrained optimization problem.
    Here f is the implicit function representing the manifold constraints: $M = \{ z : f(z) = 0 \}$.
    The returned function takes $\hat z$ as input and returns the projected z.

    $$\text{arg}\min_{z} \tfrac{1}{2} (z - \hat{z})^T W (z - \hat{z})$$

    $$\text{s.t. } f(z) = 0$$

    Args:
        f: Function representing the constraints, in implicit form. The signature of f should be $f(z): \mathbb{R}^n \rightarrow \mathbb{R}^m$ where $n$ is the dimension of the input and m the output.
        W: Weight matrix.
        n_iterations:  Number of iterations for the learning process.

    Returns:
        A JIT-compiled function that takes z_hat as input and returns the projected z.
    """

    n_input, n_constraints = infer_io_shapes(f)

    if w is None:
        W = jnp.eye(n_input[0])
    else:
        W = jnp.asarray(w)
    jac_f = jax.jacfwd(f)

    #n_constraints = jnp.atleast_1d(f(jnp.zeros(W.shape[0]))).shape[0]

    def step(z, lam, zhat):
        J = jnp.atleast_2d(jac_f(z))  # (m,n)
        g =  W @ (z - zhat) + J.T @ lam  # stationarity residual
        cvec = jnp.atleast_1d(f(z))  # constraint residual (m,)

        H = W + damping * jnp.eye(W.shape[0])  # (n,n)
        Hinv_g = jnp.linalg.solve(H, g)
        Hinv_Jt = jnp.linalg.solve(H, J.T)
        S = J @ Hinv_Jt + damping * jnp.eye(J.shape[0])  # (m,m)

        dlam = jnp.linalg.solve(S, cvec - J @ Hinv_g)  # (m,)
        dz = -(Hinv_g + Hinv_Jt @ dlam)  # (n,)
        return dz, dlam, cvec

    def backtrack(z, dz, c0):
        # reduce alpha until ||f(z+alpha dz)|| <= (1 - c_armijo*alpha) ||f(z)||
        alpha0 = 1.0
        fnorm0 = jnp.linalg.norm(c0)

        def body_fun(state):
            k, alpha = state
            zt = z + alpha * dz
            fnorm = jnp.linalg.norm(jnp.atleast_1d(f(zt)))
            ok = fnorm <= (1.0 - c_armijo * alpha) * fnorm0
            alpha = jnp.where(ok, alpha, beta * alpha)
            return (k + 1, alpha)

        def cond_fun(state):
            k, alpha = state
            zt = z + alpha * dz
            fnorm = jnp.linalg.norm(jnp.atleast_1d(f(zt)))
            return (k < max_bt) & (fnorm > (1.0 - c_armijo * alpha) * fnorm0)

        _, alpha = lax.while_loop(cond_fun, body_fun, (0, alpha0))
        return alpha
    
    if return_history:
        def solve_single(zhat: jnp.ndarray) -> jnp.ndarray:
            z = zhat
            lam = jnp.zeros(n_constraints)

            def body(carry, _):
                z, lam = carry
                dz, dlam, cvec = step(z, lam, zhat)
                alpha = backtrack(z, dz, cvec)
                return (z + alpha * dz, lam + alpha * dlam), z + alpha * dz

            (z_final, _), zs = lax.scan(body, (z, lam), xs=None, length=n_iterations)
            return zs
    else:
        def solve_single(zhat: jnp.ndarray) -> jnp.ndarray:
            z = zhat
            lam = jnp.zeros(n_constraints)

            def body_fn(_, state):
                z, lam = state
                dz, dlam, cvec = step(z, lam, zhat)
                alpha = backtrack(z, dz, cvec)
                return (z + alpha * dz, lam + alpha * dlam)

            z_final, _ = lax.fori_loop(0, n_iterations, body_fn, (z, lam))
            return z_final

    if vmapped:
        return jax.jit(jax.vmap(solve_single))
    else:
        return jax.jit(solve_single)

