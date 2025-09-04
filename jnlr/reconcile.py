import jax.numpy as jnp
import jax
from jax import lax
import optax

def make_projector_alm_optax(
    f, W,
    *,
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
):
    """
    Returns: proj(zhat_batch) -> z_proj_batch
    Projects onto {z : f(z)=0} in metric W using ALM + Optax L-BFGS (zoom line search).
    """

    # --- Whitening W = L^T L ---
    W = jnp.asarray(W); W = 0.5 * (W + W.T)
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
        """
        Minimize L(y; lam, rho) = 0.5||y - yhat||^2 + lam^T c + 0.5*rho*||c||^2
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







    return jax.jit(jax.vmap(solve_single))


def make_solver_proj_nt_curv(
    f, W,
    *,
    n_iterations: int = 30,        # outer iters
    n_tangent_micro: int = 3,      # relinearized tangent micro-steps / iter
    alpha_n: float = 1.0,          # normal step fraction
    alpha_t: float = 0.7,          # initial tangent step fraction
    beta: float = 0.5,             # shrink for tangent backtracking
    max_bt: int = 6,               # max shrink attempts per micro-step
    damping: float = 1e-6,         # tiny Tikhonov on JJ^T
    return_history: bool = False,
):
    """
    Constrained projection: min 0.5||z-zhat||^2 s.t. f(z)=0
    Iteration:
      1) Normal correction at current z:
           s_n = -J^T (JJ^T + eps I)^(-1) f(z);  z <- z + alpha_n s_n
      2) Curvature-aware tangent slide (n_tangent_micro times):
           relinearize at each sub-step:
             Pt = I - J^T (JJ^T + eps I)^(-1) J
             s_t = - Pt (z - zhat)
             accept z + a s_t only if ||f|| decreases; else a <- beta*a
    This keeps moves small, exploits curvature, and has very few knobs.
    """
    _ = jnp.asarray(W)  # API compat
    jac_f = jax.jacfwd(f)

    def f_norm(z):       # scalar feasibility
        r = jnp.atleast_1d(f(z))
        return jnp.sqrt(jnp.sum(r*r))

    def normal_step(z):
        J   = jnp.atleast_2d(jac_f(z))
        m   = J.shape[0]
        JJt = J @ J.T + damping * jnp.eye(m, dtype=J.dtype)
        r   = jnp.atleast_1d(f(z))
        lam = jnp.linalg.solve(JJt, r)
        s_n = - J.T @ lam
        return z + alpha_n * s_n

    def one_tangent_micro(z, zhat, a):
        # build tangent projector at *current* z (relinearize each micro step)
        J   = jnp.atleast_2d(jac_f(z))
        m   = J.shape[0]
        JJt = J @ J.T + damping * jnp.eye(m, dtype=J.dtype)
        Pt  = jnp.eye(J.shape[1], dtype=J.dtype) - J.T @ jnp.linalg.solve(JJt, J)
        s_t = - Pt @ (z - zhat)         # restricted descent for 0.5||z-zhat||^2

        phi0 = f_norm(z)

        def bt_body(i, best):
            z_best, accepted = best
            a_i = beta**i * a
            z_cand = z + a_i * s_t
            phi_c  = f_norm(z_cand)
            ok = phi_c <= phi0   # accept only if feasibility doesn't worsen
            z_new = jnp.where(accepted, z_best, jnp.where(ok, z_cand, z_best))
            acc_new = jnp.logical_or(accepted, ok)
            return (z_new, acc_new)

        z_bt, acc = lax.fori_loop(0, max_bt, bt_body, (z, jnp.array(False)))
        z_fb = z  # if nothing improved, skip tangent step this time
        return jnp.where(acc, z_bt, z_fb)

    def one_iteration(z, zhat):
        # 1) normal correction
        z = normal_step(z)
        # 2) a few small tangent steps with relinearization + decrease guard on ||f||
        def micro_body(_, state):
            z_curr = state
            return one_tangent_micro(z_curr, zhat, alpha_t)
        z = lax.fori_loop(0, n_tangent_micro, micro_body, z)
        return z

    def scan_step(carry, _):
        z, zhat = carry
        z_next = one_iteration(z, zhat)
        return (z_next, zhat), z_next

    def solve_single(zhat: jnp.ndarray):
        z0 = zhat
        (zT, _), hist = lax.scan(scan_step, (z0, zhat), xs=None, length=n_iterations)
        return hist if return_history else zT

    return jax.jit(jax.vmap(solve_single))


def make_solver_vanilla(
    f,
    W: jnp.ndarray,                      # kept for API compatibility
    n_iterations: int = 50,
    damping: float = 1e-4,               # a bit larger for stability
    alpha: float = 0.5,                  # relaxation (0<alpha<=1)
    use_backtracking: bool = False,
    beta: float = 0.5,                   # backtracking shrink
    c_armijo: float = 1e-4,              # sufficient decrease
    max_bt: int = 10,
    return_history: bool = False,
):
    jac_f = jax.jacfwd(f)
    _ = jnp.asarray(W)  # unused, but keeps signature stable

    def merit(z, zhat):
        r = jnp.atleast_1d(f(z)) + jnp.sum((zhat - z)**2)
        # Simple feasibility merit; you can add + rho * ||z-zhat||^2 if helpful
        return 0.5 * (r @ r)

    def anchored_trial(z, zhat):
        J = jnp.atleast_2d(jac_f(z))                # (m,n)
        c = jnp.atleast_1d(f(z) + J @ (zhat - z))   # (m,)
        m = J.shape[0]
        JJt = J @ J.T + damping * jnp.eye(m, dtype=J.dtype)
        lam = jnp.linalg.solve(JJt, c)              # (m,)
        z_trial = zhat - J.T @ lam                  # anchored solution
        return z_trial

    def step(z, zhat):
        z_trial = anchored_trial(z, zhat)

        # Option A: fixed relaxation (cheap & reliable)
        if not use_backtracking:
            return z + alpha * (z_trial - z)

        # Option B: Armijo backtracking on feasibility (JAX-safe)
        phi0 = merit(z, zhat)

        def bt_body(i, state):
            z_best, accepted = state
            a = beta ** i
            z_cand = z + a * (z_trial - z)
            phi_c = merit(z_cand, zhat)
            # accept if phi decreases enough
            cond = (phi_c <= phi0 - c_armijo * a * phi0)
            z_new  = jnp.where(accepted, z_best,
                               jnp.where(cond, z_cand, z_best))
            acc_new = jnp.logical_or(accepted, cond)
            return (z_new, acc_new)

        z_bt, acc = lax.fori_loop(0, max_bt, bt_body, (z, jnp.array(False)))
        # Fallback: if never accepted, take a small relaxed step
        z_fallback = z + 0.1 * (z_trial - z)
        return jnp.where(acc, z_bt, z_fallback)

    if return_history:
        def solve_single(zhat: jnp.ndarray):
            def body(z, _):
                z_next = step(z, zhat)
                return z_next, z_next
            z0 = zhat
            _, zs = lax.scan(body, z0, xs=None, length=n_iterations)
            return zs  # (T, n)
    else:
        def solve_single(zhat: jnp.ndarray):
            def body(_, z):
                return step(z, zhat)
            z0 = zhat
            z_final = lax.fori_loop(0, n_iterations, body, z0)
            return z_final  # (n,)

    return jax.jit(jax.vmap(solve_single))




def make_solver(f, W: jnp.ndarray,
                n_iterations: int = 50,
                damping: float = 1e-5,          # a bit larger for f32; reduce if using x64
                beta: float = 0.5,              # backtracking factor
                c_armijo: float = 1e-4,         # sufficient decrease on ||f||
                max_bt: int = 12,
                return_history: bool = False):
    r"""
    Create a v-mapped and JIT-compiled solver function for the constrained optimization problem.
    Here f is the implicit function representing the manifold constraints: $M = \{ z : f(z) = 0 \}$.
    The returned function takes z_hat as input and returns the projected z.
    $$
    \text{arg}\min_{z} \tfrac{1}{2} (z - \hat{z})^T W (z - \hat{z}) \\
    $$
    $$
    \text{s.t. } f(z) = 0
    $$

    Args:
        f: Function representing the constraints, in implicit form. The signature of f should be $f(z): \mathbb{R}^n \rightarrow \mathbb{R}^m$ where $n$ is the dimension of the input and m the output.
        W: Weight matrix.
        n_iterations:  Number of iterations for the learning process.

    Returns:
        A JIT-compiled function that takes z_hat as input and returns the projected z.
    """

    W = jnp.asarray(W)
    jac_f = jax.jacfwd(f)
    from jnlr.utils.function_utils import infer_io_shapes
    n_input, n_constraints = infer_io_shapes(f)

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

    return jax.jit(jax.vmap(solve_single))


