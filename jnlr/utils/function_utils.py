import jax.numpy as jnp
import jax


def f_impl(f_expl, n: int = None, m: int = None):
    r"""
    Build an implicit function g(z) from explicit f_expl:
    - z = [x, y] where x in R^n and y in R^m
    - Returns:
      - scalar \(\) when m == 1
      - vector (m, ) when m > 1
    Shape is fixed at factory time to preserve JAX tracing.
    """
    if n is None or m is None:
        n_in, out_shape = infer_io_shapes(f_expl)
        n = n_in[0]
        m = out_shape[0]

    if m == 1:
        # jax.gradient is unhappy with shape (1,) outputs, it requires scalar ()
        def wrapped_scalar(z):
            x = z[:n]
            y0 = z[n]  # scalar ()
            u0 = jnp.squeeze(f_expl(x))  # ensure scalar ()
            return u0 - y0
        return wrapped_scalar
    else:
        def wrapped_vect(z):
            x = z[:n]
            y = z[n:n + m]  # (m,)
            u = jnp.asarray(f_expl(x))
            u = jnp.reshape(u, (m,))  # enforce static (m,)
            return u - y
        return wrapped_vect

def _try_call(f, x):
    try:
        return True, f(x)
    except Exception:
        return False, None

def _equal_outs(y1, y2):
    # Works for array or scalar outputs; extend for pytrees if needed.
    if not (hasattr(y1, "shape") and hasattr(y2, "shape")):
        return False
    if y1.shape != y2.shape:
        return False
    return jnp.array_equal(jnp.asarray(y1), jnp.asarray(y2))

def infer_min_input_size(f, *, dtype=jnp.float32, max_dim=1024):
    # Finds the smallest `D` where:
    # 1) f(arange(D)) succeeds, and either
    # 2a) f(arange(D+1)) fails (exact-length functions), or
    # 2b) f(arange(D+1)) succeeds and output is stable, and
    #     f(arange(D-1)) either fails or is not yet stable.
    with jax.disable_jit():
        for D in range(1, max_dim):
            okD, yD = _try_call(f, jnp.arange(D, dtype=dtype))
            if not okD:
                continue

            # If D+1 fails, accept D (exact-length signature).
            okNext, yNext = _try_call(f, jnp.arange(D + 1, dtype=dtype))
            if not okNext:
                return D

            # If both D and D+1 succeed, require stabilization at D,
            # and non-stabilization at D-1 (or D-1 failing) for minimality.
            if _equal_outs(yD, yNext):
                if D == 1:
                    return D
                okPrev, yPrev = _try_call(f, jnp.arange(D - 1, dtype=dtype))
                if (not okPrev) or (not _equal_outs(yPrev, yD)):
                    return D

    raise ValueError("Could not infer minimal input size up to max_dim.")

def infer_io_shapes(f, *, dtype=jnp.float32, d_input=None):
    if d_input is None:
        d_input = infer_min_input_size(f, dtype=dtype)
    out_aval = jax.eval_shape(f, jax.ShapeDtypeStruct((d_input,), dtype))
    return (d_input,), jnp.atleast_1d(f(jnp.arange(d_input, dtype=dtype))).shape