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

LOOKAHEAD_CAP = 8  # how far to look ahead

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

def _try_eval_shape(f, D, dtype):
    try:
        out = jax.eval_shape(f, jax.ShapeDtypeStruct((D,), dtype))
        return True, out.shape
    except Exception:
        return False, None


def infer_min_input_size(f, *, dtype=jnp.float32, max_dim=21):
    # Finds the smallest D where f accepts a (D,) input and either:
    #   - f rejects (D+1,)  (exact-length signature), or
    #   - f accepts (D+1,) with a stable output shape/value, and
    #     f rejects (D-1,) or is not yet stable there.
    rd = jax.random.normal(jax.random.PRNGKey(0), (1,))

    def probe(D):
        return _try_call(f, jnp.arange(D, dtype=dtype) + rd)

    cached_next = (False, None, None)  # (ok, shape, y) for D reused from prior D+1

    for D in range(1, max_dim):
        if cached_next[0] and cached_next is not None:
            okD, shapeD = True, cached_next[1]
        else:
            okD, shapeD = _try_eval_shape(f, D, dtype)
        if not okD:
            cached_next = (False, None, None)
            continue

        okNext, shapeNext = _try_eval_shape(f, D + 1, dtype)

        if not okNext:
            for D2 in range(D + 2, min(max_dim, D + 2 + LOOKAHEAD_CAP)):
                if _try_eval_shape(f, D2, dtype)[0]:
                    return D2
            return D

        if len(shapeD) != len(shapeNext):
            return D
        if len(shapeD) > 1 and shapeD[1:] != shapeNext[1:]:
            return D

        # Value-equality stability check (requires actual execution)
        with jax.disable_jit():
            _, yD = probe(D)
            _, yNext = probe(D + 1)
        if _equal_outs(yD, yNext):
            if D == 1:
                return D
            with jax.disable_jit():
                okPrev, yPrev = probe(D - 1)
            if (not okPrev) or (not _equal_outs(yPrev, yD)):
                return D

        cached_next = (okNext, shapeNext, None)

    raise ValueError("Could not infer minimal input size up to max_dim.")


def infer_io_shapes(f, *, dtype=jnp.float32, d_input=None):
    if d_input is None:
        d_input = infer_min_input_size(f, dtype=dtype)
    return (d_input,), jnp.atleast_1d(f(jnp.arange(d_input, dtype=dtype))).shape
