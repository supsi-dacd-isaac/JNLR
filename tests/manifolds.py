import jax.numpy as jnp

def f_paraboloid(v):
    x, y = v
    return x**2 + y**2

def f_mixed_quadratic(v):
    x, y = v
    return x**2 + x*y + y**2

def f_exponential(v):
    x, y = v
    return jnp.exp(x) + jnp.exp(y)

def f_quartic(v):
    x, y = v
    return x**4 + y**4 + x**2 + y**2

def f_abs(b):
    return jnp.abs(b[0]) + jnp.abs(b[1])


def f_himmelblau(v):
    x, y = v
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def f_rosenbrock(v):
    x, y = v
    return (1 - x)**2 + 100 * (y - x**2)**2

def f_ackley(x):
    x1, x2 = x
    a = 20
    b = 0.2
    c = 2 * jnp.pi

    sum_sq = 0.5 * (x1**2 + x2**2)
    cos_comp = 0.5 * (jnp.cos(c * x1) + jnp.cos(c * x2))

    return -a * jnp.exp(-b * jnp.sqrt(sum_sq)) - jnp.exp(cos_comp) + a + jnp.e

def f_eggholder(x):
    x1, x2 = x
    offset = 0.2
    frequency = 5.0

    term1 = -(x2 + offset) * jnp.sin(frequency * jnp.sqrt(x2 + x1 / 2 + offset))
    term2 = -x1 * jnp.sin(frequency * jnp.sqrt(jnp.abs(x1 - (x2 + offset))))

    return term1 + term2

def f_rastrigin(v):
    x1, x2 = v
    A = 10.0
    return A * 2 + (x1**2 - A * jnp.cos(2 * jnp.pi * x1)) + (x2**2 - A * jnp.cos(2 * jnp.pi * x2))

def f_shubert(v):
    """
    Shubert function. Input: array of shape (2,)
    Use vmap externally for batching.
    """
    x1, x2 = v
    total1 = 0.0
    total2 = 0.0
    for j in range(1, 6):
        total1 += j * jnp.cos((j + 1) * x1 + j)
        total2 += j * jnp.cos((j + 1) * x2 + j)
    return total1 * total2


# --------------------------------------------------------------------------------------------------------------------
# -------------------------- Vector valued functions -----------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

def lin_quad(z):
    """
    Linear + quadratic function.
    Input: array of shape (2,)
    """
    x, y = z
    g1 = x
    g2 = x**2
    return jnp.array([g1, g2])

def f_vv_csc_22(z):
    z1, z2 = z
    g1 = z1**2 + z2**2
    g2 = 2*(z1**2 + z2**2) + 10
    return jnp.array([g1, g2])


def f_vv_csc_24(z):
    z1, z2 = z
    g1 = z1**2 + z2**2
    g2 = 2*(z1**2 + z2**4) + 10
    return jnp.array([g1, g2])

def f_vv_csc_e2_s2(z):
    z1, z2 = z
    g1 = jnp.exp((z1/10)**2 + (z2/10)**2)
    g2 = jnp.sin(z1**2 + z2**2) + (z1**2 + z2**2)**2
    return jnp.array([g1, g2])


def f_test(z):
    z1, z2 = z
    g1 = z1**2 + z2**2
    g2 = 2*(z1 + z2) - 10
    return jnp.array([g1, g2])

# ----------------------------------------------------------
# 1.  Globally convex + convex  (radial quartic + offset)
#     M is the intersection of two nested convex bowls
# ----------------------------------------------------------
def f_vv_csc_44(z):
    x, y = z
    g1 = (x**2 + y**2)**2                    # r^4
    g2 = 0.5 * (x**2 + y**2)**2 + 5.0        # scaled & lifted
    return jnp.array([g1, g2])

# ----------------------------------------------------------
# 2.  One convex, one *non*-convex (sin) component
#     First bowl is convex; second undulates.
# ----------------------------------------------------------
def f_vv_bowl_sin(z):
    x, y = z
    r2 = x**2 + y**2
    g1 = r2                                  # convex paraboloid
    g2 = jnp.sin(1.5 * r2) + 0.3 * r2        # oscillatory
    return jnp.array([g1, g2])

# ----------------------------------------------------------
# 3.  Two non-convex polynomial components (monkey saddle family)
#     Saddle behaviour around origin.
# ----------------------------------------------------------
def f_vv_saddle_poly(z):
    x, y = z
    g1 = x**3 - 3*x*y**2                     # monkey saddle
    g2 = y**3 - 3*y*x**2                     # rotated version
    return jnp.array([g1, g2])

# ----------------------------------------------------------
# 4.  Banana / Rosenbrock-style pair
#     Narrow curved valley; Hessian indef. away from valley.
# ----------------------------------------------------------
def f_vv_rosenbrock(z, a=1.0, b=5.0):
    x, y = z
    g1 = a - x                               # linear
    g2 = y - x**2                            # valley surface
    return jnp.array([g1, b * g2])

# ----------------------------------------------------------
# 5.  Quartic "ring" + trigonometric height
#     Non-convex but bounded level-set ring.
# ----------------------------------------------------------
def f_vv_ring_trig(z):
    x, y = z
    r2 = x**2 + y**2
    g1 = (r2 - 1.0)**2                       # quartic ring
    g2 = jnp.cos(2.0 * jnp.pi * r2)          # ripples
    return jnp.array([g1, g2])

# ----------------------------------------------------------
# 6.  Mixed exponential–cosh pair (steep + gentle)
#     First is strictly convex; second has saddle lines.
# ----------------------------------------------------------
def f_vv_exp_cosh(z):
    x, y = z
    g1 = jnp.exp((x/10)**2 + (y/10)**2)                # steep convex
    g2 = jnp.cosh(x) - jnp.cosh(y)           # neither convex nor concave
    return jnp.array([g1, g2])