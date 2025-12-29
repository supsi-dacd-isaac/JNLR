import numpy as np
from scipy.spatial import Delaunay
from jnlr.utils.function_utils import infer_io_shapes
from typing import Callable, Tuple
import jax.numpy as jnp
from jax import vmap

def faces_from_2d_grid(nx: int, ny: int) -> np.ndarray:
    """
    Return (2*(nx-1)*(ny-1), 3) triangle indices for an nx x ny grid
    flattened in C-order (i major, then j).
    """
    i = np.arange(nx - 1)[:, None]          # (nx-1, 1)
    j = np.arange(ny - 1)[None, :]          # (1, ny-1)

    v0 = i * ny + j
    v1 = (i + 1) * ny + j
    v2 = (i + 1) * ny + (j + 1)
    v3 = i * ny + (j + 1)

    tri1 = np.stack([v0, v1, v2], axis=-1).reshape(-1, 3)
    tri2 = np.stack([v0, v2, v3], axis=-1).reshape(-1, 3)
    return np.vstack([tri1, tri2]).astype(np.int64)

def explicit_3d(
    phi,
    u_range: tuple[float, float],
    v_range: tuple[float, float],
    nu: int,
    nv: int,
    *,
    expect_vectorized: bool = True,
):
    """
    Build a plottable triangular mesh from a param map phi: R^2 -> R^3.

    Parameters
    ----------
    phi : callable
        If expect_vectorized=True: takes (M,2) array of (u,v) and returns (M,3).
        If False: takes (2,) array and returns (3,) — called pointwise.
    u_range, v_range : (min, max)
        Parameter box limits.
    nu, nv : int
        Number of samples along u and v (>= 2).
    expect_vectorized : bool
        Whether phi can process batches.

    Returns
    -------
    V : (nu*nv, 3) float
        Vertex positions in R^3.
    F : (2*(nu-1)*(nv-1), 3) int
        Triangle indices.
    (U, UV) : optional diagnostic — parameter grid (nu*nv, 2), reshaped (nu, nv, 2).
    """
    if nu < 2 or nv < 2:
        raise ValueError("nu and nv must be >= 2")

    u = np.linspace(*u_range, nu)
    v = np.linspace(*v_range, nv)

    # Parameter grid
    UU, VV = np.meshgrid(u, v, indexing="ij")            # (nu, nv)
    UV = np.stack([UU, VV], axis=-1)                     # (nu, nv, 2)
    U = UV.reshape(-1, 2)                                # (nu*nv, 2)

    # Map to R^3

    V = vmap(phi)(U)

    if V.ndim != 2 or V.shape[0] != U.shape[0] or V.shape[1] != 3:
        raise ValueError("phi must return an array of shape (nu*nv, 3)")

    # Faces
    F = faces_from_2d_grid(nu, nv)

    return V, F, (U, UV)


def get_mesh(f, kind='explicit', method='delaunay', grid_ranges:Tuple=None, nu=10, nv=10):
    """
    Generate a mesh representation of the manifold defined by f.

    Args:
        f: Function defining the manifold. Can be explicit (f: R^n -> R^m) or implicit (f: R^d -> R).
        kind: 'explicit' or 'implicit'.
        method: Method to generate the mesh ('delaunay' or 'grid').
        grid_ranges: Tuple of tuples defining the parameter ranges for the grid method.
        grid_limits: Tuple defining the min and max limits for each axis in the grid method.

    Returns:
        vertices: Array of shape (N, d) representing points on the manifold.
        faces: Array of shape (M, 3) representing triangular faces (only for 3D manifolds).
    """
    n, m = infer_io_shapes(f)

    if grid_ranges is None:
        if n[0] != 2:
            raise ValueError("grid_ranges must be provided for non-2D input functions.")
        grid_ranges = ((-1, 1), (-1, 1))
    if kind == 'explicit':
        if m[0] != 1 and m[0] != 3:
            raise ValueError("Explicit function must map to R or R3 for 3D mesh generation.")
        if m[0] == 1:
            # Convert to implicit function
            f_expl = lambda x: jnp.hstack([x, f(x)])
        else:
            f_expl = f
        V, F, _ = explicit_3d(f_expl, grid_ranges[0], grid_ranges[1], nu=nu, nv=nv)

    return V, F