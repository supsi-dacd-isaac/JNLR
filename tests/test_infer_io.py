import unittest
from jnlr.utils import manifolds as M
import jax.numpy as jnp
from jnlr.utils.function_utils import infer_io_shapes


def f_paraboloid(v):
    x, y = v
    return x**2 + y**2

def f_implicit_paraboloid(v):
    z = v[2]
    return f_paraboloid(v[:2]) - z

def sdf_box(p: jnp.ndarray, b: jnp.ndarray) -> float:
    """Signed distance to an axis–aligned box with half‑sizes ``b``.

    ``p`` is the query point in 3D space.  ``b`` should be a vector of
    positive half–extents along ``x``, ``y`` and ``z``.  Based on
    ``sdBox`` from ``Cube.glsl``.  For a cube of side length 1 the
    appropriate ``b`` is ``(0.5, 0.5, 0.5)``.
    """
    d = jnp.abs(p) - b
    inside = jnp.minimum(jnp.maximum(d[0], jnp.maximum(d[1], d[2])), 0.0)
    outside = jnp.linalg.norm(jnp.maximum(d, 0.0))
    return jnp.array(inside + outside)

def sdf_cube(p: jnp.ndarray, size: float = 1.0) -> float:
    """Convenience wrapper around :func:`sdf_box` for cubes.

    ``size`` is the full length of each side.  The GLSL version uses a
    cube of size 1, corresponding to ``b = (0.5, 0.5, 0.5)``.
    """
    half = 0.5 * size
    b = jnp.array([half, half, half])
    return sdf_box(p, b)


class InferIOShapesTests(unittest.TestCase):
    def test_infer_io_shapes_on_10_manifold_functions(self):
        # (function, expected output dim)
        cases = [
            (M.f_paraboloid, 1),
            (M.f_abs, 1),
            (M.f_quartic, 1),
            (M.f_himmelblau, 1),
            (M.f_rosenbrock, 1),
            (M.f_rastrigin, 1),
            (M.lin_quad, 2),
            (M.f_vv_csc_24, 2),
            (M.f_vv_bowl_sin, 2),
            (M.f_vv_rosenbrock, 2),
        ]

        for f, m_expected in cases:
            in_shape, out_shape = infer_io_shapes(f)
            self.assertEqual(in_shape, (2,))
            self.assertEqual(out_shape, (m_expected,))

    def test_infer_io_shapes_with_scalar_input(self):
        def f_scalar_input(x):
            return jnp.array([x[0]**2 + 1, x[0]**3 - 2])
        in_shape, out_shape = infer_io_shapes(f_scalar_input)
        self.assertEqual(in_shape, (1,))
        self.assertEqual(out_shape, (2,))

    def test_sdf_cube_and_box(self):
        in_shape, out_shape = infer_io_shapes(sdf_cube)
        self.assertEqual(in_shape, (3,))
        self.assertEqual(out_shape, (1, ))

    def test_implicit_paraboloid(self):
        in_shape, out_shape = infer_io_shapes(f_implicit_paraboloid)
        self.assertEqual(in_shape, (3,))
        self.assertEqual(out_shape, (1, ))

    def test_complex_hypersrfaces(self):
        import jnlr.utils.complex_hypersurfaces as chy
        for f in [chy.sdf_tardigrade, chy.surface1, chy.surface2, chy.surface3, chy.surface4]:
            in_shape, out_shape = infer_io_shapes(f)
            self.assertEqual(in_shape, (3,))
            self.assertEqual(out_shape, (1, ))

if __name__ == "__main__":
    unittest.main()

