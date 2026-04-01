"""Exercise every public function in ``jnlr.utils.manifolds``."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest

import jax
import jax.numpy as jnp

import jnlr.utils.manifolds as M


class ManifoldsAllFunctionsTests(unittest.TestCase):
    """Scalar maps R^2 -> R."""

    def setUp(self):
        self.v = jnp.array([0.15, -0.22], dtype=jnp.float32)

    def _assert_scalar_finite(self, y, msg=""):
        y = jnp.asarray(y)
        self.assertEqual(y.shape, (), msg=msg)
        self.assertTrue(jnp.isfinite(y), msg=msg)

    def test_all_scalar_surfaces(self):
        funcs = [
            M.f_paraboloid,
            M.f_mixed_quadratic,
            M.f_exponential,
            M.f_quartic,
            M.f_abs,
            M.f_himmelblau,
            M.f_rosenbrock,
            M.f_ackley,
            M.f_eggholder,
            M.f_rastrigin,
            M.f_shubert,
        ]
        for fn in funcs:
            with self.subTest(fn=fn.__name__):
                self._assert_scalar_finite(fn(self.v), fn.__name__)

    def test_all_vector_valued_surfaces(self):
        funcs = [
            M.lin_quad,
            M.f_vv_csc_22,
            M.f_vv_csc_24,
            M.f_vv_csc_e2_s2,
            M.f_test,
            M.f_vv_csc_44,
            M.f_vv_bowl_sin,
            M.f_vv_saddle_poly,
            M.f_vv_ring_trig,
            M.f_vv_exp_cosh,
        ]
        for fn in funcs:
            with self.subTest(fn=fn.__name__):
                y = fn(self.v)
                self.assertEqual(y.shape, (2,), fn.__name__)
                self.assertTrue(jnp.all(jnp.isfinite(y)), fn.__name__)

    def test_f_vv_rosenbrock_defaults_and_kwargs(self):
        y0 = M.f_vv_rosenbrock(self.v)
        self.assertEqual(y0.shape, (2,))
        self.assertTrue(jnp.all(jnp.isfinite(y0)))
        y1 = M.f_vv_rosenbrock(self.v, a=0.5, b=3.0)
        self.assertEqual(y1.shape, (2,))
        self.assertTrue(jnp.all(jnp.isfinite(y1)))

    def test_vmap_scalar_manifold(self):
        batch = jnp.array([[0.0, 0.0], [0.5, -0.3], [-0.2, 0.4]], dtype=jnp.float32)
        out = jax.vmap(M.f_paraboloid)(batch)
        self.assertEqual(out.shape, (3,))
        self.assertTrue(jnp.all(jnp.isfinite(out)))


if __name__ == "__main__":
    unittest.main()
