"""Exercise primitives, helpers, and surfaces in ``implicit_hypersurfaces``."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest

import jax.numpy as jnp

import jnlr.utils.implicit_hypersurfaces as ih


class ImplicitHypersurfacesAllTests(unittest.TestCase):
    def setUp(self):
        self.p = jnp.array([0.1, -0.2, 0.3], dtype=jnp.float32)
        self.xyz = jnp.array([0.2, -0.15, 0.25], dtype=jnp.float32)

    def _assert_finite_scalar(self, x, name=""):
        x = jnp.asarray(x)
        self.assertEqual(x.shape, (), msg=name)
        self.assertTrue(jnp.isfinite(x), msg=name)

    def test_primitives(self):
        self._assert_finite_scalar(ih.sd_sphere(self.p, 1.0), "sd_sphere")
        self._assert_finite_scalar(
            ih.sd_ellipsoid(self.p, jnp.array([1.0, 0.5, 0.3], dtype=jnp.float32)),
            "sd_ellipsoid",
        )
        b = jnp.array([0.5, 0.5, 0.5], dtype=jnp.float32)
        self._assert_finite_scalar(ih.sd_box(self.p, b), "sd_box")
        self._assert_finite_scalar(ih.sdf_box(self.p, b), "sdf_box")
        self._assert_finite_scalar(ih.f_cube(self.p, size=1.0), "f_cube")
        a0 = jnp.zeros(3, dtype=jnp.float32)
        a1 = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32)
        self._assert_finite_scalar(ih.sd_capsule(self.p, a0, a1, 0.1), "sd_capsule")
        self._assert_finite_scalar(
            ih.sd_cone_frustum(
                self.p,
                jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
                r1=0.5,
                r2=0.3,
                h=1.0,
            ),
            "sd_cone_frustum",
        )
        n = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
        self._assert_finite_scalar(ih.sd_plane(self.p, n, 0.0), "sd_plane")
        center = jnp.zeros(3, dtype=jnp.float32)
        right = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
        up = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32)
        dim = jnp.array([0.4, 0.3, 0.2], dtype=jnp.float32)
        self._assert_finite_scalar(
            ih.sd_box_oriented(self.p, center, right, up, dim, r=0.02),
            "sd_box_oriented",
        )

    def test_helpers(self):
        R = ih._rotation_matrix(jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32), 0.3)
        self.assertEqual(R.shape, (3, 3))
        self.assertTrue(jnp.all(jnp.isfinite(R)))
        sm = ih._soft_min(1.0, 2.0, 0.1)
        sx = ih._soft_max(1.0, 2.0, 0.1)
        self.assertTrue(jnp.isfinite(sm))
        self.assertTrue(jnp.isfinite(sx))
        qr = ih._quaternion_rotate(
            self.p, jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32), 15.0
        )
        self.assertEqual(qr.shape, (3,))
        self.assertTrue(jnp.all(jnp.isfinite(qr)))
        op = ih._op_rep(self.p, jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32))
        self.assertEqual(op.shape, (3,))
        self.assertTrue(jnp.all(jnp.isfinite(op)))
        r2 = ih._rotate_2d(jnp.array([1.0, 0.0], dtype=jnp.float32), 0.5)
        self.assertEqual(r2.shape, (2,))
        self.assertTrue(jnp.all(jnp.isfinite(r2)))
        dirs = [
            jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32),
            jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32),
        ]
        mad = ih._max_abs_dot(self.p, dirs)
        self.assertTrue(jnp.isfinite(mad))

    def test_sdf_tardigrade(self):
        d = ih.sdf_tardigrade(self.p)
        self._assert_finite_scalar(d, "sdf_tardigrade")

    def test_implicit_surfaces_numeric(self):
        surfaces = [
            ih.surface1,
            ih.surface2,
            ih.surface3,
            ih.surface4,
            ih.surface5,
            ih.surface6,
            ih.surface7,
            ih.surface8,
            ih.surface_a,
            ih.surface_b,
            ih.surface_c,
        ]
        for fn in surfaces:
            with self.subTest(fn=fn.__name__):
                v = fn(self.xyz)
                v = jnp.asarray(v)
                self.assertEqual(v.shape, (), fn.__name__)
                self.assertTrue(jnp.isfinite(v), fn.__name__)

    def test_f_dodecahedron(self):
        d = ih.f_dodecahedron(self.p, scale=0.7, r=1.0)
        self._assert_finite_scalar(d, "f_dodecahedron")


if __name__ == "__main__":
    unittest.main()
