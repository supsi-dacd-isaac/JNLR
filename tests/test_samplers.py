import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest

import jax.numpy as jnp

from jnlr.utils.manifolds import f_paraboloid
from jnlr.utils.samplers import sample


class SamplersTests(unittest.TestCase):
    def test_sample_volume_small(self):
        bounds = jnp.array([[-0.4, 0.4], [-0.4, 0.4]])
        pts = sample(
            f_paraboloid,
            method="volume",
            n_samples=24,
            bounds=bounds,
            oversample=4,
            min_pool=4,
        )
        self.assertEqual(pts.ndim, 2)
        self.assertEqual(pts.shape[1], 3)
        self.assertGreater(pts.shape[0], 0)
        self.assertTrue(jnp.all(jnp.isfinite(pts)))

    def test_sample_random_small(self):
        bounds = jnp.array([[-0.5, 0.5], [-0.5, 0.5]])
        pts = sample(
            f_paraboloid,
            method="random",
            n_samples=20,
            bounds=bounds,
        )
        self.assertEqual(pts.shape[1], 3)
        self.assertGreaterEqual(pts.shape[0], 1)
        self.assertTrue(jnp.all(jnp.isfinite(pts)))

    def test_sample_unknown_method_raises(self):
        with self.assertRaises(ValueError) as ctx:
            sample(f_paraboloid, method="not_a_method", n_samples=5)
        self.assertIn("Unknown sampling method", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
