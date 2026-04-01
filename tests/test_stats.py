import unittest

import jax.numpy as jnp

from jnlr.stats import beta_ppf_approx, clopper_pearson_intervals


class StatsTests(unittest.TestCase):
    def test_beta_ppf_approx_scalar(self):
        q = jnp.array(0.5)
        x = beta_ppf_approx(q, jnp.array(2.0), jnp.array(3.0))
        self.assertTrue(jnp.isfinite(x))
        self.assertGreater(float(x), 0.0)
        self.assertLess(float(x), 1.0)

    def test_beta_ppf_approx_vector(self):
        q = jnp.array([0.1, 0.5, 0.9])
        x = beta_ppf_approx(q, jnp.array(2.0), jnp.array(3.0))
        self.assertEqual(x.shape, (3,))
        self.assertTrue(jnp.all(jnp.isfinite(x)))

    def test_clopper_pearson_k_zero(self):
        lower, upper = clopper_pearson_intervals(0, 10, alpha=0.05)
        self.assertEqual(float(lower), 0.0)
        self.assertTrue(jnp.isfinite(upper))
        self.assertLessEqual(float(lower), float(upper))

    def test_clopper_pearson_k_equals_n(self):
        lower, upper = clopper_pearson_intervals(10, 10, alpha=0.05)
        self.assertEqual(float(upper), 1.0)
        self.assertTrue(jnp.isfinite(lower))
        self.assertLessEqual(float(lower), float(upper))

    def test_clopper_pearson_interior(self):
        lower, upper = clopper_pearson_intervals(3, 10, alpha=0.05)
        self.assertTrue(jnp.isfinite(lower))
        self.assertTrue(jnp.isfinite(upper))
        self.assertLessEqual(float(lower), float(upper))
        self.assertGreaterEqual(float(lower), 0.0)
        self.assertLessEqual(float(upper), 1.0)


if __name__ == "__main__":
    unittest.main()
