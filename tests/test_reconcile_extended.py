import unittest

import jax
import jax.numpy as jnp

from jnlr.reconcile import make_solver, make_solver_alm_optax
from jnlr.utils.function_utils import f_impl
from jnlr.utils.manifolds import f_paraboloid


class ReconcileExtendedTests(unittest.TestCase):
    def setUp(self):
        self.f = f_impl(f_paraboloid)

    def test_make_solver_return_history_shape(self):
        solver = make_solver(self.f, jnp.eye(3), n_iterations=5, return_history=True)
        z_hat = jnp.array([[0.3, -0.4, 0.5]], dtype=jnp.float32)
        hist = solver(z_hat)
        self.assertEqual(hist.shape, (1, 5, 3))

    def test_make_solver_vmapped_false_single_point(self):
        solver = make_solver(self.f, jnp.eye(3), n_iterations=10, vmapped=False)
        z_hat = jnp.array([0.3, -0.4, 0.5], dtype=jnp.float32)
        z_tilde = solver(z_hat)
        self.assertEqual(z_tilde.shape, (3,))
        self.assertAlmostEqual(float(jnp.abs(self.f(z_tilde))), 0.0, places=4)

    def test_make_solver_custom_w_diagonal(self):
        W = jnp.diag(jnp.array([2.0, 2.0, 1.0], dtype=jnp.float32))
        solver = make_solver(self.f, W, n_iterations=15)
        z_hat = jnp.array([[0.2, 0.2, 0.2]], dtype=jnp.float32)
        z_tilde = solver(z_hat)
        self.assertAlmostEqual(float(jnp.abs(self.f(z_tilde[0]))), 0.0, places=4)

    def test_make_solver_alm_fixed_lr_branch(self):
        solver = make_solver_alm_optax(
            self.f,
            jnp.eye(3),
            n_iterations=8,
            lbfgs_learning_rate=1.0,
        )
        z_hat = jnp.array([[0.3, -0.2, 0.4]], dtype=jnp.float32)
        z_tilde = solver(z_hat)
        self.assertAlmostEqual(float(jnp.abs(self.f(z_tilde[0]))), 0.0, places=4)

    def test_make_solver_alm_return_history_shape(self):
        solver = make_solver_alm_optax(
            self.f,
            jnp.eye(3),
            n_iterations=4,
            return_history=True,
        )
        z_hat = jnp.array([[0.1, 0.2, 0.3]], dtype=jnp.float32)
        hist = solver(z_hat)
        self.assertEqual(hist.shape, (1, 4, 3))

    def test_make_solver_alm_vmapped_false(self):
        solver = make_solver_alm_optax(
            self.f,
            jnp.eye(3),
            n_iterations=10,
            vmapped=False,
        )
        z_hat = jnp.array([0.25, -0.25, 0.2], dtype=jnp.float32)
        z_tilde = solver(z_hat)
        self.assertEqual(z_tilde.shape, (3,))
        self.assertAlmostEqual(float(jnp.abs(self.f(z_tilde))), 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
