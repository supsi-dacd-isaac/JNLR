import unittest
import jax
import jax.numpy as jnp
from jnlr.utils.function_utils import f_impl

from jnlr.should import (
    constant_sign_curvature,
    vector_valued_convex,
    p_reduction,
    p_reduction_and_intervals,
)
from jnlr.reconcile import make_solver
from jnlr.utils.manifolds import (
    f_paraboloid,
    f_vv_csc_44,lin_quad, f_vv_csc_24
)


class ShouldModuleTests(unittest.TestCase):
    def setUp(self):
        # Scalar manifold on R^2: f(x, y) = x^2 + y^2, constraint f(z) = 0 (origin)
        self.f_scalar = f_impl(f_paraboloid)
        self.grad_f = jax.grad(self.f_scalar)
        self.hess_f = jax.hessian(self.f_scalar)
        self.vmapped_solver_scalar = make_solver(self.f_scalar, jnp.eye(3), n_iterations=10)

        # Vector-valued manifold on R^2 with 2 constraints; pick a simple feasible one
        # lin_quad(z) = [x, x^2] -> feasible set is {x = 0}, a proper 1D manifold
        self.F_vv = f_impl(lin_quad)
        self.jac_F = jax.jacobian(self.F_vv, argnums=0)
        def _hess_F(z):
            comps = [jax.hessian(lambda u: self.F_vv(u)[i])(z) for i in range(2)]
            return jnp.stack(comps, axis=0)
        self.hess_F = _hess_F
        self.vmapped_solver_vv = make_solver(self.F_vv, jnp.eye(4), n_iterations=10)

    def test_implicit_f(self):
        """
        Test that implicit function construction works as expected.
        """

        z_hat = jnp.array([0.5, -0.5], dtype=jnp.float32)
        self.assertEqual(self.f_scalar(jnp.hstack([z_hat, f_paraboloid(z_hat)])),0)

        z_hat = jnp.array([0.5, -0.5], dtype=jnp.float32)

        self.assertEqual(self.F_vv(jnp.hstack([z_hat, lin_quad(z_hat)])).sum(),0)


    def test_vmapped_solver_projects_to_origin_for_paraboloid(self):
        n = 3  # dimension of z (2 inputs + 1 output)
        m = 1  # number of constraints

        z_hat = jnp.array([[0.3, -0.4, 0.7], [0.1, 0.2, 0.7]], dtype=jnp.float32)
        solver = make_solver(self.f_scalar, jnp.eye(n))
        z_tilde = solver(z_hat)

        # Should project close to the origin for this constraint
        self.assertAlmostEqual(jnp.abs(jax.vmap(self.f_scalar)(z_tilde)).sum(), 0.0, places=5)

        n = 4  # dimension of z (2 inputs + 2 output)
        m = 2  # number of constraints

        xs_hat = jnp.array([[0.0, 2.0], [  4.0, -1.0]], dtype=jnp.float32)
        z_hat = jnp.vstack([jnp.hstack([xs_hat[i], f_vv_csc_24(xs_hat[i]) + 0.1*jax.random.normal(jax.random.PRNGKey(0), (m,))]) for i in range(2)])  # shape (4,2)


        solver = make_solver(self.F_vv, jnp.eye(n), n_iterations=10)
        z_tilde = solver(z_hat)

        # Should project close to the origin for this constraint
        self.assertAlmostEqual(jnp.abs(jax.vmap(self.F_vv)(z_tilde)).sum(), 0.0, places=4)

    def test_p_reduction_paraboloid_returns_one(self):
        # Build samples far enough so pre-reconciliation max distance exceeds post distance
        z_hat = jnp.array([[0.3, -0.4, 0], [0.2, 0.1, 0]], dtype=jnp.float32)
        # Create bootstrap samples with displacements up to radius 1.0
        disps = jnp.array([
            [[1.0, 0.0, 0], [-1.0, 0.0, 0], [0.0, 1.0, 0], [0.0, -1.0, 0]],
            [[0.7, 0.7, 0], [-0.7, 0.7, 0], [0.7, -0.7, 0], [-0.7, -0.7, 0]],
        ], dtype=jnp.float32)
        z_hat_samples = z_hat[:, None, :] + disps

        probs = p_reduction(self.vmapped_solver_scalar, z_hat, z_hat_samples)
        self.assertEqual(probs.shape, (2,))
        self.assertTrue(jnp.all(probs == 1.0))  # All should show RMSE reduction

    def test_p_reduction_and_intervals_paraboloid(self):
        z_hat = jnp.array([[0.3, -0.4, 0], [0.2, 0.1, 0]], dtype=jnp.float32)
        disps = jnp.array([
            [[1.0, 0.0, 0], [-1.0, 0.0, 0], [0.0, 1.0, 0], [0.0, -1.0, 0]],
            [[0.7, 0.7, 0], [-0.7, 0.7, 0], [0.7, -0.7, 0], [-0.7, -0.7, 0]],
        ], dtype=jnp.float32)
        z_hat_samples = z_hat[:, None, :] + disps

        probs, intervals, delta_pi = p_reduction_and_intervals(
            self.vmapped_solver_scalar, z_hat, z_hat_samples, alpha=0.1
        )
        # Shapes
        self.assertEqual(probs.shape, (2,))
        self.assertEqual(intervals[0].shape, ((2,)))
        self.assertEqual(delta_pi.shape, z_hat.shape)

        self.assertTrue(jnp.all(probs <= intervals[1]))
        self.assertTrue(jnp.all(probs >= intervals[0]))


    def test_constant_sign_curvature_paraboloid_returns_false(self):
        # For paraboloid, projection is origin where grad==0 -> curvature check returns False
        z_hat = jnp.array([[1, 1, 0], [2, 2, 0], [0, 0, 1]], dtype=jnp.float32)
        accepted = constant_sign_curvature(
            self.f_scalar, self.grad_f, self.hess_f, self.vmapped_solver_scalar, z_hat
        )
        self.assertEqual(accepted.shape, (3,))
        self.assertTrue(accepted[0])
        self.assertTrue(accepted[1])
        self.assertTrue(~accepted[2])


    def test_vector_valued_convex_shapes(self):
        # Use batch of points; constraints likely inconsistent, but we just check shapes and finiteness
        z_hat = jnp.array([[0.3, -0.2, 0.3, -0.2], [1.0, 0.5, 0.3, -0.2]], dtype=jnp.float32)
        accepted, ratios = vector_valued_convex(
            self.F_vv, self.jac_F, self.hess_F, self.vmapped_solver_vv, z_hat
        )
        self.assertEqual(accepted.shape, (2,))
        self.assertEqual(ratios.shape, (2,))
        # accepted is boolean-like
        self.assertTrue(jnp.issubdtype(accepted.dtype, jnp.bool_))
        # ratios may be finite or nan depending on degeneracy; ensure not +/-inf
        self.assertTrue(jnp.all(~jnp.isinf(ratios)))


if __name__ == "__main__":
    unittest.main()
