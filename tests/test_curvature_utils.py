import unittest

import jax
import jax.numpy as jnp

from jnlr.utils.curvature_utils import (
    curvature_along_projection,
    min_eigenvalue,
    min_tangent_eigenvalue,
    min_tangent_eigenvalue_vv,
    solve_lagrange_multipliers,
    tangent_space_basis,
    tangent_space_basis_vv,
)
from jnlr.utils.function_utils import f_impl
from jnlr.utils.manifolds import f_paraboloid, lin_quad


class CurvatureUtilsTests(unittest.TestCase):
    def test_tangent_space_basis_orthonormal_orthogonal_to_nu(self):
        key = jax.random.PRNGKey(0)
        nu = jax.random.normal(key, (5,))
        nu = nu / jnp.linalg.norm(nu)
        Q = tangent_space_basis(nu)
        self.assertEqual(Q.shape, (5, 4))
        ortho = Q.T @ Q
        self.assertTrue(jnp.allclose(ortho, jnp.eye(4), atol=1e-5, rtol=1e-5))
        self.assertTrue(jnp.allclose(Q.T @ nu, jnp.zeros(4), atol=1e-5))

    def test_tangent_space_basis_vv(self):
        J = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        Q = tangent_space_basis_vv(J)
        self.assertEqual(Q.shape, (4, 2))
        self.assertTrue(jnp.allclose(Q.T @ Q, jnp.eye(2), atol=1e-5))

    def test_solve_lagrange_multipliers_residual(self):
        J = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        delta_pi = jnp.array([0.5, -0.25])
        lam = solve_lagrange_multipliers(J, delta_pi, reg=1e-8)
        rhs = -2.0 * (J @ delta_pi)
        residual = J @ J.T @ lam - rhs
        self.assertTrue(jnp.linalg.norm(residual) < 1e-4)

    def test_min_tangent_eigenvalue_paraboloid_finite(self):
        f = f_impl(f_paraboloid)
        grad_f = jax.grad(f)
        hess_f = jax.hessian(f)
        z = jnp.array([0.3, 0.4, 0.25], dtype=jnp.float32)
        lam = min_tangent_eigenvalue(grad_f, hess_f, z)
        self.assertTrue(jnp.isfinite(lam))

    def test_min_tangent_eigenvalue_zero_grad_branch(self):
        f = lambda z: jnp.array(0.0)
        grad_f = jax.grad(f)
        hess_f = jax.hessian(f)
        z = jnp.zeros(3, dtype=jnp.float32)
        lam = min_tangent_eigenvalue(grad_f, hess_f, z)
        self.assertTrue(jnp.isnan(lam))

    def test_min_eigenvalue(self):
        def quad(z):
            return jnp.sum(z ** 2)

        hess_f = jax.hessian(quad)
        z = jnp.array([1.0, 2.0, 3.0])
        lam_min = min_eigenvalue(hess_f, z)
        self.assertAlmostEqual(float(lam_min), 2.0, places=5)

    def test_min_tangent_eigenvalue_vv_lin_quad(self):
        F = f_impl(lin_quad)
        jac_F = jax.jacobian(F, argnums=0)

        def hess_F(z):
            comps = [jax.hessian(lambda u: F(u)[i])(z) for i in range(2)]
            return jnp.stack(comps, axis=0)

        z_hat = jnp.array([0.5, -0.3, 0.1, 0.2], dtype=jnp.float32)
        z_tilde = jnp.array([0.5, -0.3, 0.5, (-0.3) ** 2], dtype=jnp.float32)
        out = min_tangent_eigenvalue_vv(F, jac_F, hess_F, z_tilde, z_hat)
        self.assertEqual(out.shape, (2,))
        self.assertTrue(jnp.all(jnp.isfinite(out)))

    def test_curvature_along_projection(self):
        def quad(z):
            return jnp.sum(z ** 2)

        hess_f = jax.hessian(quad)
        z = jnp.ones(3)
        d = jnp.array([1.0, 0.0, 0.0])
        c = curvature_along_projection(hess_f, z, d)
        self.assertTrue(jnp.isfinite(c))


if __name__ == "__main__":
    unittest.main()
