import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest

import jax
import jax.numpy as jnp

from jnlr.reconcile import make_solver
from jnlr.utils.function_utils import f_impl
from jnlr.utils.manifolds import f_paraboloid, lin_quad
from jnlr.utils.samplers import (
    build_tangent_ops_chol,
    build_tangent_ops_qr,
    gumbel_top_k_indices,
    latin_hypercube,
    newton_project,
    roi_langevin_step,
    sample,
    sqrtdet_G,
    volume_expl,
)


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

    def test_sqrtdet_G_well_conditioned(self):
        J = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=jnp.float32)
        w = sqrtdet_G(J)
        self.assertTrue(jnp.isfinite(w))
        self.assertGreater(float(w), 0.0)

    def test_gumbel_top_k_indices(self):
        key = jax.random.PRNGKey(7)
        logits = jnp.log(jnp.array([1.0, 2.0, 0.5, 3.0, 0.1], dtype=jnp.float32))
        idx = gumbel_top_k_indices(key, logits, k=3)
        self.assertEqual(idx.shape, (3,))
        uidx = set(int(i) for i in idx.tolist())
        self.assertEqual(len(uidx), 3)

    def test_latin_hypercube_bounds_scaling(self):
        bounds = jnp.array([[-1.0, 1.0], [0.0, 2.0]], dtype=jnp.float32)
        pts = latin_hypercube(bounds, n_samples=16)
        self.assertEqual(pts.shape, (16, 2))
        self.assertTrue(jnp.all(pts[:, 0] >= -1.0) and jnp.all(pts[:, 0] <= 1.0))
        self.assertTrue(jnp.all(pts[:, 1] >= 0.0) and jnp.all(pts[:, 1] <= 2.0))

    def test_volume_expl_roi_R(self):
        bounds = jnp.array([[-0.3, 0.3], [-0.3, 0.3]], dtype=jnp.float32)
        pts = volume_expl(
            f_paraboloid,
            n_samples=20,
            bounds=bounds,
            oversample=4,
            min_pool=6,
            roi_R=1.5,
        )
        self.assertEqual(pts.shape[0], 20)
        self.assertEqual(pts.shape[1], 3)
        self.assertTrue(jnp.all(jnp.linalg.norm(pts, axis=1) <= 1.5 + 1e-5))

    def test_build_tangent_ops_project_small_residual(self):
        F = f_impl(lin_quad)
        JF = jax.jacfwd(F)
        u = jnp.array([0.1, 0.2], dtype=jnp.float32)
        x0 = jnp.concatenate([u, lin_quad(u)])
        J = JF(x0)

        pv_c, _ = build_tangent_ops_chol(x0, JF)
        pv_q, _ = build_tangent_ops_qr(x0, JF)
        v = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
        for pv in (pv_c, pv_q):
            w = pv(v)
            r = J @ w
            self.assertLess(float(jnp.linalg.norm(r)), 0.05)

    def test_newton_project_chol_and_qr(self):
        F = f_impl(lin_quad)
        JF = jax.jacfwd(F)
        x_off = jnp.array([0.1, 0.2, 0.0, 0.0], dtype=jnp.float32)
        for method in ("chol", "qr"):
            x_new = newton_project(x_off, F, JF, iters=12, method=method)
            self.assertTrue(jnp.all(jnp.isfinite(x_new)))
            self.assertLess(float(jnp.linalg.norm(F(x_new))), 1e-3)

    def test_roi_langevin_step_finite(self):
        F = f_impl(lin_quad)
        JF = jax.jacfwd(F)
        solver = make_solver(F, n_iterations=6, return_history=False, vmapped=False)
        u = jnp.array([0.05, -0.1], dtype=jnp.float32)
        x0 = jnp.concatenate([u, lin_quad(u)])
        key = jax.random.PRNGKey(11)
        y = roi_langevin_step(
            key,
            x0,
            JF=JF,
            solver=solver,
            sigma=1e-3,
            lam=0.0,
            kappa=0.0,
            R=10.0,
            proj_method="qr",
        )
        self.assertEqual(y.shape, (4,))
        self.assertTrue(jnp.all(jnp.isfinite(y)))


if __name__ == "__main__":
    unittest.main()
