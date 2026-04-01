"""Tests for ``geodesics`` (compute, mmp, shooting, generate)."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from jnlr.geodesics.compute import GeodesicSolver, make_energy_pushforward
from jnlr.geodesics import compute as compute_mod
from jnlr.geodesics.mmp import MMPBatchEvaluator, geo_mmp
from jnlr.geodesics import shooting as shooting_mod
from jnlr.geodesics.shooting import (
    compute_christoffel,
    compute_metric,
    geodesic_rhs,
    integrate_geodesic,
    shooting_distance,
    shooting_loss,
)
from jnlr.utils.function_utils import f_impl
from jnlr.utils.manifolds import f_paraboloid, lin_quad
from jnlr.utils.meshes import get_mesh


def _phi_paraboloid_embed(u):
    u = jnp.asarray(u).ravel()
    x, y = u[0], u[1]
    return jnp.array([x, y, x**2 + y**2], dtype=u.dtype)


class MMPTests(unittest.TestCase):
    def setUp(self):
        self.vertices, self.triangles = get_mesh(
            f_paraboloid,
            kind="explicit",
            method="grid",
            grid_ranges=((-0.4, 0.4), (-0.4, 0.4)),
            nu=6,
            nv=6,
        )

    def test_geo_mmp_paths_parallel_off(self):
        z0 = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        z1 = np.array([0.2, 0.15, 0.2**2 + 0.15**2], dtype=np.float64)
        paths, dist = geo_mmp(
            z0,
            z1,
            self.vertices,
            self.triangles,
            need_paths=True,
            parallel=False,
        )
        self.assertTrue(np.isfinite(dist))
        self.assertIsNotNone(paths)

    def test_geo_mmp_distance_only(self):
        z0 = np.array([[0.05, -0.05, f_paraboloid(jnp.array([0.05, -0.05]))]], dtype=np.float64)
        z1 = np.array([-0.1, 0.1, f_paraboloid(jnp.array([-0.1, 0.1]))], dtype=np.float64)
        d = geo_mmp(
            z0,
            z1,
            self.vertices,
            self.triangles,
            need_paths=False,
            parallel=False,
        )
        self.assertTrue(np.isfinite(d))

    def test_geo_mmp_batched_reduce_mean(self):
        t, s = 2, 3
        rng = np.random.default_rng(0)
        starts = rng.normal(size=(t, s, 3)) * 0.05
        starts[..., 2] = starts[..., 0] ** 2 + starts[..., 1] ** 2
        ends = np.array(
            [
                [0.1, 0.0, 0.01],
                [-0.05, 0.05, 0.05**2 + 0.05**2],
            ],
            dtype=np.float64,
        )
        out = geo_mmp(
            starts,
            ends,
            self.vertices,
            self.triangles,
            need_paths=False,
            parallel=False,
            reduce="mean",
        )
        self.assertEqual(out.shape, (t,))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_mmp_batch_evaluator_invalid_mesh(self):
        with self.assertRaises(ValueError):
            MMPBatchEvaluator(np.zeros(3), np.array([[0, 1, 2]], dtype=np.int32))
        with self.assertRaises(ValueError):
            MMPBatchEvaluator(
                np.zeros((4, 3)), np.array([[0.0, 1.0, 2.0]], dtype=np.float64)
            )

    def test_geo_mmp_paths_need_reduce_conflict(self):
        with self.assertRaises(ValueError):
            geo_mmp(
                np.zeros((1, 3)),
                np.zeros(3),
                self.vertices,
                self.triangles,
                need_paths=True,
                reduce="mean",
            )


class ComputeModuleTests(unittest.TestCase):
    def test_make_energy_pushforward_codimension_error(self):
        ep = make_energy_pushforward(_phi_paraboloid_embed)
        with self.assertRaises(ValueError):
            ep(
                jnp.array([0.0, 0.0, 0.0]),
                jnp.array([1.0, 0.0, 0.0]),
                num_steps=8,
                codimension=5,
            )

    def test_geodesic_solver_unknown_method(self):
        with self.assertRaises(ValueError):
            GeodesicSolver(f_paraboloid, method="not_a_method", n_samples=10)

    def test_set_samples_validation(self):
        gs = GeodesicSolver(
            f_paraboloid, method="graph", n_samples=20, ranges=((-1.0, 1.0), (-1.0, 1.0))
        )
        with self.assertRaises(ValueError):
            gs.set_samples(np.zeros((5, 2)))

    def test_set_mesh_validation(self):
        gs = GeodesicSolver(f_paraboloid, method="mmp", n_samples=64, ranges=((-0.5, 0.5), (-0.5, 0.5)))
        # Intrinsic chart is 2D; vertex columns must match n_inputs[0], not ambient R^3.
        with self.assertRaises(ValueError):
            gs.set_mesh(np.zeros((4, 3)), np.array([[0, 1, 2]], dtype=np.int64))
        with self.assertRaises(ValueError):
            gs.set_mesh(np.zeros((4, 2)), np.array([[0, 1]], dtype=np.int64))

    def test_pointcloud_distance_mmp(self):
        gs = GeodesicSolver(
            f_paraboloid,
            method="mmp",
            n_samples=100,
            ranges=((-0.5, 0.5), (-0.5, 0.5)),
        )
        z0 = np.zeros((1, 3, 3), dtype=np.float64)
        z1 = np.array([[0.1, 0.0, 0.01]], dtype=np.float64)
        d = gs.pointcloud_distance(z0, z1)
        self.assertTrue(np.isfinite(np.asarray(d)).all())

    def test_pointcloud_distance_requires_mesh(self):
        gs = GeodesicSolver(f_paraboloid, method="mmp", mesh=None, n_samples=50)
        gs.mesh = None
        with self.assertRaises(ValueError):
            gs.pointcloud_distance(np.zeros((1, 2, 3)), np.zeros((1, 3)))

    def test_geodesic_shooting_method(self):
        gs = GeodesicSolver(
            f_paraboloid,
            method="shooting",
            n_samples=10,
            ranges=((-0.5, 0.5), (-0.5, 0.5)),
            n_steps_shooting=8,
        )
        tiny_path = jnp.zeros((8, 3), dtype=jnp.float32)
        with mock.patch.object(
            compute_mod,
            "shooting_distance",
            return_value=(tiny_path, jnp.array(0.1, dtype=jnp.float32)),
        ):
            z0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            z1 = np.array([0.05, 0.05, 0.005], dtype=np.float64)
            path, dist = gs.geodesic(z0, z1)
        self.assertTrue(jnp.isfinite(dist))
        self.assertGreater(path.shape[0], 0)


class ShootingTests(unittest.TestCase):
    def test_compute_metric_and_christoffel(self):
        u = jnp.array([0.1, -0.05], dtype=jnp.float32)
        g = compute_metric(_phi_paraboloid_embed, u)
        self.assertEqual(g.shape, (2, 2))
        self.assertTrue(jnp.all(jnp.isfinite(g)))
        Gamma = compute_christoffel(_phi_paraboloid_embed, u)
        self.assertEqual(Gamma.shape, (2, 2, 2))
        self.assertTrue(jnp.all(jnp.isfinite(Gamma)))

    def test_geodesic_rhs_shape(self):
        u = jnp.array([0.0, 0.0], dtype=jnp.float32)
        du = jnp.array([0.1, 0.2], dtype=jnp.float32)
        state = jnp.concatenate([u, du])
        out = geodesic_rhs(0.0, state, _phi_paraboloid_embed)
        self.assertEqual(out.shape, (4,))
        self.assertTrue(jnp.all(jnp.isfinite(out)))

    def test_integrate_geodesic_short(self):
        u0 = jnp.zeros(2, dtype=jnp.float32)
        v0 = jnp.array([0.05, 0.02], dtype=jnp.float32)
        ys = integrate_geodesic(_phi_paraboloid_embed, u0, v0, t_max=0.3, n_steps=12)
        self.assertEqual(ys.shape, (12, 4))
        self.assertTrue(jnp.all(jnp.isfinite(ys)))

    def test_shooting_loss_finite(self):
        u0 = jnp.zeros(2, dtype=jnp.float32)
        u1 = jnp.array([0.08, 0.06], dtype=jnp.float32)
        v0 = jnp.array([0.1, 0.1], dtype=jnp.float32)
        loss = shooting_loss(v0, _phi_paraboloid_embed, u0, u1, t_max=0.4, n_steps=10)
        self.assertTrue(jnp.isfinite(loss))

    def test_shooting_distance_smoke(self):
        u0 = jnp.array([0.0, 0.0], dtype=jnp.float32)
        u1 = jnp.array([0.08, 0.06], dtype=jnp.float32)
        with mock.patch.object(
            shooting_mod,
            "optimize_shooting_optax",
            return_value=jnp.array([0.12, 0.09], dtype=jnp.float32),
        ):
            path, dist = shooting_distance(
                _phi_paraboloid_embed, u0, u1, t_max=0.5, n_steps=12
            )
        self.assertTrue(jnp.isfinite(dist))
        self.assertEqual(path.shape[1], 3)
        self.assertTrue(jnp.all(jnp.isfinite(path)))


class GenerateModuleTests(unittest.TestCase):
    def test_trapezoid_helpers(self):
        from jnlr.geodesics import generate as gen

        x = jnp.linspace(0.0, 1.0, 11)
        y = x**2
        s = gen.trapezoid_integral(y, x)
        c = gen.trapezoid_cumulative(y, x)
        self.assertTrue(jnp.isfinite(s))
        self.assertEqual(c.shape, (11,))
        self.assertTrue(jnp.all(jnp.isfinite(c)))

    def test_project_velocity_to_tangent(self):
        from jnlr.geodesics import generate as gen

        F = f_impl(lin_quad)
        J = jax.jacfwd(F)
        x = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
        v = jnp.array([1.0, 1.0, 0.0, 0.0], dtype=jnp.float32)
        vt = gen.project_velocity_to_tangent(J, x, v)
        self.assertEqual(vt.shape, (4,))
        self.assertTrue(jnp.all(jnp.isfinite(vt)))

    def test_vHv_all_and_rhs_soa(self):
        from jnlr.geodesics import generate as gen

        F = f_impl(lin_quad)
        x = jnp.array([0.0, 0.1, 0.0, 0.01], dtype=jnp.float32)
        v = jnp.array([0.01, 0.02, 0.0, 0.0], dtype=jnp.float32)
        b = gen.vHv_all(F, x, v)
        self.assertEqual(b.shape, (2,))
        self.assertTrue(jnp.all(jnp.isfinite(b)))
        rhs_fn = gen.make_geodesic_rhs_soa(F)
        y = jnp.concatenate([x, v])
        dydt = rhs_fn(0.0, y, None)
        self.assertEqual(dydt.shape, (8,))
        self.assertTrue(jnp.all(jnp.isfinite(dydt)))

    def test_integrate_geodesic_implicit_generic_no_project(self):
        from jnlr.geodesics import generate as gen

        F = f_impl(lin_quad)
        x0 = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
        v0 = jnp.array([0.05, 0.03, 0.0, 0.0], dtype=jnp.float32)
        ts, X, V, Ltot, Lcum = gen.integrate_geodesic_implicit_generic(
            F, x0, v0, t1=0.25, n_steps=14, project_init=False, project_after=False
        )
        self.assertEqual(ts.shape, (14,))
        self.assertEqual(X.shape, (14, 4))
        self.assertTrue(jnp.isfinite(Ltot))
        self.assertEqual(Lcum.shape, (14,))


if __name__ == "__main__":
    unittest.main()
