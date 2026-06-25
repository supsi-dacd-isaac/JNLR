import os
from pathlib import Path
import time
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go

from jnlr.learn import (
    KKTProjector,
    Projector,
    compute_V,
    kkt_fixed_point_step,
    kkt_residual,
    project_to_manifold,
)


def parabola_constraint(z):
    x, y = z
    return jnp.array([x**2 - y], dtype=z.dtype)


def nearest_point_on_parabola(point):
    """True Euclidean projection of ``point`` onto ``y = x**2``."""
    x0, y0 = float(point[0]), float(point[1])
    roots = np.roots([2.0, 0.0, 1.0 - 2.0 * y0, -x0])
    a = roots[np.isreal(roots)].real
    a = a[np.argmin((a - x0) ** 2 + (a**2 - y0) ** 2)]
    return np.array([a, a**2])


PLOT_DIR = Path(__file__).resolve().parent


class LearnMapTests(unittest.TestCase):
    def test_compute_V_scalar_constraint_shape(self):
        z = jnp.array([0.0, 1.0], dtype=jnp.float32)
        v = compute_V(lambda x: x[0] ** 2 - x[1], z)
        self.assertEqual(v.shape, z.shape)
        self.assertLess(float(v[1]), 0.0)

    def test_project_to_manifold_lands_on_constraint(self):
        z = jnp.array([1.5, 0.0], dtype=jnp.float32)
        projected = project_to_manifold(parabola_constraint, z, n_steps=8, eta=1.0)
        self.assertLess(float(jnp.abs(parabola_constraint(projected)[0])), 1e-5)

    def test_kkt_fixed_point_step_reduces_kkt_residual(self):
        z0 = jnp.array([1.5, 0.0], dtype=jnp.float32)
        y0 = jnp.concatenate([z0, jnp.zeros((1,), dtype=z0.dtype)])

        y1 = kkt_fixed_point_step(
            parabola_constraint,
            z0,
            y0,
            damping=1.0,
            lambda_reg=1e-6,
        )

        before = float(jnp.linalg.norm(kkt_residual(parabola_constraint, z0, y0)))
        after = float(jnp.linalg.norm(kkt_residual(parabola_constraint, z0, y1)))
        self.assertLess(after, before)

    def test_kkt_fixed_point_step_supports_sqp_matrix(self):
        z0 = jnp.array([1.5, 0.0], dtype=jnp.float32)
        y0 = jnp.concatenate([z0, jnp.zeros((1,), dtype=z0.dtype)])

        sqp_step = kkt_fixed_point_step(
            parabola_constraint,
            z0,
            y0,
            damping=1.0,
            lambda_reg=1e-6,
            matrix="sqp",
        )
        before = float(jnp.linalg.norm(kkt_residual(parabola_constraint, z0, y0)))
        after = float(jnp.linalg.norm(kkt_residual(parabola_constraint, z0, sqp_step)))
        self.assertLess(after, before)

        y_with_multiplier = jnp.array([1.2, 0.4, 0.3], dtype=jnp.float32)
        exact_step = kkt_fixed_point_step(
            parabola_constraint,
            z0,
            y_with_multiplier,
            damping=0.5,
            lambda_reg=1e-6,
            matrix="exact",
        )
        sqp_step = kkt_fixed_point_step(
            parabola_constraint,
            z0,
            y_with_multiplier,
            damping=0.5,
            lambda_reg=1e-6,
            matrix="sqp",
        )
        self.assertFalse(np.allclose(np.asarray(exact_step), np.asarray(sqp_step)))

    def test_kkt_steps_fix_true_projection_point(self):
        z0 = jnp.array([1.5, 0.0], dtype=jnp.float32)
        z_star = jnp.asarray(nearest_point_on_parabola(np.asarray(z0)), dtype=z0.dtype)
        # For g(x, y) = x^2 - y, stationarity gives lambda = y_star - y0.
        lambda_star = jnp.array([z_star[1] - z0[1]], dtype=z0.dtype)
        y_star = jnp.concatenate([z_star, lambda_star])

        self.assertLess(
            float(jnp.linalg.norm(kkt_residual(parabola_constraint, z0, y_star))),
            1e-6,
        )
        for matrix in ("exact", "sqp"):
            stepped = kkt_fixed_point_step(
                parabola_constraint,
                z0,
                y_star,
                damping=1.0,
                lambda_reg=0.0,
                matrix=matrix,
            )
            self.assertTrue(np.allclose(np.asarray(stepped), np.asarray(y_star)))

    def test_amortised_map_learns_standalone_projection(self):
        xs = jnp.linspace(-1.5, 1.5, 31)
        ys = jnp.linspace(-0.5, 2.5, 31)
        points = jnp.stack(jnp.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
        n_steps = 600

        projector = Projector(parabola_constraint, key=0)
        losses = projector.train(
            points,
            n_steps=n_steps,
            batch_size=256,
            learning_rate=2e-3,
            n_proj_steps=8,
            proj_eta=1.0,
            consistency_weight=1.0,
        )
        self.assertEqual(losses.shape, (n_steps,))

        truth = np.stack([nearest_point_on_parabola(p) for p in np.asarray(points)])

        # Single forward pass: lands close to the manifold and near the nearest
        # point, but only approximately (it is an amortised one-shot map).
        standalone = np.asarray(projector.predict(points))
        standalone_res = float(
            jnp.mean(jnp.abs(jax.vmap(parabola_constraint)(standalone)))
        )
        standalone_err = float(np.mean(np.linalg.norm(standalone - truth, axis=1)))
        self.assertLess(standalone_res, 0.2)
        self.assertLess(standalone_err, 0.2)

        # Optional refinement: warm-started numerical projection is exact on the
        # manifold.
        refined = np.asarray(projector.predict(points, project=True))
        refined_res = float(jnp.mean(jnp.abs(jax.vmap(parabola_constraint)(refined))))
        self.assertLess(refined_res, 1e-4)

    def test_kkt_map_learns_comparable_projection(self):
        xs = jnp.linspace(-1.2, 1.2, 80)
        ys = jnp.linspace(-0.2, 1.8, 80)
        points = jnp.stack(jnp.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
        truth = np.stack([nearest_point_on_parabola(p) for p in np.asarray(points)])

        baseline = Projector(parabola_constraint, key=1)
        start = time.perf_counter()
        baseline_losses = baseline.train(
            points,
            n_steps=300,
            batch_size=128,
            learning_rate=2e-3,
            n_proj_steps=8,
            proj_eta=1.0,
            consistency_weight=1.0,
        )
        baseline_losses.block_until_ready()
        baseline_time = time.perf_counter() - start

        kkt_projector = KKTProjector(parabola_constraint, key=1)
        start = time.perf_counter()
        losses = kkt_projector.train(
            points,
            n_steps=6000,
            batch_size=128,
            learning_rate=1e-3,
            damping=0.8,
            lambda_reg=1e-5,
            multiplier_weight=0.2,
            matrix="exact",
        )
        losses.block_until_ready()
        exact_time = time.perf_counter() - start


        field_xs = jnp.linspace(-1.2, 1.2, 12)
        field_ys = jnp.linspace(-0.2, 1.8, 12)
        field_points = jnp.stack(
            jnp.meshgrid(field_xs, field_ys, indexing="xy"), axis=-1
        ).reshape(-1, 2)
        curve = jnp.stack([xs, xs**2], axis=1)

        baseline_fig = baseline.plot_vector_field(
            field_points,
            curve=curve,
            show=False,
            title="Projection Head",
        )
        baseline_path = PLOT_DIR / "learn_map_projection_head.html"
        baseline_fig.write_html(baseline_path, include_plotlyjs="cdn")

        exact_fig = kkt_projector.plot_vector_field(
            field_points,
            curve=curve,
            show=False,
            title="KKT Exact Newton",
        )
        exact_path = PLOT_DIR / "learn_map_kkt_exact_newton.html"
        exact_fig.write_html(exact_path, include_plotlyjs="cdn")

        sqp_projector = KKTProjector(parabola_constraint, key=2)
        start = time.perf_counter()
        sqp_losses = sqp_projector.train(
            points,
            n_steps=6000,
            batch_size=128,
            learning_rate=1e-3,
            damping=0.8,
            lambda_reg=1e-5,
            multiplier_weight=0.2,
            matrix="sqp",
        )
        sqp_losses.block_until_ready()
        sqp_time = time.perf_counter() - start
        print(
            "training times: "
            f"projection_head={baseline_time:.3f}s, "
            f"kkt_exact_newton={exact_time:.3f}s, "
            f"kkt_sqp_gauss_newton={sqp_time:.3f}s",
            flush=True,
        )

        sqp_fig = sqp_projector.plot_vector_field(
            field_points,
            curve=curve,
            show=False,
            title="KKT SQP Gauss-Newton",
        )
        sqp_path = PLOT_DIR / "learn_map_kkt_sqp_gauss_newton.html"
        sqp_fig.write_html(sqp_path, include_plotlyjs="cdn")

        baseline_pred = np.asarray(baseline.predict(points))
        kkt_pred, multipliers = kkt_projector.predict_primal_dual(points)
        kkt_pred = np.asarray(kkt_pred)
        sqp_pred = np.asarray(sqp_projector.predict(points))

        baseline_err = float(np.mean(np.linalg.norm(baseline_pred - truth, axis=1)))
        kkt_err = float(np.mean(np.linalg.norm(kkt_pred - truth, axis=1)))
        kkt_res = float(jnp.mean(jnp.abs(jax.vmap(parabola_constraint)(kkt_pred))))
        sqp_err = float(np.mean(np.linalg.norm(sqp_pred - truth, axis=1)))
        sqp_res = float(jnp.mean(jnp.abs(jax.vmap(parabola_constraint)(sqp_pred))))

        self.assertEqual(multipliers.shape, (points.shape[0], 1))
        self.assertTrue(baseline_path.exists())
        self.assertTrue(exact_path.exists())
        self.assertTrue(sqp_path.exists())
        self.assertEqual(baseline_fig.layout.title.text, "Projection Head")
        self.assertEqual(exact_fig.layout.title.text, "KKT Exact Newton")
        self.assertEqual(sqp_fig.layout.title.text, "KKT SQP Gauss-Newton")
        self.assertLess(kkt_res, 0.15)
        self.assertLess(kkt_err, 0.2)
        self.assertLess(kkt_err, baseline_err + 0.2)
        self.assertLess(sqp_res, 0.15)
        self.assertLess(sqp_err, 0.2)
        self.assertLess(sqp_err, baseline_err + 0.2)

    def test_plot_vector_field_returns_expected_traces(self):
        xs = jnp.linspace(-1.5, 1.5, 21)
        ys = jnp.linspace(-0.5, 2.5, 21)
        points = jnp.stack(jnp.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)

        projector = Projector(parabola_constraint, key=0)
        projector.train(points, n_steps=50)

        field_xs = jnp.linspace(-1.2, 1.2, 7)
        field_ys = jnp.linspace(-0.2, 1.8, 7)
        field_points = jnp.stack(
            jnp.meshgrid(field_xs, field_ys, indexing="xy"), axis=-1
        ).reshape(-1, 2)
        curve = jnp.stack([xs, xs**2], axis=1)

        fig = projector.plot_vector_field(field_points, curve=curve, show=False)
        self.assertIsInstance(fig, go.Figure)
        names = {trace.name for trace in fig.data}
        self.assertIn("constraint", names)
        self.assertIn("vectors", names)
        self.assertIn("source", names)
        self.assertIn("mapped", names)
        self.assertGreater(len(fig.layout.annotations), 0)
        self.assertEqual({ann.arrowsize for ann in fig.layout.annotations}, {0.3})


if __name__ == "__main__":
    unittest.main()
