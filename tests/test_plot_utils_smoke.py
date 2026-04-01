import os
import unittest
from unittest import mock

import numpy as np
import plotly.graph_objects as go

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from jnlr.utils.manifolds import f_paraboloid
from jnlr.utils.plot_utils import plot_3d_projection, plot_mesh_plotly


class PlotUtilsSmokeTests(unittest.TestCase):
    def test_plot_mesh_plotly_returns_figure(self):
        # Single triangle in the plane z = 0
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
        )
        triangles = np.array([[0, 1, 2]], dtype=np.int64)
        fig = plot_mesh_plotly(
            vertices,
            triangles,
            show_edges=False,
            title="test",
            width=200,
            height=200,
        )
        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(fig.data), 0)

    def test_plot_mesh_plotly_edges_lines_points_constant_z(self):
        # Constant z exercises the intensity normalization branch (zero spread).
        vertices = np.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float64
        )
        triangles = np.array([[0, 1, 2]], dtype=np.int64)
        line = np.array([[0.0, 0.0, 1.0], [0.5, 0.5, 1.0]], dtype=np.float64)
        pts = np.array([[0.25, 0.25, 1.0]], dtype=np.float64)
        fig = plot_mesh_plotly(
            vertices,
            triangles,
            color="lightblue",
            show_edges=True,
            lines=line,
            points=pts,
            width=180,
            height=180,
        )
        self.assertIsInstance(fig, go.Figure)
        self.assertGreaterEqual(len(fig.data), 2)

    def test_plot_mesh_plotly_lines_list(self):
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
        )
        triangles = np.array([[0, 1, 2]], dtype=np.int64)
        fig = plot_mesh_plotly(
            vertices,
            triangles,
            show_edges=False,
            lines=[np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])],
            line_color="red",
            width=160,
            height=160,
        )
        self.assertIsInstance(fig, go.Figure)

    def test_plot_3d_projection_originals_only(self):
        X = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=np.float64)
        fig = plot_3d_projection(X, width=200, height=200)
        self.assertIsInstance(fig, go.Figure)
        names = [t.name for t in fig.data]
        self.assertIn("original", names)

    def _fake_solver_builder(self):
        def builder(f_implicit, W, n_iterations=10, return_history=False):
            del f_implicit, W, n_iterations

            def solver(Z):
                if return_history:
                    Z = np.asarray(Z)
                    return np.stack([Z, Z * 0.99 + 0.001], axis=1)
                return Z

            return solver

        return builder

    @mock.patch("jnlr.utils.plot_utils.make_solver_alm_optax")
    def test_plot_3d_projection_explicit_mocked_solver(self, mock_make):
        mock_make.side_effect = self._fake_solver_builder()
        X = np.array(
            [[0.1, 0.1, 0.02], [-0.1, 0.2, 0.05]],
            dtype=np.float64,
        )
        fig = plot_3d_projection(
            X,
            f_explicit=f_paraboloid,
            n_grid=10,
            n_iterations=2,
            width=220,
            height=220,
            remove_axes=True,
        )
        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(fig.data), 2)

    @mock.patch("jnlr.utils.plot_utils.make_solver_alm_optax")
    def test_plot_3d_projection_explicit_show_kde(self, mock_make):
        mock_make.side_effect = self._fake_solver_builder()
        rng = np.random.default_rng(0)
        X = np.column_stack(
            [
                rng.uniform(-0.2, 0.2, size=12),
                rng.uniform(-0.2, 0.2, size=12),
                rng.uniform(0.0, 0.1, size=12),
            ]
        )
        fig = plot_3d_projection(
            X,
            f_explicit=f_paraboloid,
            n_grid=14,
            n_iterations=2,
            show_kde=True,
            n_isolines=4,
            width=220,
            height=220,
        )
        self.assertIsInstance(fig, go.Figure)

    @mock.patch("jnlr.utils.plot_utils.make_solver_alm_optax")
    def test_plot_3d_projection_plot_history(self, mock_make):
        mock_make.side_effect = self._fake_solver_builder()
        X = np.array([[0.05, -0.05, 0.005]], dtype=np.float64)
        fig = plot_3d_projection(
            X,
            f_explicit=f_paraboloid,
            n_grid=8,
            n_iterations=2,
            plot_history=True,
            shrink_projection=True,
            width=200,
            height=200,
        )
        self.assertIsInstance(fig, go.Figure)


if __name__ == "__main__":
    unittest.main()
