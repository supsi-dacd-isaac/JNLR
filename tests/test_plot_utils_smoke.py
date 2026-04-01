import unittest

import numpy as np
import plotly.graph_objects as go

from jnlr.utils.plot_utils import plot_mesh_plotly


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


if __name__ == "__main__":
    unittest.main()
