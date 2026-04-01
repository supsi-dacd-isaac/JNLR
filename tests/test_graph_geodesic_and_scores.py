"""Tests for ``graph_geodesic`` and ``scores``."""

import os
import sys
import unittest
from unittest import mock

import numpy as np
import jax.numpy as jnp
from scipy.sparse import csr_matrix

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from jnlr.geodesics.graph_geodesic import (
    geo_graph,
    geo_graph_inner,
    get_adj_matrix,
    get_symmetric_knn_graph,
    graph_pointcloud_distance,
    map_to_sample_vertices,
    nearest_indices_kdtree,
    nearest_indices_matmul,
)
from jnlr.geodesics.scores import generalized_energy_score, geodesic_score_p


def _chain_adjacency_4(n=4):
    """Undirected path 0—1—…—(n-1) with unit edge weights."""
    row, col, data = [], [], []
    for i in range(n - 1):
        row.extend([i, i + 1])
        col.extend([i + 1, i])
        data.extend([1.0, 1.0])
    return csr_matrix((data, (row, col)), shape=(n, n))


class GraphGeodesicTests(unittest.TestCase):
    def test_get_adj_matrix_shape(self):
        pts = np.random.default_rng(0).normal(size=(40, 3))
        adj = get_adj_matrix(pts, k_neighbors=5)
        self.assertEqual(adj.shape, (40, 40))

    def test_geo_graph_inner_line_graph(self):
        # k-NN can add long chords; use an explicit chain graph for a unique shortest path.
        adj = _chain_adjacency_4(4)
        idx, dist = geo_graph_inner(adj, 0, 3)
        np.testing.assert_array_equal(idx, np.array([0, 1, 2, 3]))
        self.assertTrue(np.isfinite(dist[3]))
        self.assertAlmostEqual(float(dist[3]), 3.0, places=5)

    def test_geo_graph_inner_unreachable(self):
        pts_a = np.array([[0.0, 0.0], [1.0, 0.0]])
        pts_b = np.array([[100.0, 0.0], [101.0, 0.0]])
        pts = np.vstack([pts_a, pts_b])
        adj = get_adj_matrix(pts, k_neighbors=1)
        path, d = geo_graph_inner(adj, 0, 3)
        self.assertEqual(len(path), 0)
        self.assertTrue(np.isinf(d))

    def test_nearest_indices_kdtree_matches_matmul(self):
        rng = np.random.default_rng(1)
        points = rng.normal(size=(25, 2))
        queries = rng.normal(size=(7, 2))
        i1 = nearest_indices_kdtree(points, queries)
        i2 = nearest_indices_matmul(points, queries)
        np.testing.assert_array_equal(i1, i2)

    def test_geo_graph_one_dimensional_endpoints(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float64)
        adj = _chain_adjacency_4(4)
        paths, dist = geo_graph(pts, adj, np.array([0.0, 0.0]), np.array([3.0, 0.0]))
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].shape[0], 4)
        self.assertTrue(np.isfinite(float(dist)))

    def test_geo_graph_batch_two_dimensional(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
        adj = get_adj_matrix(pts, k_neighbors=2)
        z0 = np.array([[0.0, 0.0], [2.0, 0.0]])
        z1 = np.array([[2.0, 0.0], [0.0, 0.0]])
        paths, dists = geo_graph(pts, adj, z0, z1)
        self.assertEqual(len(paths), 2)
        self.assertEqual(dists.shape, (2,))
        self.assertTrue(np.all(np.isfinite(dists)))

    def test_geo_graph_invalid_z_shape_raises(self):
        pts = np.zeros((5, 2))
        adj = get_adj_matrix(pts, k_neighbors=2)
        with self.assertRaises(ValueError):
            geo_graph(pts, adj, np.zeros((2, 2)), np.zeros(2))

    def test_get_symmetric_knn_graph_symmetric(self):
        pts = np.random.default_rng(2).uniform(-1.0, 1.0, size=(30, 3))
        A = get_symmetric_knn_graph(pts, k_neighbors=4)
        diff = A - A.T
        self.assertLess(np.abs(diff.data).max() if diff.nnz else 0.0, 1e-10)

    def test_map_to_sample_vertices(self):
        samples = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
        queries = np.array([[0.05, 0.0], [1.9, 0.0]])
        idx = map_to_sample_vertices(samples, queries)
        np.testing.assert_array_equal(idx, np.array([0, 2]))

    def test_graph_pointcloud_distance_reduces(self):
        rng = np.random.default_rng(3)
        samples = rng.normal(size=(50, 3)) * 0.3
        T, S, n = 2, 4, 3
        z0 = rng.normal(size=(T, S, n)) * 0.2
        z1 = rng.normal(size=(T, n)) * 0.2
        for red in ("mean", "min", "max"):
            out = graph_pointcloud_distance(
                z0, z1, samples, reduce=red, k_neighbors=6, parallel=False
            )
            self.assertEqual(out.shape, (T,))
            self.assertTrue(np.all(np.isfinite(out)))
        raw = graph_pointcloud_distance(
            z0, z1, samples, reduce="none", k_neighbors=6, parallel=False
        )
        self.assertEqual(raw.shape, (T, S))

    def test_graph_pointcloud_distance_bad_reduce(self):
        z0 = np.zeros((1, 2, 3))
        z1 = np.zeros((1, 3))
        samples = np.zeros((10, 3))
        with self.assertRaises(ValueError):
            graph_pointcloud_distance(z0, z1, samples, reduce="median", parallel=False)

    @unittest.skipUnless(
        sys.platform.startswith("linux") or sys.platform == "darwin",
        "multiprocessing fork pool not available on all platforms",
    )
    def test_graph_pointcloud_distance_parallel_fork(self):
        rng = np.random.default_rng(4)
        samples = rng.normal(size=(36, 3)) * 0.25
        z0 = rng.normal(size=(3, 5, 3)) * 0.15
        z1 = rng.normal(size=(3, 3)) * 0.15
        try:
            out = graph_pointcloud_distance(
                z0,
                z1,
                samples,
                reduce="mean",
                k_neighbors=5,
                parallel=True,
                max_workers=2,
                chunk_sources=1,
            )
        except PermissionError:
            self.skipTest("ProcessPoolExecutor blocked in this environment")
        self.assertEqual(out.shape, (3,))
        self.assertTrue(np.all(np.isfinite(out)))


class ScoresTests(unittest.TestCase):
    def test_generalized_energy_score_shape_and_alpha(self):
        rng = np.random.default_rng(5)
        n, m, d = 6, 20, 2
        y_true = rng.normal(size=(n, d))
        y_samples = rng.normal(size=(n, m, d))
        scores = generalized_energy_score(y_true, y_samples, alpha=1.0)
        self.assertEqual(scores.shape, (n,))
        self.assertTrue(np.all(np.isfinite(scores)))

        scores2 = generalized_energy_score(y_true, y_samples, alpha=1.5)
        self.assertEqual(scores2.shape, (n,))

    def test_generalized_energy_score_alpha_invalid(self):
        with self.assertRaises(AssertionError):
            generalized_energy_score(
                np.zeros((2, 1)), np.zeros((2, 3, 1)), alpha=0.0
            )

    def test_geodesic_score_p_euclidean_stub(self):
        def geo_fun(a, b):
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            d = float(np.linalg.norm(a - b))
            return d, None

        y_true = np.array([[0.0, 0.0], [1.0, 1.0]])
        y_samples = np.array(
            [
                [[0.0, 0.1], [0.2, 0.0]],
                [[1.0, 1.1], [0.9, 1.0]],
            ],
            dtype=np.float64,
        )
        with mock.patch("jnlr.geodesics.scores.tqdm", lambda x, **kwargs: x):
            out = geodesic_score_p(geo_fun, jnp.asarray(y_true), jnp.asarray(y_samples))
        self.assertEqual(out.shape, (2,))
        self.assertTrue(np.all(out >= 0.0))
        self.assertTrue(np.all(np.isfinite(out)))


if __name__ == "__main__":
    unittest.main()
