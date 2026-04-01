import os

# Stable CI / local: avoid GPU init in geodesics stack
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest
import numpy as np
import jax.numpy as jnp

from jnlr.geodesics.compute import GeodesicSolver, make_energy_pushforward
from jnlr.utils.manifolds import f_paraboloid
from jnlr.utils.meshes import faces_from_2d_grid, get_mesh


class MeshesGeodesicsSmokeTests(unittest.TestCase):
    def test_faces_from_2d_grid(self):
        nx, ny = 3, 3
        F = faces_from_2d_grid(nx, ny)
        expected = 2 * (nx - 1) * (ny - 1)
        self.assertEqual(F.shape[0], expected)
        self.assertEqual(F.shape[1], 3)
        n_v = nx * ny
        self.assertTrue(np.all(F >= 0))
        self.assertTrue(np.all(F < n_v))

    def test_get_mesh_explicit_paraboloid(self):
        V, Ftri = get_mesh(
            f_paraboloid,
            kind="explicit",
            method="grid",
            grid_ranges=((-0.5, 0.5), (-0.5, 0.5)),
            nu=4,
            nv=4,
        )
        self.assertEqual(V.shape[1], 3)
        self.assertEqual(Ftri.shape[1], 3)
        self.assertGreater(V.shape[0], 0)

    def test_make_energy_pushforward(self):
        def phi(u):
            u = jnp.asarray(u).ravel()
            x, y = u[0], u[1]
            return jnp.array([x, y, x ** 2 + y ** 2])

        ep = make_energy_pushforward(phi)
        sp = jnp.array([0.0, 0.0])
        ep_pt = jnp.array([0.5, 0.25])
        energy, length, path_pts = ep(sp, ep_pt, num_steps=16, codimension=0)
        self.assertTrue(jnp.isfinite(energy))
        self.assertTrue(jnp.isfinite(length))
        self.assertEqual(path_pts.shape[1], 3)

    def test_geodesic_solver_graph(self):
        ranges = ((-1.0, 1.0), (-1.0, 1.0))
        gs = GeodesicSolver(
            f_paraboloid,
            method="graph",
            n_samples=64,
            ranges=ranges,
            k_neighbors=8,
        )
        z0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        z1 = np.array([0.4, 0.3, 0.4 ** 2 + 0.3 ** 2], dtype=np.float64)
        path, dist = gs.geodesic(z0, z1)
        self.assertTrue(np.isfinite(dist))
        self.assertIsNotNone(path)


if __name__ == "__main__":
    unittest.main()
