import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import NearestNeighbors
import pygeodesic.geodesic as geodesic
from jnlr.utils.function_utils import infer_io_shapes
from jnlr.utils.log_utils import configure_logging
from jnlr.utils.meshes import get_mesh
from jnlr.utils.samplers import sample
from jnlr.geodesics.shooting import shooting_distance
from jnlr.geodesics.mmp import geo_mmp
from jnlr.geodesics.graph_geodesic import geo_graph, graph_pointcloud_distance, get_adj_matrix

logger = configure_logging(__name__)

# --- 1. Straight lines in input space (part of Connecting Neural Models Latent
#  Geometries with Relative Geodesic Representations Hanlin Yu, 2025 preprint) ---
def make_energy_pushforward(phi):
    J_phi = jax.jacobian(phi)

    @partial(jax.jit, static_argnames=("num_steps", "codimension"))
    def energy_pushforward(start_point, end_point, *, num_steps: int = 100, codimension: int = 1):
        """
        Pullback straight-line energy/length between u0,u1 where phi: R^d->R^n.

        Inputs are assumed 'ambient' if codimension>0 and we take the first d = n - codim
        entries as intrinsic coords. If codimension==0, inputs must already be intrinsic.
        """
        sp = jnp.asarray(start_point).ravel()
        ep = jnp.asarray(end_point).ravel()

        # Infer intrinsic dimension and slice consistently (no lax.cond needed).
        d = sp.shape[0] - codimension
        if d <= 0:
            raise ValueError("codimension too large for provided points.")
        u0 = sp[:d]   # (d,)
        u1 = ep[:d]   # (d,)

        delta = u1 - u0
        du_dt = delta

        # Sample the straight segment in intrinsic coords
        t = jnp.linspace(0.0, 1.0, num_steps)
        u_t = u0[None, :] + t[:, None] * delta[None, :]   # (T, d)

        # q(u) = (du/dt)^T G(u) (du/dt), with G = J^T J
        def q_of_u(u):
            J = J_phi(u)         # (n, d)
            G = J.T @ J          # (d, d)
            return du_dt @ (G @ du_dt)   # scalar

        quad_vals = jax.vmap(q_of_u)(u_t)   # (T,)
        path_pts  = jax.vmap(phi)(u_t)      # (T, n)

        # Uniform-grid trapezoid on [0,1]
        dt = 1.0 / (num_steps - 1)
        def trapz_uniform(y):
            # works for T>=2; if T==1, returns 0 (zero interval)
            return dt * (0.5 * (y[0] + y[-1]) + jnp.sum(y[1:-1])) if y.size > 1 else 0.0

        energy = 0.5 * trapz_uniform(quad_vals)
        length = trapz_uniform(jnp.sqrt(jnp.maximum(quad_vals, 0.0)))
        return energy, length, path_pts

    return energy_pushforward




# --- 4. Geodesic Jaxable wrappers ---

def g_mmp():
    pass

def g_graph():
    pass

def g_shooting():
    pass


class GeodesicSolver:
    def __init__(self, phi, method='mmp', n_samples=1000, ranges=None, k_neighbors=10, n_steps_shooting=100, mesh=None, samples=None):
        self.phi = phi
        self.n_inputs, self.n_outputs = infer_io_shapes(phi)
        self.mesh = mesh
        self.samples = samples
        self.adj_matrix = None
        self.n_samples = n_samples if samples is None else samples.shape[0]
        default_ranges = [(0.0, 1.0)] * self.n_inputs[0]  # default grid ranges for mesh generation
        self.ranges = default_ranges if ranges is None else ranges
        self.k_neighbors = k_neighbors
        self.n_steps_shooting = n_steps_shooting
        self.method = method
        self.initialize()

    def initialize(self):
        if self.method == 'mmp':
            if self.mesh is None:
                logger.info("Mesh not provided. I'm building and storing a default mesh on the range {}.".format(self.ranges))
                n_samples_dim = int(np.ceil(np.sqrt(self.n_samples)))
                if self.n_inputs[0] == 2:
                    mesh_kind = 'explicit'
                elif self.n_inputs[0] == 3 and self.n_outputs[0] == 1:
                    mesh_kind = 'implicit'
                else:
                    logger.critical('Cannot build mesh: input dim {}, output dim {}. We can only build meshes for input '
                                    'dim 2 or 3 and output dim 1'.format(self.n_inputs[0], self.n_outputs[0]))
                v, t = get_mesh(self.phi, kind=mesh_kind, method='grid', grid_ranges=self.ranges, nu=n_samples_dim, nv=n_samples_dim)
                self.mesh = {'vertices': v, 'triangles': t}
        elif self.method == 'graph':
            if self.samples is None:
                logger.info("Samples not provided. I'm generating and storing default samples in the 0-1 hypercube.")
                self.samples = sample(self.phi, method='random', n_samples=self.n_samples)
            self.adj_matrix = get_adj_matrix(self.samples, k_neighbors=self.k_neighbors)
        elif self.method == 'shooting':
            pass
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def set_samples(self, samples):

        if self.samples.ndim != 2 or self.samples.shape[1] != self.n_inputs[0] + self.n_outputs[0]:
            raise ValueError(f"Samples must be of shape (N, {self.n_inputs[0] + self.n_outputs[0]:})")
        self.samples = np.asarray(samples)
        self.n_samples = self.samples.shape[0]

    def set_mesh(self, vertices, triangles):
        if vertices.ndim != 2 or vertices.shape[1] != self.n_inputs[0]:
            raise ValueError(f"Vertices must be of shape (N, {self.n_inputs[0]:})")
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError("Triangles must be of shape (M, 3)")
        self.mesh = {'vertices': np.asarray(vertices), 'triangles': np.asarray(triangles)}

    def geodesic(self, z0, z1):
        method = self.method
        z0 = np.asarray(z0)
        z1 = np.asarray(z1)
        if method == "mmp":
            path, distance = geo_mmp(np.atleast_2d(z0)[:, None, :], np.atleast_2d(z1), **self.mesh, parallel=True, need_paths=True)
        elif method == "graph":
            path, distance = geo_graph(self.samples, self.adj_matrix, z0, z1)
        elif method == 'shooting':
            phi = lambda xy: jnp.hstack([xy, self.phi(xy)])
            d = self.n_inputs[0]
            u0 = np.asarray(z0).ravel()[:d]
            u1 = np.asarray(z1).ravel()[:d]
            path, distance = shooting_distance(phi, u0, u1, n_steps=self.n_steps_shooting)
        else:
            raise ValueError(f"Unknown method: {method}")
        return path, distance

    def pointcloud_distance(self, z0, z1):
        """
        Compute the Euclidean distance between point clouds in z0 and ground truth points in z1 on the manifold.
        Args:
            z0 : point clouds, dimension (t, s, n) where t is the number of point clouds, s is the number of points in each cloud, and n is the ambient dimension.
            z1 : ground truth points, dimension (t, n) where t is the number of point clouds and n is the ambient dimension.

        Returns:

        """
        method = self.method
        if np.ndim(z0) != 3:
            z0 = jnp.expand_dims(z0, axis=0)  # make it (1, s, n)
        if np.ndim(z1) != 2:
            z1 = jnp.expand_dims(z1, axis=0)  # make it (1, n)
        if z0.shape[0] != z1.shape[0]:
            raise ValueError("z0 and z1 must have the same number of point clouds (first dimension)")
        if method == 'mmp':
            if self.mesh is None:
                raise ValueError("Mesh must be set for mmp method. Use set_mesh() or provide a mesh at initialization.")
            dists = geo_mmp(z0, z1, **self.mesh, parallel=True, max_workers=6, reduce='mean')
        elif method == 'graph':
            if self.samples is None:
                raise ValueError("Samples must be set for graph method. Use set_samples() or provide samples at initialization.")
            dists = graph_pointcloud_distance(z0, z1, self.samples, parallel=True, max_workers=6, reduce='mean')
        return dists