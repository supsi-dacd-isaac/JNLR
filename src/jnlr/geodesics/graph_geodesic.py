import os
import numpy as np
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context


def get_adj_matrix(points, k_neighbors=10):
    # 1. Use NearestNeighbors to build the k-NN graph efficiently
    neighbors_model = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto', metric='euclidean')
    neighbors_model.fit(points)

    # 2. Get the adjacency matrix in a sparse format
    adj_matrix = neighbors_model.kneighbors_graph(mode='distance')
    return adj_matrix

def geo_graph_inner(adj_matrix, start_idx, end_idx):
    # 2. Dijkstra's algorithm to find the shortest path in the graph
    dist_matrix, predecessors = dijkstra(
        csgraph=adj_matrix,
        indices=start_idx,
        return_predecessors=True,
    )

    # 4. Reconstruct the path with an error check
    if dist_matrix[end_idx] == np.inf:
        print("Error: The end point is unreachable from the start point in the k-NN graph.")
        return [], np.inf

    path_indices = [end_idx]
    current_node = end_idx
    while current_node != start_idx and predecessors[current_node] != -9999:
        current_node = predecessors[current_node]
        path_indices.append(current_node)
    path_indices.reverse()
    return np.array(path_indices), dist_matrix


def nearest_indices_kdtree(points, queries):
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean')
    nn.fit(points)
    idx = nn.kneighbors(queries, return_distance=False)
    return idx.ravel()

def nearest_indices_matmul(points, queries):
    pts_sq = np.sum(points**2, axis=1)        # (P,)
    qry_sq = np.sum(queries**2, axis=1)       # (Q,)
    # d2[i,j] = ||queries[i]-points[j]||^2
    d2 = qry_sq[:, None] + pts_sq[None, :] - 2 * queries @ points.T
    # Numerical guard (optional)
    # d2 = np.maximum(d2, 0.0)
    return np.argmin(d2, axis=1)

def geo_graph(points, adj_matrix, z0, z1):

    if np.ndim(z0) == 2 and np.ndim(z1) == 2:
        start_idx = nearest_indices_matmul(points, z0)
        end_idx = nearest_indices_matmul(points, z1)
    elif np.ndim(z0) == 1 and np.ndim(z1) == 1:
        start_idx = np.argmin(np.linalg.norm(points - z0, axis=1))
        end_idx = np.argmin(np.linalg.norm(points - z1, axis=1))
    else:
        raise ValueError("z0 and z1 must both be either 1D or 2D arrays.")

    if np.ndim(z0) == 2 and np.ndim(z1) == 2:
        paths = []
        dists = []
        for s_idx, e_idx in zip(start_idx, end_idx):
            path_indices, dist_matrix = geo_graph_inner(adj_matrix, s_idx, e_idx)
            path_points = points[path_indices]
            paths.append(np.array(path_points))
            approx_geodesic_distance = dist_matrix[e_idx]
            dists.append(approx_geodesic_distance)
        return paths, np.array(dists)
    else:
        path_indices, dist_matrix = geo_graph_inner(adj_matrix, start_idx, end_idx)
        path_points = [np.array(points[path_indices])]
        approx_geodesic_distance = dist_matrix[end_idx]

        return path_points, np.array(approx_geodesic_distance)




# --- optional: curb BLAS oversubscription (do this before importing numpy in real code) ---
# for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
#     os.environ.setdefault(k, "1")

# Global (for forked workers) -----------------
_G_ADJ = None

def _set_global_adj(adj):
    """Call in parent before creating a forked pool."""
    global _G_ADJ
    _G_ADJ = adj

def _dijkstra_chunk(indices_chunk):
    """Worker: run Dijkstra for a chunk of source nodes using global adjacency."""
    # No predecessors, undirected graph
    return dijkstra(_G_ADJ, directed=False, indices=indices_chunk, return_predecessors=False)

# -------------- Utilities --------------------

def get_symmetric_knn_graph(points, k_neighbors=10):
    nn = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto', metric='euclidean')
    nn.fit(points)
    A = nn.kneighbors_graph(mode='distance')            # directed kNN
    A = A.maximum(A.T)                                  # make it undirected
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A.tocsr()

def map_to_sample_vertices(samples, queries):
    # 1-NN in batch
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean').fit(samples)
    idx = nn.kneighbors(queries, return_distance=False)
    return idx.ravel()

# -------------- Fast main --------------------

def graph_pointcloud_distance(
    z0, z1, samples, *, reduce='mean', k_neighbors=10,
    parallel=False, max_workers=None, chunk_sources=128
):
    """
    z0: (T, S, n)   point clouds
    z1: (T, n)      ground-truth points
    samples: (N, n) sampled manifold points (graph nodes)
    Returns: (T,) distances reduced per cloud (mean|min|max)
    """
    z0 = np.asarray(z0, dtype=float)
    z1 = np.asarray(z1, dtype=float)
    samples = np.asarray(samples, dtype=float)
    T, S, n = z0.shape
    assert z1.shape == (T, n)

    # 1) Build symmetric kNN graph once
    adj = get_symmetric_knn_graph(samples, k_neighbors=k_neighbors)

    # 2) Map all points to graph vertices (vectorized)
    start_vids = map_to_sample_vertices(samples, z0.reshape(-1, n)).reshape(T, S)
    end_vids   = map_to_sample_vertices(samples, z1)              # (T,)

    # 3) Unique end vertices → run Dijkstra once per unique end
    uniq_ends, inv = np.unique(end_vids, return_inverse=True)     # len K
    K = len(uniq_ends)

    # Single-process (often fastest due to low Python overhead)
    def run_all_sources(indices_arr):
        return dijkstra(adj, directed=False, indices=indices_arr, return_predecessors=False)  # (K, N)

    if not parallel or K <= 1:
        dist_all = run_all_sources(uniq_ends)                     # (K, N)
    else:
        # Process-parallel across source sets (use fork to avoid pickling the huge adj)
        if max_workers is None or max_workers < 1:
            max_workers = min(8, os.cpu_count() or 1)
        ctx = get_context('fork')  # IMPORTANT on Linux; avoids shipping 'adj' over IPC
        _set_global_adj(adj)
        chunks = [uniq_ends[i:i+chunk_sources] for i in range(0, K, chunk_sources)]
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            parts = list(ex.map(_dijkstra_chunk, chunks))
        dist_all = np.vstack(parts)                                # (K, N)

    # 4) Gather per-instance distances: pick row for each end, then columns for that cloud
    #    dist_rows[i] = distances from end_vids[i] to all nodes
    dist_rows = dist_all[inv]                                      # (T, N)
    d_cloud = dist_rows[np.arange(T)[:, None], start_vids]         # (T, S)

    # 5) Reduce
    if reduce == 'mean':
        out = d_cloud.mean(axis=1)
    elif reduce == 'min':
        out = d_cloud.min(axis=1)
    elif reduce == 'max':
        out = d_cloud.max(axis=1)
    elif reduce == 'none':
        return d_cloud
    else:
        raise ValueError("reduce must be one of {'mean','min','max','none'}")

    # Optional: sanity — unreachable nodes give inf
    # You can decide to mask or raise here as needed.

    return out