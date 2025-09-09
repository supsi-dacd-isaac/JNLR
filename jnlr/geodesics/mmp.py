# import os
# import numpy as np
# from collections import defaultdict
# from concurrent.futures import ProcessPoolExecutor
# from multiprocessing import get_context
# from sklearn.neighbors import KDTree
# import pygeodesic.geodesic as geodesic
#
# # ===== Worker globals =====
# _G_VERTICES = None
# _G_TRIANGLES = None
# _G_GEO = None
# _G_HAS_PROPAGATE = False
#
# # --- 2. Exact geodesics on meshes Mitchell, Mount and Papadimitriou (1987) ---
# def geo_mmp(vertices, start_point, end_point, triangles=None, geoalg=None):
#     """
#     Finds a mesh-based geodesic path using the pygeodesic library.
#     This provides an 'exact' solution to compare against the numerical method.
#     """
#     # 1. Initialize the PyGeodesicAlgorithmExact class with the mesh
#     if geoalg is None:
#       geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, triangles)
#
#     # 2. Find the indices of the start and end points
#     start_idx = np.argmin(np.linalg.norm(vertices - start_point, axis=1))
#     end_idx = np.argmin(np.linalg.norm(vertices - end_point, axis=1))
#
#     # 3. Compute the geodesic distance and path with a single call
#     distance, path = geoalg.geodesicDistance(start_idx, end_idx)
#
#     return path, distance
#
# def _ensure_geo():
#     """Lazy init inside workers (safe both in parent and forked children)."""
#     global _G_GEO, _G_HAS_PROPAGATE
#     if _G_GEO is None:
#         import pygeodesic.geodesic as geodesic
#         _G_GEO = geodesic.PyGeodesicAlgorithmExact(_G_VERTICES, _G_TRIANGLES)
#         _G_HAS_PROPAGATE = hasattr(_G_GEO, "propagate")
#     return _G_GEO, _G_HAS_PROPAGATE
#
# def _init_worker(vertices, triangles):
#     """Run once per worker."""
#     global _G_VERTICES, _G_TRIANGLES, _G_GEO, _G_HAS_PROPAGATE
#     _G_VERTICES = np.ascontiguousarray(vertices, dtype=np.float64)
#     _G_TRIANGLES = np.ascontiguousarray(triangles, dtype=np.int32)
#     _G_GEO = geodesic.PyGeodesicAlgorithmExact(_G_VERTICES, _G_TRIANGLES)
#     _G_HAS_PROPAGATE = hasattr(_G_GEO, "propagate")
#
# def _worker_group(args):
#     end_vid, flat_starts, counts, need_paths = args
#     geo, has_propagate = _ensure_geo()
#
#     S_total = int(flat_starts.size)
#     flat_d = np.empty(S_total, dtype=np.float64)
#     paths = [] if need_paths else None
#
#     if _G_HAS_PROPAGATE:
#         dist_field = None
#         try:
#             _G_GEO.propagate([int(end_vid)])
#             for attr in ("get_distances", "distances", "distance"):
#                 if hasattr(_G_GEO, attr):
#                     tmp = getattr(_G_GEO, attr)()
#                     if tmp is not None:
#                         dist_field = np.asarray(tmp)
#                         break
#         except Exception:
#             dist_field = None
#     else:
#         dist_field = None
#
#     # Fill distances in one pass
#     if dist_field is not None:
#         flat_d[:] = dist_field[flat_starts]
#         if need_paths:
#             # Only paths require per-pair calls
#             cursor = 0
#             for S in counts:
#                 svs = flat_starts[cursor:cursor+S]
#                 inst_paths = []
#                 for sv in svs:
#                     if hasattr(_G_GEO, "geodesicPath"):
#                         _, path = _G_GEO.geodesicPath(int(sv), int(end_vid))
#                     else:
#                         # get exact distance too (could differ slightly from field)
#                         _, path = _G_GEO.geodesicDistance(int(sv), int(end_vid))
#                     inst_paths.append(path)
#                 paths.append(inst_paths)
#                 cursor += S
#     else:
#         # No propagate: compute pairwise
#         cursor = 0
#         if need_paths:
#             for S in counts:
#                 inst_paths = []
#                 for sv in flat_starts[cursor:cursor+S]:
#                     d, path = _G_GEO.geodesicDistance(int(sv), int(end_vid))
#                     flat_d[cursor] = d
#                     inst_paths.append(path)
#                     cursor += 1
#                 paths.append(inst_paths)
#         else:
#             for sv_i, sv in enumerate(flat_starts):
#                 d, _ = _G_GEO.geodesicDistance(int(sv), int(end_vid))
#                 flat_d[sv_i] = d
#
#     return flat_d, paths
#
#
# class MMPBatchEvaluator:
#     def __init__(self, vertices, triangles):
#         self.vertices, self.triangles = self._prepare_mesh(vertices, triangles)
#         self.kdt = KDTree(self.vertices)
#
#         # Make these globals BEFORE starting the pool (so fork inherits them)
#         global _G_VERTICES, _G_TRIANGLES, _G_GEO, _G_HAS_PROPAGATE
#         _G_VERTICES = self.vertices
#         _G_TRIANGLES = self.triangles
#         _G_GEO = None
#         _G_HAS_PROPAGATE = False
#     @staticmethod
#     def _prepare_mesh(vertices, triangles):
#         v = np.asarray(vertices)
#         if v.ndim != 2 or v.shape[1] != 3:
#             raise ValueError("vertices must have shape (N,3)")
#         v = np.ascontiguousarray(v, dtype=np.float64)
#         tri = np.asarray(triangles)
#         if tri.ndim != 2 or tri.shape[1] != 3:
#             raise ValueError("triangles must have shape (M,3)")
#         if not np.issubdtype(tri.dtype, np.integer):
#             raise ValueError("triangles must be integer indices")
#         tri = np.ascontiguousarray(tri, dtype=np.int32)
#         return v, tri
#
#     def _nearest_vertex_indices(self, pts):
#         pts = np.asarray(pts)
#         if pts.ndim == 1:
#             pts = pts[None, :]
#         _, idx = self.kdt.query(pts, k=1)
#         return idx.ravel().astype(np.int32)
#
#     @staticmethod
#     def _reduce(arr, mode):
#         if mode == 'none': return arr
#         if mode == 'mean': return arr.mean(axis=-1)
#         if mode == 'min': return arr.min(axis=-1)
#         if mode == 'max': return arr.max(axis=-1)
#         raise ValueError(f"Unknown reduce: {mode}")
#
#     # -------- Single-instance fast path --------
#     def _compute_instance_local(self, start_vids, end_vid, need_paths):
#         geo = geodesic.PyGeodesicAlgorithmExact(self.vertices, self.triangles)
#         dist_field = None
#         if self.has_propagate:
#             try:
#                 geo.propagate([int(end_vid)])
#                 for attr in ("get_distances", "distances", "distance"):
#                     if hasattr(geo, attr):
#                         tmp = getattr(geo, attr)()
#                         if tmp is not None:
#                             dist_field = np.asarray(tmp)
#                             break
#             except Exception:
#                 dist_field = None
#
#         S = len(start_vids)
#         dists = np.empty(S, dtype=np.float64)
#         paths = [] if need_paths else None
#
#         if dist_field is not None:
#             dists[:] = dist_field[start_vids]
#             if need_paths:
#                 for sv in start_vids:
#                     if hasattr(geo, "geodesicPath"):
#                         _, path = geo.geodesicPath(int(sv), int(end_vid))
#                     else:
#                         _, path = geo.geodesicDistance(int(sv), int(end_vid))
#                     paths.append(path)
#         else:
#             for i, sv in enumerate(start_vids):
#                 d, path = geo.geodesicDistance(int(sv), int(end_vid))
#                 dists[i] = d
#                 if need_paths:
#                     paths.append(path)
#         return dists, paths
#
#     # -------- Batched distances with group-by optimization --------
#     def distances(self,
#                   starts,
#                   ends,
#                   need_paths=False,
#                   reduce='none',
#                   parallel=True,
#                   max_workers=None,
#                   group_chunk=1_000_000  # max number of start vids per submitted group (to bound IPC)
#                   ):
#         if reduce != 'none' and need_paths:
#             raise ValueError("need_paths requires reduce='none'.")
#
#         starts = np.asarray(starts)
#         ends = np.asarray(ends)
#
#         # ---- Single instance: (S,3) + (3,) ----
#         if starts.ndim == 2 and ends.ndim == 1:
#             start_vids = self._nearest_vertex_indices(starts)
#             end_vid = self._nearest_vertex_indices(ends)[0]
#             dists, paths = self._compute_instance_local(start_vids, end_vid, need_paths)
#             dists = self._reduce(dists, reduce)
#             return (dists, paths) if need_paths else dists
#
#         # ---- Batched: (T,S,3) + (T,3) ----
#         if starts.ndim != 3 or ends.ndim != 2 or starts.shape[0] != ends.shape[0] or starts.shape[2] != 3 or ends.shape[1] != 3:
#             raise ValueError("Use shapes (T,S,3) for starts and (T,3) for ends")
#
#         T, S, _ = starts.shape
#
#         # 1) Map all points to nearest vertices (vectorized, single pass)
#         starts_flat = starts.reshape(-1, 3)
#         start_vids_flat = self._nearest_vertex_indices(starts_flat).reshape(T, S)
#         end_vids = self._nearest_vertex_indices(ends)
#
#         # 2) Group instance indices by end_vid
#         groups = defaultdict(list)  # end_vid -> list of t indices
#         for t, ev in enumerate(end_vids):
#             groups[int(ev)].append(t)
#
#         # 3) Build submission payloads per group, with optional chunking to limit payload size
#         submissions = []  # (end_vid, flat_starts_chunk, counts_chunk, need_paths, order_info)
#         order_info = []   # to stitch results back: list of (t_indices_chunk, counts_chunk)
#         for ev, t_list in groups.items():
#             # Concatenate start vids for all t in this group
#             flat = start_vids_flat[t_list].ravel()
#             counts = np.full(len(t_list), S, dtype=np.int32)
#
#             # Chunk to cap IPC size (optional, but robust for huge batches)
#             cursor = 0
#             while cursor < flat.size:
#                 take = min(group_chunk, flat.size - cursor)
#                 # figure how many full instances fit into this chunk
#                 full = take // S
#                 if full == 0:
#                     # ensure at least one instance per chunk
#                     full = 1
#                     take = S
#                 t_chunk = t_list[:full]
#                 start_chunk = start_vids_flat[t_chunk].ravel()
#                 counts_chunk = np.full(full, S, dtype=np.int32)
#
#                 submissions.append((ev, start_chunk.astype(np.int32, copy=False), counts_chunk, need_paths))
#                 order_info.append((t_chunk, counts_chunk))
#                 # advance
#                 t_list = t_list[full:]
#                 flat = flat[take:]
#                 cursor = 0  # we rebuild flat above
#
#         # 4) Execute groups
#         def run_local(ev, start_chunk, counts_chunk, need_paths):
#             # local worker for non-parallel mode
#             geo = geodesic.PyGeodesicAlgorithmExact(self.vertices, self.triangles)
#             dist_field = None
#             if self.has_propagate:
#                 try:
#                     geo.propagate([int(ev)])
#                     for attr in ("get_distances", "distances", "distance"):
#                         if hasattr(geo, attr):
#                             tmp = getattr(geo, attr)()
#                             if tmp is not None:
#                                 dist_field = np.asarray(tmp)
#                                 break
#                 except Exception:
#                     dist_field = None
#
#             flat_d = np.empty(start_chunk.size, dtype=np.float64)
#             paths = [] if need_paths else None
#             if dist_field is not None:
#                 flat_d[:] = dist_field[start_chunk]
#                 if need_paths:
#                     cursor = 0
#                     for S_ in counts_chunk:
#                         inst_paths = []
#                         for sv in start_chunk[cursor:cursor+S_]:
#                             if hasattr(geo, "geodesicPath"):
#                                 _, path = geo.geodesicPath(int(sv), int(ev))
#                             else:
#                                 _, path = geo.geodesicDistance(int(sv), int(ev))
#                             inst_paths.append(path)
#                         paths.append(inst_paths)
#                         cursor += S_
#             else:
#                 cursor = 0
#                 for S_ in counts_chunk:
#                     inst_paths = [] if need_paths else None
#                     for sv in start_chunk[cursor:cursor+S_]:
#                         d, path = geo.geodesicDistance(int(sv), int(ev))
#                         flat_d[cursor] = d
#                         if need_paths:
#                             inst_paths.append(path)
#                         cursor += 1
#                     if need_paths:
#                         paths.append(inst_paths)
#             return flat_d, paths
#
#         results = []
#         if parallel and len(submissions) > 1:
#             if max_workers is None:
#                 max_workers = min(8, os.cpu_count() or 1)  # often best for memory-bound work
#             ctx = get_context("fork")
#             with ProcessPoolExecutor(
#                 max_workers=max_workers,
#                 mp_context=ctx
#             ) as ex:
#                 for out in ex.map(_worker_group, submissions, chunksize=1):
#                     results.append(out)
#         else:
#             # sequential (also used when only one group)
#             for ev, start_chunk, counts_chunk, _np in submissions:
#                 results.append(run_local(ev, start_chunk, counts_chunk, need_paths))
#
#         # 5) Stitch results back to (T, S)
#         dmat = np.empty((T, S), dtype=np.float64)
#         all_paths = [[] for _ in range(T)] if need_paths else None
#
#         for (t_chunk, counts_chunk), (flat_d, paths_chunk) in zip(order_info, results):
#             # rebuild per-instance pieces, in order
#             cursor = 0
#             for i, t in enumerate(t_chunk):
#                 S_ = counts_chunk[i]
#                 dmat[t] = flat_d[cursor:cursor+S_]
#                 if need_paths:
#                     all_paths[t] = paths_chunk[i]
#                 cursor += S_
#
#         dout = self._reduce(dmat, reduce)
#         return (all_paths, dout) if need_paths else dout
#
#
# def mmp_pointcloud_distance(starts,
#                                ends,
#                                vertices,
#                                triangles,
#                                need_paths=False,
#                                reduce='none',
#                                parallel=True,
#                                max_workers=None,
#                                group_chunk=1_000_000):
#     evaluator = MMPBatchEvaluator(vertices, triangles)
#     return evaluator.distances(
#         starts, ends,
#         need_paths=need_paths,
#         reduce=reduce,
#         parallel=parallel,
#         max_workers=max_workers,
#         group_chunk=group_chunk
#     )

import os
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from sklearn.neighbors import KDTree
import pygeodesic.geodesic as geodesic

_G_VERTICES = None
_G_TRIANGLES = None
_G_GEO = None
_G_HAS_PROPAGATE = False

def _ensure_geo():
    global _G_GEO, _G_HAS_PROPAGATE
    if _G_GEO is None:
        _G_GEO = geodesic.PyGeodesicAlgorithmExact(_G_VERTICES, _G_TRIANGLES)
        _G_HAS_PROPAGATE = hasattr(_G_GEO, "propagate")
    return _G_GEO, _G_HAS_PROPAGATE

def _init_worker(vertices, triangles):
    global _G_VERTICES, _G_TRIANGLES, _G_GEO, _G_HAS_PROPAGATE
    _G_VERTICES = np.ascontiguousarray(vertices, dtype=np.float64)
    _G_TRIANGLES = np.ascontiguousarray(triangles, dtype=np.int32)
    _G_GEO = geodesic.PyGeodesicAlgorithmExact(_G_VERTICES, _G_TRIANGLES)
    _G_HAS_PROPAGATE = hasattr(_G_GEO, "propagate")

def _worker_group(args):
    end_vid, flat_starts, counts, need_paths = args
    geo, has_propagate = _ensure_geo()

    S_total = int(flat_starts.size)
    flat_d = np.empty(S_total, dtype=np.float64)
    paths = [] if need_paths else None

    if has_propagate:
        dist_field = None
        try:
            geo.propagate([int(end_vid)])
            for attr in ("get_distances", "distances", "distance"):
                if hasattr(geo, attr):
                    tmp = getattr(geo, attr)()
                    if tmp is not None:
                        dist_field = np.asarray(tmp)
                        break
        except Exception:
            dist_field = None
    else:
        dist_field = None

    if dist_field is not None:
        flat_d[:] = dist_field[flat_starts]
        if need_paths:
            cursor = 0
            for S in counts:
                inst_paths = []
                for sv in flat_starts[cursor:cursor+S]:
                    if hasattr(geo, "geodesicPath"):
                        _, path = geo.geodesicPath(int(sv), int(end_vid))
                    else:
                        _, path = geo.geodesicDistance(int(sv), int(end_vid))
                    inst_paths.append(path)
                paths.append(inst_paths)
                cursor += S
    else:
        cursor = 0
        if need_paths:
            for S in counts:
                inst_paths = []
                for sv in flat_starts[cursor:cursor+S]:
                    d, path = geo.geodesicDistance(int(sv), int(end_vid))
                    flat_d[cursor] = d
                    inst_paths.append(path)
                    cursor += 1
                paths.append(inst_paths)
        else:
            for i, sv in enumerate(flat_starts):
                d, _ = geo.geodesicDistance(int(sv), int(end_vid))
                flat_d[i] = d
    return flat_d, paths

class MMPBatchEvaluator:
    def __init__(self, vertices, triangles):
        self.vertices, self.triangles = self._prepare_mesh(vertices, triangles)
        self.kdt = KDTree(self.vertices)
        test_geo = geodesic.PyGeodesicAlgorithmExact(self.vertices, self.triangles)
        self.has_propagate = hasattr(test_geo, "propagate")
        global _G_VERTICES, _G_TRIANGLES, _G_GEO, _G_HAS_PROPAGATE
        _G_VERTICES = self.vertices
        _G_TRIANGLES = self.triangles
        _G_GEO = None
        _G_HAS_PROPAGATE = False

    @staticmethod
    def _prepare_mesh(vertices, triangles):
        v = np.asarray(vertices)
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError("vertices must have shape (N,3)")
        v = np.ascontiguousarray(v, dtype=np.float64)
        tri = np.asarray(triangles)
        if tri.ndim != 2 or tri.shape[1] != 3:
            raise ValueError("triangles must have shape (M,3)")
        if not np.issubdtype(tri.dtype, np.integer):
            raise ValueError("triangles must be integer indices")
        tri = np.ascontiguousarray(tri, dtype=np.int32)
        return v, tri

    def _nearest_vertex_indices(self, pts):
        pts = np.asarray(pts)
        if pts.ndim == 1:
            pts = pts[None, :]
        _, idx = self.kdt.query(pts, k=1)
        return idx.ravel().astype(np.int32)

    @staticmethod
    def _reduce(arr, mode):
        if mode == 'none': return arr
        if mode == 'mean': return arr.mean(axis=-1)
        if mode == 'min': return arr.min(axis=-1)
        if mode == 'max': return arr.max(axis=-1)
        raise ValueError(f"Unknown reduce: {mode}")

    def _compute_instance_local(self, start_vids, end_vid, need_paths):
        geo = geodesic.PyGeodesicAlgorithmExact(self.vertices, self.triangles)
        dist_field = None
        if self.has_propagate:
            try:
                geo.propagate([int(end_vid)])
                for attr in ("get_distances", "distances", "distance"):
                    if hasattr(geo, attr):
                        tmp = getattr(geo, attr)()
                        if tmp is not None:
                            dist_field = np.asarray(tmp)
                            break
            except Exception:
                dist_field = None
        S = len(start_vids)
        dists = np.empty(S, dtype=np.float64)
        paths = [] if need_paths else None
        if dist_field is not None:
            dists[:] = dist_field[start_vids]
            if need_paths:
                for sv in start_vids:
                    if hasattr(geo, "geodesicPath"):
                        _, path = geo.geodesicPath(int(sv), int(end_vid))
                    else:
                        _, path = geo.geodesicDistance(int(sv), int(end_vid))
                    paths.append(path)
        else:
            for i, sv in enumerate(start_vids):
                d, path = geo.geodesicDistance(int(sv), int(end_vid))
                dists[i] = d
                if need_paths:
                    paths.append(path)
        return dists, paths

    def distances(self,
                  starts,
                  ends,
                  need_paths=False,
                  reduce='none',
                  parallel=True,
                  max_workers=None,
                  group_chunk=1_000_000):
        if reduce != 'none' and need_paths:
            raise ValueError("need_paths requires reduce='none'.")

        starts = np.asarray(starts)
        ends = np.asarray(ends)

        # Single instance
        if starts.ndim == 2 and ends.ndim == 1:
            start_vids = self._nearest_vertex_indices(starts)
            end_vid = self._nearest_vertex_indices(ends)[0]
            dists, paths = self._compute_instance_local(start_vids, end_vid, need_paths)
            dists = self._reduce(dists, reduce)
            return (paths, dists) if need_paths else dists

        # Batched validation
        if not (starts.ndim == 3 and ends.ndim == 2 and
                starts.shape[0] == ends.shape[0] and
                starts.shape[2] == 3 and ends.shape[1] == 3):
            raise ValueError("Use shapes (T,S,3) for starts and (T,3) for ends")

        T, S, _ = starts.shape

        # Map to vertex ids
        start_vids_flat = self._nearest_vertex_indices(starts.reshape(-1, 3)).reshape(T, S)
        end_vids = self._nearest_vertex_indices(ends)

        # Group by end vertex
        groups = defaultdict(list)
        for t, ev_id in enumerate(end_vids):
            groups[int(ev_id)].append(t)

        submissions = []
        order_info = []
        for end_vid, t_list in groups.items():
            # Flatten all starts for these instances
            flat = start_vids_flat[t_list].ravel()
            counts_full = np.full(len(t_list), S, dtype=np.int32)
            cursor = 0
            remaining_t = t_list
            remaining_flat = flat
            while remaining_t:
                max_items = min(group_chunk, remaining_flat.size)
                full_instances = max_items // S
                if full_instances == 0:
                    full_instances = 1
                t_chunk = remaining_t[:full_instances]
                start_chunk = start_vids_flat[t_chunk].ravel().astype(np.int32, copy=False)
                counts_chunk = np.full(len(t_chunk), S, dtype=np.int32)
                submissions.append((end_vid, start_chunk, counts_chunk, need_paths))
                order_info.append((t_chunk, counts_chunk))
                remaining_t = remaining_t[full_instances:]
                remaining_flat = remaining_flat[full_instances * S:]

        results = []
        if parallel and len(submissions) > 1:
            if max_workers is None:
                max_workers = min(8, os.cpu_count() or 1)
            ctx = get_context("fork")
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
                for out in ex.map(_worker_group, submissions, chunksize=1):
                    results.append(out)
        else:
            for sub in submissions:
                results.append(_worker_group(sub))

        dmat = np.empty((T, S), dtype=np.float64)
        all_paths = [[] for _ in range(T)] if need_paths else None

        for (t_chunk, counts_chunk), (flat_d, paths_chunk) in zip(order_info, results):
            cursor = 0
            for i, t in enumerate(t_chunk):
                S_ = counts_chunk[i]
                dmat[t] = flat_d[cursor:cursor+S_]
                if need_paths:
                    all_paths[t] = paths_chunk[i]
                cursor += S_

        dout = self._reduce(dmat, reduce)
        return (all_paths, dout) if need_paths else dout




def geo_mmp(starts,
            ends,
            vertices,
            triangles,
            need_paths=False,
            reduce='none',
            parallel=True,
            max_workers=None,
            group_chunk=1_000_000,
            unwrap_single=True):
    evaluator = MMPBatchEvaluator(vertices, triangles)
    out = evaluator.distances(
        starts, ends,
        need_paths=need_paths,
        reduce=reduce,
        parallel=parallel,
        max_workers=max_workers,
        group_chunk=group_chunk
    )
    if not need_paths:
        return out  # distances only

    paths, dists = out  # existing order: (paths, distances)

    # if path is a list containing singleton lists, flatten the list
    if np.all([isinstance(p, list) and len(p) == 1 for p in paths]) and unwrap_single:
        paths = [p[0] for p in paths]
        dists = dists if dists.ndim == 0 else dists.ravel()
    return paths, dists