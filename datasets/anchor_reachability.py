"""Cached, bounded GT geodesics; no graph search in the training step."""

import hashlib
import os
import tempfile
from collections import OrderedDict

import cv2
import numpy as np
from torch.utils.data._utils.collate import default_collate


def lookup_distance(graph, source, target):
    source, target = np.broadcast_arrays(source, target)
    keys = source.astype(np.int64) * int(graph['node_count']) + target
    stored = graph['keys']
    if stored.size == 0:
        return np.full(keys.shape, np.inf, dtype=np.float32)
    index = np.searchsorted(stored, keys)
    safe = np.minimum(index, stored.size - 1)
    return np.where((index < stored.size) & (stored[safe] == keys),
                    graph['distances'][safe], np.inf)


def build_graph(mask, max_geodesic=96.0):
    from scipy.ndimage import distance_transform_edt
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree
    from datasets.dataset_road_skeleton import RoadSkeletonDataset

    skeleton = RoadSkeletonDataset._skeletonize_binary(mask.astype(np.float32)) > 127
    coords = np.argwhere(skeleton)
    count = len(coords)
    node_map = np.full(mask.shape, -1, dtype=np.int32)
    node_map[skeleton] = np.arange(count)
    if count == 0:
        return dict(node_count=np.array(0), keys=np.empty(0, np.int64),
                    distances=np.empty(0, np.float32), nearest=node_map,
                    snap_distance=np.full(mask.shape, np.inf, np.float32),
                    ambiguous=np.ones(mask.shape, bool), coords=coords)
    rows, cols, lengths = [], [], []
    height, width = mask.shape
    for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                   (0, 1), (1, -1), (1, 0), (1, 1)):
        neighbour = coords + (dy, dx)
        inside = ((neighbour[:, 0] >= 0) & (neighbour[:, 0] < height) &
                  (neighbour[:, 1] >= 0) & (neighbour[:, 1] < width))
        ids = np.flatnonzero(inside)
        other = node_map[neighbour[inside, 0], neighbour[inside, 1]]
        present = other >= 0
        rows.append(ids[present])
        cols.append(other[present])
        lengths.append(np.full(present.sum(), np.hypot(dy, dx), np.float32))
    adjacency = csr_matrix((np.concatenate(lengths),
                            (np.concatenate(rows), np.concatenate(cols))),
                           shape=(count, count))
    # Bounded chunks avoid an N x N dense allocation for complex road masks.
    key_chunks, distance_chunks = [], []
    for start in range(0, count, 32):
        distances = dijkstra(adjacency, directed=False,
                             indices=np.arange(start, min(start + 32, count)),
                             limit=max_geodesic)
        row, col = np.nonzero(np.isfinite(distances))
        key_chunks.append((row.astype(np.int64) + start) * count + col)
        distance_chunks.append(distances[row, col].astype(np.float32))
    snap_distance, indices = distance_transform_edt(~skeleton, return_indices=True)
    nearest = node_map[indices[0], indices[1]]
    graph = dict(node_count=np.array(count), keys=np.concatenate(key_chunks),
                 distances=np.concatenate(distance_chunks), nearest=nearest,
                 snap_distance=snap_distance.astype(np.float32), coords=coords)
    # A near-tie between distant skeleton branches is an ambiguous snap;
    # adjacent pixels on the same branch/junction are not ambiguous.
    points = np.indices(mask.shape).reshape(2, -1).T
    distance, candidate = cKDTree(coords).query(points, k=min(8, count))
    if count == 1:
        ambiguous = np.zeros(mask.shape, bool)
    else:
        primary = nearest.ravel()[:, None]
        geodesic = lookup_distance(graph, primary, candidate)
        ambiguous = ((distance <= distance[:, :1] + 0.75) &
                     (geodesic > 8.0)).any(axis=1).reshape(mask.shape)
    graph['ambiguous'] = ambiguous
    return graph


class ReachabilityCache:
    def __init__(self, directory, max_geodesic=96.0):
        self.directory = directory
        self.max_geodesic = float(max_geodesic)
        self.memory = OrderedDict()
        os.makedirs(directory, exist_ok=True)

    def get(self, mask):
        mask = np.ascontiguousarray(mask > 0.5, dtype=np.uint8)
        digest = hashlib.sha256(mask.tobytes() +
                                repr((mask.shape, self.max_geodesic, 'reach-v1')).encode()).hexdigest()
        if digest in self.memory:
            self.memory.move_to_end(digest)
            return self.memory[digest]
        path = os.path.join(self.directory, digest + '.npz')
        if os.path.isfile(path):
            with np.load(path, allow_pickle=False) as archive:
                graph = {key: archive[key] for key in archive.files}
        else:
            graph = build_graph(mask, self.max_geodesic)
            descriptor, temporary = tempfile.mkstemp(dir=self.directory, suffix='.npz')
            try:
                with os.fdopen(descriptor, 'wb') as handle:
                    np.savez_compressed(handle, **graph)
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.remove(temporary)
        self.memory[digest] = graph
        while len(self.memory) > 8:
            self.memory.popitem(last=False)
        return graph


def collate_anchor_graphs(samples):
    graphs = [sample['anchor_graph'] for sample in samples]
    batch = default_collate([{key: value for key, value in sample.items()
                              if key != 'anchor_graph'} for sample in samples])
    batch['anchor_graph'] = graphs
    return batch
