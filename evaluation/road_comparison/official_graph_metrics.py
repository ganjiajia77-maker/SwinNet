"""Adapters for pinned Sat2Graph TOPO and CosmiQ APLS source functions.

Only raster skeleton to graph conversion is ours. Matching, control point
insertion, path scoring and local topology walking use upstream functions.
"""

import ast
import hashlib
from contextlib import redirect_stdout
from functools import lru_cache
import math
import random
import os
from pathlib import Path
import tempfile
import time
from types import ModuleType

import networkx as nx
import numpy as np
import scipy
import shapely
from shapely.geometry import LineString, Point
import utm

from vendor.sat2graph import graph as sat_graph
from vendor.sat2graph import topo as sat_topo


VENDOR = Path(__file__).parent / "vendor" / "cosmiq"
TLTS_VENDOR = Path(__file__).parent / "vendor" / "metrics_delin"


class CompatibleMultiGraph(nx.MultiGraph):
    """Expose the NetworkX 1.x node alias used by CosmiQ's KD tree helper."""

    @property
    def node(self):
        return self.nodes

    def to_undirected_class(self):
        return CompatibleMultiGraph


def _load_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    missing = set(names) - {node.name for node in selected}
    if missing:
        raise RuntimeError(f"Pinned CosmiQ source lacks functions: {sorted(missing)}")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


@lru_cache(maxsize=1)
def _cosmiq():
    import copy
    import random

    utils = ModuleType("apls_utils")
    utils.__dict__.update(np=np, scipy=scipy, time=time)
    _load_functions(VENDOR / "apls_utils.upstream.py.txt",
                    ("nodes_near_point", "G_to_kdtree", "_query_kd_nearest", "_query_kd_ball"),
                    utils.__dict__)
    module = ModuleType("apls_official_core")
    module.__dict__.update(nx=nx, np=np, scipy=scipy, shapely=shapely,
                           Point=Point, LineString=LineString, utm=utm,
                           copy=copy, random=random, math=math, time=time, os=os,
                           apls_utils=utils)
    _load_functions(VENDOR / "apls.upstream.py.txt",
                    ("add_travel_time", "create_edge_linestrings", "cut_linestring",
                     "get_closest_edge_from_G", "insert_point_into_G", "insert_control_points",
                     "create_graph_midpoints", "make_graphs", "single_path_metric",
                     "path_sim_metric", "compute_apls_metric"), module.__dict__)
    return module


@lru_cache(maxsize=1)
def _tlts():
    """Load the paper authors' TLTS function and its required helpers."""
    import time as clock

    utils = ModuleType("metrics_delin_tlts_utils")
    utils.__dict__.update(np=np, nx=nx, random=random, time=clock)
    _load_functions(TLTS_VENDOR / "utils.upstream.py.txt",
                    ("is_empty", "interpolate_new_nodes", "oversampling_graph",
                     "length_path", "find_closest", "uniform_node_sampling"), utils.__dict__)
    # The released code targets NetworkX 1.x; its rare fallback passes a
    # NodeView to np.random.choice. Adapt only that API call for NetworkX 3.x.
    sampler = utils.uniform_node_sampling
    source = (TLTS_VENDOR / "utils.upstream.py.txt").read_text(encoding="utf-8")
    source = source.replace("np.random.choice(G.nodes())", "np.random.choice(list(G.nodes()))")
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "uniform_node_sampling")
    exec(compile(ast.Module(body=[node], type_ignores=[]), sampler.__code__.co_filename,
                 "exec"), utils.__dict__)
    module = ModuleType("metrics_delin_tlts_official")
    module.__dict__.update(np=np, nx=nx, random=random, utils=utils)
    _load_functions(TLTS_VENDOR / "toolong_tooshort_metric.upstream.py.txt",
                    ("toolong_tooshort",), module.__dict__)
    return module


class _NullWriter:
    def write(self, value):
        return len(value)

    def flush(self):
        pass


def _neighbors(csr):
    return [list(map(int, csr.indices[csr.indptr[i]:csr.indptr[i + 1]]))
            for i in range(csr.shape[0])]


def compact_graph(points, csr, id_offset=0):
    """Contract degree-two chains without changing their pixel polyline or length."""
    graph = CompatibleMultiGraph()
    neighbors = _neighbors(csr)
    n = len(points)
    anchors = {i for i, adjacent in enumerate(neighbors) if len(adjacent) != 2}
    visited = set()

    def pair(a, b):
        return (min(a, b), max(a, b))

    def add_node(i):
        y, x = map(int, points[i])
        graph.add_node(int(i + id_offset), x=float(x), y=float(y))

    def add_chain(chain):
        if len(chain) < 2:
            return
        coords = [(float(points[i, 1]), float(points[i, 0])) for i in chain]
        for i in (chain[0], chain[-1]):
            add_node(i)
        geometry = LineString(coords)
        graph.add_edge(int(chain[0] + id_offset), int(chain[-1] + id_offset),
                       geometry=geometry, length=float(geometry.length),
                       inferred_speed_mps=1.0)

    for anchor in sorted(anchors):
        add_node(anchor)
        for neighbor in neighbors[anchor]:
            if pair(anchor, neighbor) in visited:
                continue
            chain = [anchor, neighbor]
            visited.add(pair(anchor, neighbor))
            prev, current = anchor, neighbor
            while current not in anchors:
                nxt = next(v for v in neighbors[current] if v != prev)
                if pair(current, nxt) in visited:
                    raise ValueError("Pixel graph traversal repeated an edge")
                visited.add(pair(current, nxt))
                chain.append(nxt)
                prev, current = current, nxt
            if chain[0] == chain[-1]:
                # NetworkX loops lose the two alternative routes around a ring.
                cuts = sorted(set((0, len(chain) // 3, 2 * len(chain) // 3, len(chain) - 1)))
                for a, b in zip(cuts, cuts[1:]):
                    add_chain(chain[a:b + 1])
            else:
                add_chain(chain)

    # All-degree-two components are closed loops without a natural anchor.
    for start in range(n):
        if start in anchors:
            continue
        for neighbor in neighbors[start]:
            if pair(start, neighbor) in visited:
                continue
            chain = [start, neighbor]
            visited.add(pair(start, neighbor))
            prev, current = start, neighbor
            while current != start:
                nxt = next(v for v in neighbors[current] if v != prev)
                if pair(current, nxt) in visited:
                    raise ValueError("Pixel ring traversal repeated an edge")
                visited.add(pair(current, nxt))
                chain.append(nxt)
                prev, current = current, nxt
            cuts = sorted(set((0, len(chain) // 3, 2 * len(chain) // 3, len(chain) - 1)))
            for a, b in zip(cuts, cuts[1:]):
                add_chain(chain[a:b + 1])

    if sum(len(v) for v in neighbors) // 2 != len(visited):
        raise ValueError("Pixel graph contraction lost an edge")
    return graph


def _sat_coordinates(y, x):
    # Sat2Graph main.py's synthetic coordinates: one image pixel is one metre.
    return 41.0 - float(y) / 111111.0, -71.0 + float(x) / (111111.0 * math.cos(math.radians(41)))


def _sat_road_graph(points, csr):
    graph = sat_graph.RoadGraph()
    for i in range(len(points)):
        lat1, lon1 = _sat_coordinates(*points[i])
        for j in csr.indices[csr.indptr[i]:csr.indptr[i + 1]]:
            lat2, lon2 = _sat_coordinates(*points[j])
            graph.addEdge(int(i), lat1, lon1, int(j), lat2, lon2,
                          nodeScore1=100, nodeScore2=100, edgeScore=100)
    graph.ReverseDirectionLink()
    return graph


def prepare_official_graph(points, csr, id_offset=0):
    return {"points": points, "csr": csr,
            "compact": compact_graph(points, csr, id_offset)}


def _tlts_graph(points, csr):
    """Represent the exact eight-neighbour raster skeleton for official TLTS."""
    graph = nx.Graph()
    graph.add_nodes_from((int(i), {"pos": (float(x), float(y))})
                         for i, (y, x) in enumerate(points))
    for i in range(len(points)):
        graph.add_edges_from((i, int(j)) for j in csr.indices[csr.indptr[i]:csr.indptr[i + 1]]
                             if j > i)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    return graph


def _official_tlts(gt, pred, protocol):
    reference = _tlts_graph(gt["points"], gt["csr"])
    proposed = _tlts_graph(pred["points"], pred["csr"])
    seed_bytes = hashlib.sha256(
        np.asarray(gt["points"], dtype=np.int32).tobytes()
        + int(protocol["tlts_seed"]).to_bytes(8, "little")
    ).digest()
    seed = int.from_bytes(seed_bytes[:4], "little")
    # The upstream function seeds NumPy globally. Preserve both RNG states so
    # repeat runs and model order do not affect its randomized path sampling.
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        random.seed(seed)
        with redirect_stdout(_NullWriter()):
            values = _tlts().toolong_tooshort(
                reference, proposed,
                max_node_dist=protocol["tlts_max_node_dist_px"],
                perc_length=protocol["tlts_length_tolerance"],
                n_paths=protocol["tlts_n_paths"], seed=seed)
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
    if len(values) != 4 or not np.isfinite(values).all() or not np.isclose(sum(values), 1.0):
        raise RuntimeError(f"Official TLTS returned invalid path categories: {values}")
    return tuple(map(float, values))


def _sat_topo_score(gt, pred, protocol):
    if len(gt["points"]) == 0 and len(pred["points"]) == 0:
        return 1.0, 1.0, 0
    if len(gt["points"]) == 0 or len(pred["points"]) == 0:
        return 0.0, 0.0, 0
    ground = _sat_road_graph(gt["points"], gt["csr"])
    proposal = _sat_road_graph(pred["points"], pred["csr"])
    all_points = np.vstack((gt["points"], pred["points"]))
    min_lat, _ = _sat_coordinates(int(all_points[:, 0].max()), 0)
    _, max_lon = _sat_coordinates(0, int(all_points[:, 1].max()))
    region = [min_lat - 300 / 111111.0, -71.0 - 500 / 111111.0,
              41.0 + 300 / 111111.0, max_lon + 500 / 111111.0]
    ground.region = proposal.region = region
    with redirect_stdout(_NullWriter()):
        seeds = sat_topo.TOPOGenerateStartingPoints(
            ground, density=protocol["topo_seed_density_deg"], region=region,
            image="NULL", check=False, direction=False)
        if not seeds:
            return 0.0, 0.0, 0
        pairs = sat_topo.TOPOGeneratePairs(
            proposal, ground, seeds, threshold=0.00010, region=region)
        if not pairs:
            return 0.0, 0.0, len(seeds)
        radius = 0.00150 if 41.0 - min_lat < 0.01000 else 0.00300
        with tempfile.TemporaryDirectory() as temp:
            result = sat_topo.TOPOWithPairs(
                proposal, ground, pairs, seeds, r=radius,
                step=protocol["topo_step_deg"],
                threshold=0.00010, outputfile=str(Path(temp) / "topo.txt"),
                one2oneMatching=True, metaData=None)
    precision, matched_recall = sat_topo.topoAvg(result)
    return float(precision), float(matched_recall * len(result) / len(seeds)), len(seeds)


def _official_apls(gt, pred, protocol):
    apls = _cosmiq()
    counts = {"gt": 0, "pred": 0}
    with redirect_stdout(_NullWriter()):
        prepared = apls.make_graphs(
            gt["compact"].copy(), pred["compact"].copy(), weight="length",
            linestring_delta=protocol["apls_control_spacing_px"],
            is_curved_eps=-1, max_snap_dist=protocol["apls_snap_radius"],
            allow_renaming=True)
        if prepared is None:
            raise RuntimeError("Official CosmiQ make_graphs returned no graph tuple")
        (_, _, _, _, controls_gt, controls_pred,
         paths_gt, paths_pred, paths_gt_prime, paths_pred_prime) = prepared
        captures = []
        official_path = apls.path_sim_metric

        def capture(*args, **kwargs):
            result = official_path(*args, **kwargs)
            captures.append(result)
            return result

        apls.path_sim_metric = capture
        try:
            total, forward, reverse = apls.compute_apls_metric(
                paths_gt, paths_pred, paths_gt_prime, paths_pred_prime,
                controls_gt, controls_pred, min_path_length=protocol["apls_min_path_px"])
        finally:
            apls.path_sim_metric = official_path
    if captures:
        for direction, captured in zip(("gt", "pred"), captures):
            counts[direction] = len(captured[2])
    return float(total), float(forward), float(reverse), counts


def graph_scores(gt, pred, protocol, timing_label=None):
    started = time.perf_counter()
    precision, recall, seeds = _sat_topo_score(gt, pred, protocol)
    topo_seconds = time.perf_counter() - started
    if timing_label:
        print(f"[TIMING] {timing_label} TOPO={topo_seconds:.1f}s seeds={seeds}", flush=True)
    started = time.perf_counter()
    apls_total, apls_forward, apls_reverse, pairs = _official_apls(gt, pred, protocol)
    apls_seconds = time.perf_counter() - started
    if timing_label:
        print(f"[TIMING] {timing_label} APLS={apls_seconds:.1f}s", flush=True)
    started = time.perf_counter()
    tlts_correct, tlts_too_long, tlts_too_short, tlts_infeasible = _official_tlts(
        gt, pred, protocol)
    tlts_seconds = time.perf_counter() - started
    if timing_label:
        print(f"[TIMING] {timing_label} TLTS={tlts_seconds:.1f}s", flush=True)
    topo_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "topo_precision": precision, "topo_recall": recall, "topo_f1": topo_f1,
        "topo_gt_seeds": seeds,
        "apls_gt_to_pred": apls_forward, "apls_pred_to_gt": apls_reverse,
        "apls_bidirectional": apls_total,
        "apls_gt_pairs": pairs["gt"], "apls_pred_pairs": pairs["pred"],
        "tlts_correct": tlts_correct, "tlts_too_long": tlts_too_long,
        "tlts_too_short": tlts_too_short, "tlts_infeasible": tlts_infeasible,
        "timing_topo_seconds": topo_seconds,
        "timing_apls_seconds": apls_seconds,
        "timing_tlts_seconds": tlts_seconds,
    }
