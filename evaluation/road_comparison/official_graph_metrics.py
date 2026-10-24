"""Adapters for pinned Sat2Graph TOPO and CosmiQ APLS source functions.

Only raster skeleton to graph conversion is ours. Matching, control point
insertion, path scoring and local topology walking use upstream functions.
"""

import ast
from contextlib import redirect_stdout
from functools import lru_cache
import math
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


def _sat_topo_score(gt, pred):
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
            ground, region=region, image="NULL", check=False, direction=False)
        if not seeds:
            return 0.0, 0.0, 0
        pairs = sat_topo.TOPOGeneratePairs(
            proposal, ground, seeds, threshold=0.00010, region=region)
        if not pairs:
            return 0.0, 0.0, len(seeds)
        radius = 0.00150 if 41.0 - min_lat < 0.01000 else 0.00300
        with tempfile.TemporaryDirectory() as temp:
            result = sat_topo.TOPOWithPairs(
                proposal, ground, pairs, seeds, r=radius, step=0.00005,
                threshold=0.00010, outputfile=str(Path(temp) / "topo.txt"),
                one2oneMatching=True, metaData=None)
    precision, matched_recall = sat_topo.topoAvg(result)
    return float(precision), float(matched_recall * len(result) / len(seeds)), len(seeds)


def _official_apls(gt, pred, protocol):
    apls = _cosmiq()
    counts = {"gt": 0, "pred": 0, "infeasible": 0, "unmatched": 0}
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
        # TLTS-Infeasible uses the exact official GT control-point path set.
        for start, end in captures[0][2]:
            if (start not in paths_pred_prime or end not in paths_pred_prime[start]):
                counts["infeasible"] += 1
                if (start not in paths_pred_prime or end not in paths_pred_prime):
                    counts["unmatched"] += 1
    return float(total), float(forward), float(reverse), counts


def graph_scores(gt, pred, protocol):
    precision, recall, seeds = _sat_topo_score(gt, pred)
    apls_total, apls_forward, apls_reverse, pairs = _official_apls(gt, pred, protocol)
    topo_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "topo_precision": precision, "topo_recall": recall, "topo_f1": topo_f1,
        "topo_gt_seeds": seeds,
        "apls_gt_to_pred": apls_forward, "apls_pred_to_gt": apls_reverse,
        "apls_bidirectional": apls_total,
        "apls_gt_pairs": pairs["gt"], "apls_pred_pairs": pairs["pred"],
        "tlts_infeasible": pairs["infeasible"] / pairs["gt"] if pairs["gt"] else 0.0,
        "tlts_infeasible_pairs": pairs["infeasible"],
        "tlts_unmatched_pairs": pairs["unmatched"],
        "tlts_disconnected_pairs": pairs["infeasible"] - pairs["unmatched"],
    }
