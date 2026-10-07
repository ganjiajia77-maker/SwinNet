"""Sparse, prediction-driven road-fragment and candidate-path logit correction.

The proposal step is deliberately discrete. All learned token, path and raster
operations after proposal creation remain differentiable.
"""

import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


_NEIGHBORS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1),
              (1, -1), (1, 0), (1, 1))


def _thin(mask):
    """Vectorized Zhang-Suen thinning, with no optional OpenCV-contrib dependency."""
    image = mask.astype(np.uint8).copy()
    while True:
        removed = 0
        for first_pass in (True, False):
            p = np.pad(image, 1)
            p2, p3, p4 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:]
            p5, p6, p7 = p[2:, 2:], p[2:, 1:-1], p[2:, :-2]
            p8, p9 = p[1:-1, :-2], p[:-2, :-2]
            around = (p2, p3, p4, p5, p6, p7, p8, p9)
            degree = sum(around)
            transitions = sum(((around[k] == 0) & (around[(k + 1) % 8] == 1)).astype(np.uint8)
                              for k in range(8))
            if first_pass:
                condition = ((p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0))
            else:
                condition = ((p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0))
            remove = ((image == 1) & (degree >= 2) & (degree <= 6)
                      & (transitions == 1) & condition)
            removed += int(remove.sum())
            image[remove] = 0
        if removed == 0:
            return image.astype(bool)


def _graph_fragments(skeleton):
    """Split centerlines at merged junction neighborhoods and return branches."""
    height, width = skeleton.shape
    points = [tuple(v) for v in np.argwhere(skeleton)]
    if not points:
        return [], []
    point_set = set(points)
    neighbors = {point: [other for dy, dx in _NEIGHBORS
                         if (other := (point[0] + dy, point[1] + dx)) in point_set]
                 for point in points}
    junction = np.zeros_like(skeleton, dtype=np.uint8)
    for point, adjacent in neighbors.items():
        if len(adjacent) >= 3:
            junction[point] = 1
    # A single intersection may have several neighboring degree-3 pixels.
    expanded = cv2.dilate(junction, np.ones((3, 3), np.uint8))
    _, cluster_map = cv2.connectedComponents(expanded, connectivity=8)
    node_pixels = {}
    pixel_node = {}
    for point in points:
        cluster = int(cluster_map[point]) if junction[point] else 0
        if cluster:
            key = ("junction", cluster)
        elif len(neighbors[point]) <= 1:
            key = ("endpoint", point)
        else:
            continue
        node_pixels.setdefault(key, []).append(point)
        pixel_node[point] = key
    used_edges = set()
    fragments = []
    for node, members in node_pixels.items():
        if node[0] == "junction":
            for point in members:
                for adjacent in neighbors[point]:
                    if pixel_node.get(adjacent) == node:
                        used_edges.add(tuple(sorted((point, adjacent))))

    def trace(start, nxt):
        route = [start]
        previous, current = start, nxt
        while True:
            edge = tuple(sorted((previous, current)))
            if edge in used_edges:
                break
            used_edges.add(edge)
            route.append(current)
            if current in pixel_node and current != start:
                break
            options = [p for p in neighbors[current] if p != previous
                       and tuple(sorted((current, p))) not in used_edges]
            if not options:
                break
            previous, current = current, options[0]
        return route

    for node, members in node_pixels.items():
        for start in members:
            for nxt in neighbors[start]:
                if pixel_node.get(nxt) == node:
                    continue
                if tuple(sorted((start, nxt))) in used_edges:
                    continue
                route = trace(start, nxt)
                if len(route) > 1 and (pixel_node.get(route[-1]) != node or len(route) > 8):
                    fragments.append(route)
    # Closed loops have no endpoints or junctions. Preserve them as a branch.
    for point in points:
        for nxt in neighbors[point]:
            if tuple(sorted((point, nxt))) not in used_edges:
                route = trace(point, nxt)
                if len(route) > 1:
                    fragments.append(route)
    return fragments, node_pixels


def _propose(probability, threshold=0.45):
    """Create branch regions and endpoint records entirely from detached P0."""
    candidate = (probability >= threshold).astype(np.uint8)
    count, component, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    skeleton = _thin(candidate)
    height, width = candidate.shape
    branches = []
    endpoints = []
    for component_id in range(1, count):
        if stats[component_id, cv2.CC_STAT_AREA] < 2:
            continue
        x0, y0, box_w, box_h = stats[component_id, :4]
        region = component[y0:y0 + box_h, x0:x0 + box_w] == component_id
        thin_region = skeleton[y0:y0 + box_h, x0:x0 + box_w] & region
        routes, nodes = _graph_fragments(thin_region)
        if not routes:
            routes = [[tuple(v) for v in np.argwhere(thin_region)]]
        if not routes[0]:
            continue
        skeleton_owner = np.full(region.shape, -1, dtype=np.int32)
        for route_id, route in enumerate(routes):
            for py, px in route:
                skeleton_owner[py, px] = route_id
        _, nearest = ndimage.distance_transform_edt(
            skeleton_owner < 0, return_indices=True)
        region_owner = skeleton_owner[tuple(nearest)]
        route_ids = {}
        for route_id, route in enumerate(routes):
            yy, xx = np.where(region & (region_owner == route_id))
            if not yy.size:
                continue
            pixels = ((yy + y0) * width + xx + x0).astype(np.int64)
            coords = np.asarray(route, dtype=np.float32)
            midpoint = coords[len(coords) // 2] + np.asarray([y0, x0])
            mean_prob = float(probability.flat[pixels].mean())
            route_ids[route_id] = len(branches)
            branches.append({"pixels": pixels, "coord": midpoint,
                             "length": len(route), "area": len(pixels),
                             "confidence": mean_prob})
        for node, members in nodes.items():
            if node[0] != "endpoint":
                continue
            py, px = members[0]
            attached = [route_ids[idx] for idx, route in enumerate(routes)
                        if idx in route_ids and (py, px) in route]
            if attached:
                endpoints.append({"coord": np.asarray([py + y0, px + x0], np.float32),
                                  "branch": attached[0],
                                  "confidence": float(probability[py + y0, px + x0])})
    return branches, endpoints


def _path_pixels(start, finish, bend, height, width):
    start = np.asarray(start, dtype=np.float32)
    finish = np.asarray(finish, dtype=np.float32)
    midpoint = (start + finish) / 2
    distance = float(np.linalg.norm(finish - start))
    normal = np.asarray([-(finish[1] - start[1]), finish[0] - start[0]])
    normal /= max(distance, 1.0)
    control = midpoint + bend * min(8.0, distance * 0.25) * normal
    steps = max(2, int(math.ceil(distance * 1.5)))
    t = np.linspace(0, 1, steps, dtype=np.float32)[:, None]
    points = ((1 - t) ** 2 * start + 2 * (1 - t) * t * control + t ** 2 * finish)
    # Ordered centerline pixels are retained for contiguous-background labels.
    ordered = np.rint(points).astype(np.int64)
    ordered[:, 0] = np.clip(ordered[:, 0], 0, height - 1)
    ordered[:, 1] = np.clip(ordered[:, 1], 0, width - 1)
    ordered = np.unique(ordered[:, 0] * width + ordered[:, 1], return_index=True)
    ordered = ordered[0][np.argsort(ordered[1])]
    band = np.zeros((height, width), dtype=np.uint8)
    for p0, p1 in zip(points[:-1], points[1:]):
        a = tuple(np.rint(p0[::-1]).astype(int))
        b = tuple(np.rint(p1[::-1]).astype(int))
        cv2.line(band, a, b, 1, thickness=3)
    return np.flatnonzero(band).astype(np.int64), ordered.astype(np.int64)


class FragmentPathTopology(nn.Module):
    """Token-only relation attention with supervised signed surface correction."""

    def __init__(self, channels, h3_channels, max_tokens=64, heads=4,
                 candidate_threshold=0.45, radius=32, max_paths=128):
        super().__init__()
        if 64 % heads:
            raise ValueError("64 token channels must be divisible by attention heads")
        self.max_tokens = int(max_tokens)
        self.max_paths = int(max_paths)
        self.radius = float(radius)
        self.candidate_threshold = float(candidate_threshold)
        self.heads = int(heads)
        feature_channels = channels + h3_channels
        self.type_embedding = nn.Embedding(2, 4)
        self.token_projection = nn.Linear(feature_channels + 10, 64)
        self.relation_qkv = nn.Linear(64, 192)
        self.relation_bias = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, heads))
        self.relation_output = nn.Linear(64, 64)
        self.token_norm = nn.LayerNorm(64)
        self.validity_head = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
        self.path_head = nn.Sequential(nn.Linear(128 + feature_channels + 5, 64),
                                       nn.GELU(), nn.Linear(64, 1))
        initial = math.log(math.expm1(0.1))
        self.raw_beta_add = nn.Parameter(torch.tensor(initial))
        self.raw_beta_remove = nn.Parameter(torch.tensor(initial))

    @property
    def beta_add(self):
        return F.softplus(self.raw_beta_add)

    @property
    def beta_remove(self):
        return F.softplus(self.raw_beta_remove)

    def _attend(self, tokens, coords):
        count = tokens.shape[0]
        qkv = self.relation_qkv(tokens).reshape(count, 3, self.heads, 64 // self.heads)
        query, key, value = qkv.permute(1, 2, 0, 3)
        logits = query @ key.transpose(-2, -1) / math.sqrt(64 // self.heads)
        delta = coords[:, None] - coords[None, :]
        distance = torch.linalg.vector_norm(delta, dim=-1)
        relation = torch.stack((delta[..., 0], delta[..., 1],
                                torch.log1p(distance), 1 / (1 + distance)), dim=-1)
        logits = logits + self.relation_bias(relation).permute(2, 0, 1)
        attended = (logits.softmax(dim=-1) @ value).transpose(0, 1).reshape(count, 64)
        return self.token_norm(tokens + self.relation_output(attended))

    def forward(self, feature, h3, preliminary_logits):
        batch, _, height, width = preliminary_logits.shape
        if h3.shape[-2:] != (height, width):
            h3 = F.interpolate(h3, (height, width), mode="bilinear", align_corners=False)
        probability = preliminary_logits.sigmoid()
        proposals = probability.detach().float().cpu().numpy()[:, 0]
        combined = torch.cat((feature, h3), dim=1)
        add_maps, remove_maps, records = [], [], []
        for image_idx in range(batch):
            branches, endpoints = _propose(proposals[image_idx], self.candidate_threshold)
            candidates = []
            for branch_idx, branch in enumerate(branches):
                # Small or uncertain branches receive priority over easy large branches.
                priority = 1 / math.sqrt(max(branch["area"], 1)) + 1 - branch["confidence"]
                candidates.append((priority, "branch", branch_idx))
            for endpoint_idx, endpoint in enumerate(endpoints):
                candidates.append((2.0 + 1 - endpoint["confidence"], "endpoint", endpoint_idx))
            candidates.sort(key=lambda item: item[0], reverse=True)
            endpoint_budget = self.max_tokens // 2
            branch_budget = self.max_tokens - endpoint_budget
            selected = ([item for item in candidates if item[1] == "endpoint"][:endpoint_budget]
                        + [item for item in candidates if item[1] == "branch"][:branch_budget])
            chosen = {(item[1], item[2]) for item in selected}
            selected.extend(item for item in candidates if (item[1], item[2]) not in chosen)
            selected = selected[:self.max_tokens]
            flat_features = combined[image_idx].flatten(1)
            flat_prob = probability[image_idx, 0].flatten()
            token_inputs, coords = [], []
            for _, kind, idx in selected:
                item = branches[idx] if kind == "branch" else endpoints[idx]
                y, x = np.rint(item["coord"]).astype(int)
                y, x = np.clip(y, 0, height - 1), np.clip(x, 0, width - 1)
                flat_idx = y * width + x
                kind_id = 0 if kind == "branch" else 1
                area = item.get("area", 1)
                length = item.get("length", 1)
                stats = flat_features.new_tensor((y / height, x / width,
                                                  math.log1p(area) / 12,
                                                  math.log1p(length) / 8,
                                                  item["confidence"],
                                                  float(kind_id)))
                type_vector = self.type_embedding(torch.tensor(kind_id, device=feature.device))
                token_inputs.append(torch.cat((flat_features[:, flat_idx], stats, type_vector)))
                coords.append((y / height, x / width))
            zero = preliminary_logits[image_idx, 0].new_zeros(height * width)
            add_sum, add_count, remove_sum, remove_count = zero, zero, zero, zero
            branch_logits, branch_pixels = [], []
            path_logits, path_pixels, path_centerlines, path_endpoints = [], [], [], []
            if selected:
                tokens = self._attend(self.token_projection(torch.stack(token_inputs)),
                                      flat_features.new_tensor(coords))
                selected_endpoints = []
                for token_idx, (_, kind, idx) in enumerate(selected):
                    if kind == "branch":
                        pixels = torch.as_tensor(branches[idx]["pixels"], device=feature.device)
                        score = self.validity_head(tokens[token_idx]).squeeze(-1)
                        branch_logits.append(score)
                        branch_pixels.append(pixels)
                        remove_sum = remove_sum.index_add(0, pixels,
                                                          (1 - score.sigmoid()).expand(pixels.numel()))
                        remove_count = remove_count.index_add(0, pixels, torch.ones_like(pixels, dtype=zero.dtype))
                    else:
                        selected_endpoints.append((token_idx, endpoints[idx]))
                pair_candidates = {}
                for first, (token_i, end_i) in enumerate(selected_endpoints):
                    nearby = []
                    for second, (token_j, end_j) in enumerate(selected_endpoints):
                        if second == first or end_i["branch"] == end_j["branch"]:
                            continue
                        distance = float(np.linalg.norm(end_i["coord"] - end_j["coord"]))
                        if 1 < distance <= self.radius:
                            nearby.append((distance, second))
                    for distance, second in sorted(nearby)[:3]:
                        pair = tuple(sorted((first, second)))
                        pair_candidates[pair] = min(distance, pair_candidates.get(pair, float("inf")))
                ordered_pairs = sorted((distance, *pair) for pair, distance in pair_candidates.items())
                for _, first, second in ordered_pairs:
                    token_i, end_i = selected_endpoints[first]
                    token_j, end_j = selected_endpoints[second]
                    for bend in (0, -1, 1):
                        if len(path_logits) >= self.max_paths:
                            break
                        band, centerline = _path_pixels(end_i["coord"], end_j["coord"],
                                                        bend, height, width)
                        if not band.size:
                            continue
                        band_tensor = torch.as_tensor(band, device=feature.device)
                        center_tensor = torch.as_tensor(centerline, device=feature.device)
                        # Sample along the proposed centerline, not only at endpoints.
                        sample = center_tensor[torch.linspace(0, center_tensor.numel() - 1,
                                                               min(8, center_tensor.numel()),
                                                               device=feature.device).long()]
                        path_feature = flat_features[:, sample].mean(dim=1)
                        path_prob = flat_prob[sample].mean().unsqueeze(0)
                        displacement = ((end_j["coord"] - end_i["coord"])
                                        / np.asarray([height, width]))
                        offset = flat_features.new_tensor((float(displacement[0]),
                                                           float(displacement[1]),
                                                           float(bend),
                                                           len(centerline) / max(height, width)))
                        pair_input = torch.cat((tokens[token_i], tokens[token_j],
                                                path_feature, path_prob, offset))
                        score = self.path_head(pair_input).squeeze(-1)
                        path_logits.append(score)
                        path_pixels.append(band_tensor)
                        path_centerlines.append(center_tensor)
                        endpoint_idx = np.rint([end_i["coord"], end_j["coord"]]).astype(int)
                        endpoint_idx[:, 0] = np.clip(endpoint_idx[:, 0], 0, height - 1)
                        endpoint_idx[:, 1] = np.clip(endpoint_idx[:, 1], 0, width - 1)
                        path_endpoints.append(torch.as_tensor(endpoint_idx[:, 0] * width + endpoint_idx[:, 1],
                                                              device=feature.device))
                        add_sum = add_sum.index_add(0, band_tensor, score.sigmoid().expand(band_tensor.numel()))
                        add_count = add_count.index_add(0, band_tensor,
                                                        torch.ones_like(band_tensor, dtype=zero.dtype))
                    if len(path_logits) >= self.max_paths:
                        break
            add_maps.append((add_sum / add_count.clamp_min(1)).view(1, height, width))
            remove_maps.append((remove_sum / remove_count.clamp_min(1)).view(1, height, width))
            records.append({"branch_logits": branch_logits, "branch_pixels": branch_pixels,
                            "path_logits": path_logits, "path_pixels": path_pixels,
                            "path_centerlines": path_centerlines,
                            "path_endpoints": path_endpoints})
        add_map = torch.stack(add_maps)
        remove_map = torch.stack(remove_maps)
        final_logits = preliminary_logits + self.beta_add * add_map - self.beta_remove * remove_map
        return final_logits, {"stage": "fragment_path_topology", "records": records,
                              "add_map": add_map, "remove_map": remove_map,
                              "beta_add": self.beta_add, "beta_remove": self.beta_remove}
