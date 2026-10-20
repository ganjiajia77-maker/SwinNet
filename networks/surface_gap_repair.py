"""Local, candidate-conditioned surface-logit repair for the e202 Mass baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from collections import deque

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.spatial import cKDTree


PROPOSAL_SEARCH_BUDGET={
    "max_thinning_radius":64,
    "max_endpoint_sources":512,
    "max_endpoint_pair_checks":16,
    "max_road_samples":1024,
    "max_road_bins_checked":32,
}


@dataclass(frozen=True)
class GapCandidate:
    kind: str                    # endpoint_endpoint | endpoint_road
    anchor_a: tuple[int, int]    # native (y, x)
    anchor_b: tuple[int, int]
    tangent_a: tuple[float, float]
    tangent_a_valid: bool
    tangent_b: tuple[float, float]
    tangent_b_valid: bool
    distance: float
    line_alignment_a: float
    line_alignment_b: float
    tile_top: int
    tile_left: int
    label: int = -1             # 1 positive, 0 negative, -1 ignore (train only)
    gt_coverage: float = float("nan")
    predicted_gap: bool = False

    def to_dict(self):
        return asdict(self)


def sliding_positions(size: int, tile: int = 512, stride: int = 256) -> list[int]:
    if size <= tile:
        return [0]
    positions = list(range(0, size - tile + 1, stride))
    if positions[-1] != size - tile:
        positions.append(size - tile)
    return positions


def tile_grid(height: int, width: int, tile=512, stride=256):
    return [(y, x) for y in sliding_positions(height, tile, stride)
            for x in sliding_positions(width, tile, stride)]


def taper_weight(size=512, floor=0.1, device=None, dtype=torch.float32):
    axis = torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype)
    one = (1.0 - axis.abs()).clamp_min(floor)
    return torch.outer(one, one)[None, None]


def stitch_logits(tile_logits, positions, image_hw, tile=512, stride=256):
    h, w = image_hw
    if not tile_logits:
        raise ValueError("No tile logits to stitch")
    weight = taper_weight(tile, device=tile_logits[0].device)
    total = torch.zeros((1, 1, h, w), device=weight.device)
    divisor = torch.zeros_like(total)
    for logits, (top, left) in zip(tile_logits, positions):
        hh, ww = min(tile, h - top), min(tile, w - left)
        total[..., top:top+hh, left:left+ww] += logits[..., :hh, :ww] * weight[..., :hh, :ww]
        divisor[..., top:top+hh, left:left+ww] += weight[..., :hh, :ww]
    return total / divisor.clamp_min(1e-8)


def zhang_suen(binary: np.ndarray) -> np.ndarray:
    """Fixed 8-neighbour Zhang-Suen thinning, returning bool pixels."""
    image = (binary > 0).astype(np.uint8)
    while True:
        changed = False
        for step in (0, 1):
            p = np.pad(image, 1)
            p2, p3, p4 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:]
            p5, p6, p7 = p[2:, 2:], p[2:, 1:-1], p[2:, :-2]
            p8, p9 = p[1:-1, :-2], p[:-2, :-2]
            ns = (p2,p3,p4,p5,p6,p7,p8,p9)
            count = sum(ns)
            transitions = sum(((ns[i] == 0) & (ns[(i+1)%8] == 1)).astype(np.uint8) for i in range(8))
            common = (image == 1) & (count >= 2) & (count <= 6) & (transitions == 1)
            if step == 0:
                remove = common & ((p2*p4*p6) == 0) & ((p4*p6*p8) == 0)
            else:
                remove = common & ((p2*p4*p8) == 0) & ((p2*p6*p8) == 0)
            if remove.any():
                image[remove] = 0
                changed = True
        if not changed:
            return image.astype(bool)


_NEIGHBORS = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))


def _neighbors(skel, point):
    y, x = point
    h, w = skel.shape
    return [(y+dy,x+dx) for dy,dx in _NEIGHBORS
            if 0 <= y+dy < h and 0 <= x+dx < w and skel[y+dy,x+dx]]


def _trace_tangent(skel, endpoint, length=16):
    path = [endpoint]
    previous = None
    current = endpoint
    for _ in range(length):
        options = [p for p in _neighbors(skel, current) if p != previous]
        if not options:
            break
        # Deterministic path trace; a branch terminates the tangent estimate.
        if len(options) > 1:
            break
        nxt = options[0]
        path.append(nxt)
        previous, current = current, nxt
    if len(path) < 3:
        return (0.0, 0.0), False
    dy, dx = endpoint[0] - path[-1][0], endpoint[1] - path[-1][1]
    norm = math.hypot(dy, dx)
    return (dy/norm, dx/norm), norm >= max(4, length * 0.5)


def _alignment(tangent, valid, vector):
    if not valid:
        return 0.0
    norm = math.hypot(*vector)
    return float((tangent[0]*vector[0] + tangent[1]*vector[1]) / max(norm, 1e-8))


def _candidate_tile(a, b, positions, tile=512, context=16):
    best, best_margin = None, -1
    for top, left in positions:
        ay, ax = a; by, bx = b
        if not (top <= ay < top+tile and left <= ax < left+tile and
                top <= by < top+tile and left <= bx < left+tile):
            continue
        margin = min(ay-top, ax-left, top+tile-1-ay, left+tile-1-ax,
                     by-top, bx-left, top+tile-1-by, left+tile-1-bx)
        if margin > best_margin:
            best, best_margin = (top, left), margin
    return best if best is not None and best_margin >= context else None


def _locally_connected(binary, a, b, padding=8):
    h,w=binary.shape
    y0=max(0,min(a[0],b[0])-padding); y1=min(h,max(a[0],b[0])+padding+1)
    x0=max(0,min(a[1],b[1])-padding); x1=min(w,max(a[1],b[1])+padding+1)
    roi=binary[y0:y1,x0:x1].astype(np.uint8)
    _,labels=cv2.connectedComponents(roi,connectivity=8)
    return labels[a[0]-y0,a[1]-x0] != 0 and labels[a[0]-y0,a[1]-x0] == labels[b[0]-y0,b[1]-x0]


def _line_samples(a, b, n=None):
    ay, ax = a; by, bx = b
    n = n or max(abs(by-ay), abs(bx-ax), 1) + 1
    yy = np.rint(np.linspace(ay, by, n)).astype(int)
    xx = np.rint(np.linspace(ax, bx, n)).astype(int)
    return yy, xx


def _local_gt_path_length(gt_band, anchor_a, anchor_b, source, target, max_detour):
    """Shortest 8-connected path inside a bounded local GT ROI; infinity if absent."""
    h,w=gt_band.shape
    distance=math.dist(anchor_a,anchor_b)
    pad=int(min(32,max(12,round(distance*.25))))
    y0=max(0,min(anchor_a[0],anchor_b[0])-pad); y1=min(h,max(anchor_a[0],anchor_b[0])+pad+1)
    x0=max(0,min(anchor_a[1],anchor_b[1])-pad); x1=min(w,max(anchor_a[1],anchor_b[1])+pad+1)
    road=gt_band[y0:y1,x0:x1]
    sy,sx=source[0]-y0,source[1]-x0; ty,tx=target[0]-y0,target[1]-x0
    if not (0<=sy<road.shape[0] and 0<=sx<road.shape[1] and 0<=ty<road.shape[0] and 0<=tx<road.shape[1]):
        return float("inf")
    if not road[sy,sx] or not road[ty,tx]: return float("inf")
    _,components=cv2.connectedComponents(road.astype(np.uint8),connectivity=8)
    if components[sy,sx]==0 or components[sy,sx]!=components[ty,tx]: return float("inf")
    steps=np.full(road.shape,-1,np.int32); steps[sy,sx]=0
    queue=deque([(sy,sx)])
    moves=((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))
    max_steps=int(max_detour)
    while queue:
        y,x=queue.popleft(); cost=steps[y,x]
        if (y,x)==(ty,tx): return float(cost)
        if cost>=max_steps: continue
        for dy,dx in moves:
            ny,nx=y+dy,x+dx
            if 0<=ny<road.shape[0] and 0<=nx<road.shape[1] and road[ny,nx] and steps[ny,nx]<0:
                steps[ny,nx]=cost+1; queue.append((ny,nx))
    return float("inf")


def generate_candidates(probability: np.ndarray, threshold=0.45, min_distance=8,
                        max_distance=128, max_per_endpoint=8, max_total=256,
                        tangent_length=16, boundary_margin=8, tile=512, stride=256,
                        progress=None,
                        max_thinning_radius=PROPOSAL_SEARCH_BUDGET["max_thinning_radius"],
                        max_endpoint_sources=PROPOSAL_SEARCH_BUDGET["max_endpoint_sources"],
                        max_endpoint_pair_checks=PROPOSAL_SEARCH_BUDGET["max_endpoint_pair_checks"],
                        max_road_samples=PROPOSAL_SEARCH_BUDGET["max_road_samples"],
                        max_road_bins_checked=PROPOSAL_SEARCH_BUDGET["max_road_bins_checked"]):
    """Generate deterministic candidates from prediction only (no GT argument)."""
    binary = (probability >= threshold).astype(np.uint8)
    road_fraction=float(binary.mean())
    if binary.any():
        max_radius=float(cv2.distanceTransform(binary,cv2.DIST_L2,3).max())
    else:
        max_radius=0.0
    if max_radius>max_thinning_radius:
        if progress: progress(f"candidate mask too thick for bounded thinning (radius={max_radius:.1f}); skip this image")
        return [],{"endpoint_count":0,"endpoint_count_raw":0,"endpoint_budget_skipped":0,
                   "proposed":0,"kept":0,"boundary_filtered":0,"tile_skipped":0,
                   "road_fraction":road_fraction,"max_mask_radius":max_radius,
                   "skip_reason":"thick_initial_candidate_mask"}
    if progress: progress("skeletonizing native prediction")
    skel = zhang_suen(binary)
    points = np.argwhere(skel)
    if not len(points):
        return [], {"endpoint_count":0,"endpoint_count_raw":0,"endpoint_budget_skipped":0,
                    "proposed":0,"kept":0,"boundary_filtered":0,"tile_skipped":0,
                    "road_fraction":road_fraction,"max_mask_radius":max_radius,
                    "skip_reason":"empty_candidate_skeleton"}
    kernel=np.ones((3,3),np.uint8); kernel[1,1]=0
    neighbor_counts=cv2.filter2D(skel.astype(np.uint8),-1,kernel,borderType=cv2.BORDER_CONSTANT)
    endpoints=[tuple(map(int,p)) for p in np.argwhere(skel & (neighbor_counts==1))]
    h,w = probability.shape
    boundary = [p for p in endpoints if min(p[0],p[1],h-1-p[0],w-1-p[1]) < boundary_margin]
    boundary_set=set(boundary)
    endpoints = [p for p in endpoints if p not in boundary_set]
    all_endpoint_set=set(endpoints)
    raw_endpoint_count=len(endpoints)
    if len(endpoints)>max_endpoint_sources:
        ranked=sorted(endpoints,key=lambda p:(-float(probability[p]),p[0],p[1]))
        selected=[]; selected_set=set(); cells=set()
        for p in ranked:
            cell=(p[0]//64,p[1]//64)
            if cell not in cells:
                cells.add(cell); selected.append(p); selected_set.add(p)
                if len(selected)>=max_endpoint_sources: break
        if len(selected)<max_endpoint_sources:
            for p in ranked:
                if p not in selected_set:
                    selected.append(p); selected_set.add(p)
                    if len(selected)>=max_endpoint_sources: break
        endpoints=sorted(selected)
    if progress: progress(f"skeleton endpoints={raw_endpoint_count}; searched={len(endpoints)}")
    tangents = {p: _trace_tangent(skel,p,tangent_length) for p in endpoints}
    positions = tile_grid(h,w,tile,stride)
    proposals = []
    pair_search_truncated=0
    if endpoints:
        endpoint_xy=np.asarray(endpoints,dtype=np.float64)
        endpoint_tree=cKDTree(endpoint_xy)
        for i,a in enumerate(endpoints):
            # Rank nearby pairs before expensive local connectivity and tile checks.
            nearby_pairs=[]
            for j in endpoint_tree.query_ball_point(endpoint_xy[i],max_distance):
                if j<=i: continue
                b=endpoints[j]
                dy,dx=b[0]-a[0],b[1]-a[1]; distance=math.hypot(dy,dx)
                if not min_distance <= distance <= max_distance: continue
                ta,va=tangents[a]; tb,vb=tangents[b]
                align_a=_alignment(ta,va,(dy,dx)); align_b=_alignment(tb,vb,(-dy,-dx))
                if (va and align_a < math.cos(math.radians(75))) or (vb and align_b < math.cos(math.radians(75))):
                    continue
                compatibility=min([v for v,valid in ((align_a,va),(align_b,vb)) if valid],default=0.0)
                nearby_pairs.append(((-compatibility,distance,j),j,distance,align_a,align_b))
            nearby_pairs.sort(key=lambda item:item[0])
            pair_search_truncated+=max(0,len(nearby_pairs)-max_endpoint_pair_checks)
            for _,j,distance,align_a,align_b in nearby_pairs[:max_endpoint_pair_checks]:
                b=endpoints[j]; ta,va=tangents[a]; tb,vb=tangents[b]
                if _locally_connected(binary,a,b): continue
                tile_pos=_candidate_tile(a,b,positions,tile)
                if tile_pos is None: continue
                proposals.append(GapCandidate("endpoint_endpoint",a,b,ta,va,tb,vb,distance,
                                              align_a,align_b,*tile_pos))
    if progress: progress(f"endpoint pairs proposed={len(proposals)}")
    tree=cKDTree(points.astype(np.float32))
    endpoint_set=all_endpoint_set
    used_counts={}
    for candidate in proposals:
        used_counts[candidate.anchor_a]=used_counts.get(candidate.anchor_a,0)+1
    tile_skipped=0; road_proposals=0; road_samples_truncated=0; road_bins_truncated=0
    for endpoint_index,a in enumerate(endpoints,1):
        ta,va=tangents[a]
        options={}
        nearby=np.asarray(tree.query_ball_point(np.asarray(a),max_distance),dtype=np.int64)
        if len(nearby)>max_road_samples:
            nearby_points=points[nearby]
            vectors=nearby_points-np.asarray(a)
            distances=np.hypot(vectors[:,0],vectors[:,1])
            alignments=(vectors[:,0]*ta[0]+vectors[:,1]*ta[1])/np.maximum(distances,1.0) if va else np.zeros_like(distances)
            order=np.lexsort((nearby_points[:,1],nearby_points[:,0],distances,-alignments))
            road_samples_truncated+=len(nearby)-max_road_samples
            nearby=nearby[order[:max_road_samples]]
        for ix in nearby:
            b=tuple(map(int,points[ix])); distance=math.dist(a,b)
            if b==a or distance < min_distance or distance > max_distance or b in endpoint_set:
                continue
            align_a=_alignment(ta,va,(b[0]-a[0],b[1]-a[1]))
            if va and align_a < math.cos(math.radians(75)): continue
            key=(b[0]//8,b[1]//8)
            rank=(-align_a,distance,b)
            choices=options.setdefault(key,[])
            choices.append((rank,b,distance,align_a))
            choices.sort(key=lambda item:item[0])
            del choices[2:]
        ordered_bins=sorted(options.values(),key=lambda choices:choices[0][0])
        road_bins_truncated+=max(0,len(ordered_bins)-max_road_bins_checked)
        for choices in ordered_bins[:max_road_bins_checked]:
            if used_counts.get(a,0)>=max_per_endpoint: break
            for _,b,distance,align_a in choices:
                tile_pos=_candidate_tile(a,b,positions,tile)
                if tile_pos is None:
                    tile_skipped+=1; continue
                road_proposals+=1
                if _locally_connected(binary,a,b): continue
                proposals.append(GapCandidate("endpoint_road",a,b,ta,va,(0.0,0.0),False,
                                              distance,align_a,0.0,*tile_pos))
                used_counts[a]=used_counts.get(a,0)+1
                break
        if progress and (endpoint_index%32==0 or endpoint_index==len(endpoints)):
            progress(f"endpoint-road search {endpoint_index}/{len(endpoints)}")
    def proposal_rank(c):
        alignments=[]
        if c.tangent_a_valid: alignments.append(c.line_alignment_a)
        if c.kind=="endpoint_endpoint" and c.tangent_b_valid: alignments.append(c.line_alignment_b)
        # Prefer geometrically compatible, nearby proposals before applying the image cap.
        compatibility=min(alignments) if alignments else 0.0
        return (-compatibility,c.distance,c.kind,c.anchor_a,c.anchor_b)
    proposals.sort(key=proposal_rank)
    # Nearby samples along the same target road are grouped deterministically.
    dedup=[]; seen=set()
    for c in proposals:
        key=(c.kind,c.anchor_a,(c.anchor_b[0]//8,c.anchor_b[1]//8)) if c.kind=="endpoint_road" else (c.kind,*sorted((c.anchor_a,c.anchor_b)))
        if key in seen: continue
        seen.add(key); dedup.append(c)
    per_endpoint={}; kept=[]
    for c in dedup:
        if per_endpoint.get(c.anchor_a,0) >= max_per_endpoint: continue
        if c.kind=="endpoint_endpoint" and per_endpoint.get(c.anchor_b,0) >= max_per_endpoint: continue
        per_endpoint[c.anchor_a]=per_endpoint.get(c.anchor_a,0)+1
        if c.kind=="endpoint_endpoint":
            per_endpoint[c.anchor_b]=per_endpoint.get(c.anchor_b,0)+1
        kept.append(c)
    pre_cap=len(kept)
    kept=kept[:max_total]
    return kept,{"endpoint_count":len(endpoints),"proposed":len(proposals),"deduplicated":len(dedup),
                 "road_geometric_proposals":road_proposals,"kept":len(kept),
                 "truncated_by_image_cap":max(0,pre_cap-len(kept)),
                 "boundary_filtered":len(boundary),"tile_skipped":tile_skipped,
                 "endpoint_count_raw":raw_endpoint_count,
                 "endpoint_budget_skipped":raw_endpoint_count-len(endpoints),
                 "pair_checks_truncated":pair_search_truncated,
                 "road_samples_truncated":road_samples_truncated,
                 "road_bins_truncated":road_bins_truncated,
                 "road_fraction":road_fraction,"max_mask_radius":max_radius}


def label_candidate(candidate: GapCandidate, gt_mask: np.ndarray, base_probability: np.ndarray,
                    anchor_tolerance=3, corridor_radius=4, gt_skeleton=None,
                    gt_tree=None, gt_band=None, base_corridor=None):
    """Local GT path label; uncertainty is ignored instead of guessed."""
    h,w=gt_mask.shape
    if gt_skeleton is None: gt_skeleton=zhang_suen(gt_mask)
    ys,xs=np.nonzero(gt_skeleton)
    if not len(ys): return candidate.__class__(**{**candidate.to_dict(),"label":0,"gt_coverage":0.0})
    tree=gt_tree if gt_tree is not None else cKDTree(np.stack([ys,xs],axis=1))
    da,ia=tree.query(candidate.anchor_a); db,ib=tree.query(candidate.anchor_b)
    yy,xx=_line_samples(candidate.anchor_a,candidate.anchor_b)
    support=gt_band if gt_band is not None else cv2.dilate(gt_mask.astype(np.uint8),np.ones((2*corridor_radius+1,)*2,np.uint8))>0
    coverage=float(support[yy,xx].mean())
    source=tuple(map(int,tree.data[int(ia)]))
    target=tuple(map(int,tree.data[int(ib)]))
    local_path_length=_local_gt_path_length(support,candidate.anchor_a,candidate.anchor_b,
        source,target,max_detour=2.0*candidate.distance+16.0)
    local_path=math.isfinite(local_path_length)
    # The base must be locally disconnected at the proposed bridge.
    pred_corr=base_corridor if base_corridor is not None else cv2.dilate((base_probability>=0.45).astype(np.uint8),np.ones((3,3),np.uint8))
    base_path_coverage=float((pred_corr[yy,xx]>0).mean())
    predicted_gap=base_path_coverage < 0.9
    anchors_on_gt=da <= anchor_tolerance and db <= anchor_tolerance
    if anchors_on_gt and local_path and predicted_gap:
        label=1
    elif not anchors_on_gt or (not local_path and coverage <= 0.5):
        label=0
    else:
        label=-1
    return candidate.__class__(**{**candidate.to_dict(),"label":label,"gt_coverage":coverage,
                                  "predicted_gap":predicted_gap})


class _ResidualDilated(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        groups=8 if channels%8==0 else 1
        self.net=nn.Sequential(nn.Conv2d(channels,channels,3,padding=dilation,dilation=dilation,bias=False),
                               nn.GroupNorm(groups,channels),nn.GELU(),
                               nn.Conv2d(channels,channels,3,padding=1,bias=False),
                               nn.GroupNorm(groups,channels))
    def forward(self,x): return F.gelu(x+self.net(x))


class SurfaceGapRepair(nn.Module):
    """Candidate local classifier plus a bounded, signed 512px logit delta."""
    def __init__(self, shallow_channels=96, decoder_channels=96, token_dim=64):
        super().__init__()
        self.shallow_proj=nn.Sequential(nn.Conv2d(shallow_channels,16,1,bias=False),nn.GroupNorm(4,16),nn.GELU())
        self.decoder_proj=nn.Sequential(nn.Conv2d(decoder_channels,16,1,bias=False),nn.GroupNorm(4,16),nn.GELU())
        self.detail256=nn.Sequential(nn.Conv2d(10,32,3,stride=2,padding=1,bias=False),nn.GroupNorm(8,32),nn.GELU())
        self.detail128=nn.Sequential(nn.Conv2d(32,32,3,stride=2,padding=1,bias=False),nn.GroupNorm(8,32),nn.GELU())
        self.fuse=nn.Sequential(nn.Conv2d(64,64,1,bias=False),nn.GroupNorm(8,64),nn.GELU())
        self.context=nn.Sequential(_ResidualDilated(64,1),_ResidualDilated(64,2),_ResidualDilated(64,4))
        self.geometry=nn.Sequential(nn.Linear(8,token_dim),nn.GELU(),nn.Linear(token_dim,token_dim))
        # Global context (64), two anchor descriptors (2 x 64), geometry token.
        self.classifier=nn.Sequential(nn.Linear(64*3+token_dim,64),nn.GELU(),nn.Linear(64,1))
        self.delta256=nn.Sequential(nn.Conv2d(64+32,48,3,padding=1,bias=False),nn.GroupNorm(8,48),nn.GELU())
        self.delta512=nn.Sequential(nn.Conv2d(48+10,16,1,bias=False),nn.GroupNorm(4,16),nn.GELU(),nn.Conv2d(16,1,3,padding=1))
        nn.init.zeros_(self.delta512[-1].weight); nn.init.zeros_(self.delta512[-1].bias)

    @staticmethod
    def _anchor_pool(features, prompts):
        pooled=[]
        for idx in range(2):
            weight=F.interpolate(prompts[:,idx:idx+1],size=features.shape[-2:],mode="bilinear",align_corners=False)
            pooled.append((features*weight).sum((2,3))/weight.sum((2,3)).clamp_min(1e-6))
        return torch.cat(pooled,dim=1)

    def forward(self,rgb,p0,prompts,geometry,shallow,decoder):
        if geometry.ndim != 2 or geometry.shape[1] != 8:
            raise ValueError(
                "geometry must have shape [batch, 8]; "
                f"received {tuple(geometry.shape)}"
            )
        if rgb.ndim != 4 or p0.ndim != 4 or prompts.ndim != 4:
            raise ValueError("rgb, p0, and prompts must be four-dimensional NCHW tensors")
        if not (
            geometry.shape[0] == rgb.shape[0]
            and p0.shape[0] == rgb.shape[0]
            and prompts.shape[0] == rgb.shape[0]
        ):
            raise ValueError("rgb, p0, prompts, and geometry batch sizes must match")
        if p0.shape[-2:] != rgb.shape[-2:] or prompts.shape[-2:] != rgb.shape[-2:]:
            raise ValueError("p0 and prompts must have the same spatial size as rgb")
        # RGB + P0 + two independent Gaussian anchor maps + four geometry planes.
        gmap=geometry[:,:,None,None].expand(-1,-1,rgb.shape[-2],rgb.shape[-1])
        full=torch.cat([rgb,p0,prompts,gmap[:,:4]],dim=1)
        detail=self.detail256(full)
        detail128=self.detail128(detail)
        shallow=self.shallow_proj(shallow)
        decoder=F.interpolate(decoder,size=detail128.shape[-2:],mode="bilinear",align_corners=False)
        decoder=self.decoder_proj(decoder)
        if shallow.shape[-2:] != detail128.shape[-2:]:
            shallow=F.interpolate(shallow,size=detail128.shape[-2:],mode="bilinear",align_corners=False)
        context=self.context(self.fuse(torch.cat([detail128,shallow,decoder],dim=1)))
        geom_vec=geometry.flatten(1)
        geo=self.geometry(geom_vec)
        pooled=F.adaptive_avg_pool2d(context,1).flatten(1)
        anchor=self._anchor_pool(context,prompts)
        classifier_input=torch.cat([pooled,anchor,geo],dim=1)
        if classifier_input.shape[1] != self.classifier[0].in_features:
            raise RuntimeError(
                "repair classifier feature width mismatch: "
                f"got {classifier_input.shape[1]}, expected {self.classifier[0].in_features}"
            )
        cls=self.classifier(classifier_input).squeeze(1)
        up=F.interpolate(context,size=(256,256),mode="bilinear",align_corners=False)
        detail_up=F.interpolate(detail,size=(256,256),mode="bilinear",align_corners=False)
        d=self.delta256(torch.cat([up,detail_up],dim=1))
        d=F.interpolate(d,size=(512,512),mode="bilinear",align_corners=False)
        d=self.delta512(torch.cat([d,full],dim=1))
        return cls,4.0*torch.tanh(d)


def gaussian_prompt(size, point, sigma=3.0, device=None):
    y=torch.arange(size,device=device,dtype=torch.float32)
    x=torch.arange(size,device=device,dtype=torch.float32)
    yy,xx=torch.meshgrid(y,x,indexing="ij")
    return torch.exp(-((yy-point[0])**2+(xx-point[1])**2)/(2*sigma*sigma))


def candidate_tensors(candidate, device, size=512):
    a=(candidate.anchor_a[0]-candidate.tile_top,candidate.anchor_a[1]-candidate.tile_left)
    b=(candidate.anchor_b[0]-candidate.tile_top,candidate.anchor_b[1]-candidate.tile_left)
    prompts=torch.stack([gaussian_prompt(size,a,device=device),gaussian_prompt(size,b,device=device)])[None]
    kind=float(candidate.kind=="endpoint_road")
    dy=(candidate.anchor_b[0]-candidate.anchor_a[0])/128.0
    dx=(candidate.anchor_b[1]-candidate.anchor_a[1])/128.0
    geom=torch.tensor([dy,dx,candidate.distance/128.0,candidate.tangent_a_valid,
                       candidate.tangent_b_valid,candidate.line_alignment_a,
                       candidate.line_alignment_b,kind],device=device,dtype=torch.float32)[None]
    return prompts,geom


def merge_candidate_deltas(base_logit, deltas, probabilities, windows, accepted=None):
    """Order-independent weighted mean of candidate corrections."""
    numerator=torch.zeros_like(base_logit); denominator=torch.zeros_like(base_logit)
    for i,(delta,p,window) in enumerate(zip(deltas,probabilities,windows)):
        if accepted is not None and not bool(accepted[i]): continue
        weight=window*p.reshape(-1,1,1,1)
        numerator=numerator+weight*delta
        denominator=denominator+window
    return base_logit+numerator/denominator.clamp_min(1.0)
