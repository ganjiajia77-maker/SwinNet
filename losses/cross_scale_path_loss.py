"""Path-search primitives for the proposed cross-scale supervision experiment.

This module is deliberately not wired into training yet: path-pair sampling,
label ambiguity and equal-budget conflict weighting must be validated first.
"""
import heapq
import math

import numpy as np
import torch
import torch.nn.functional as F

from topology_direction_constants import CONNECTIVITY_DIRECTIONS, CONNECTIVITY_OPPOSITE


def widest_path(vertex_scores, start, end, *, allowed=None, edge_scores=None, edge_valid=None):
    """Return an eight-neighbour maximum-bottleneck path from detached scores.

    For an edge graph, scores for the two opposite directed channels are
    conservatively combined with min; both directions must be valid. `allowed`
    limits vertices, and caller is responsible for defining the GT-safe ROI.
    """
    scores=np.asarray(vertex_scores,dtype=np.float64)
    if scores.ndim!=2 or not np.isfinite(scores).all():raise ValueError('Expected finite 2-D scores')
    h,w=scores.shape
    allowed=np.ones((h,w),bool) if allowed is None else np.asarray(allowed,dtype=bool)
    if allowed.shape!=(h,w):raise ValueError('Invalid vertex mask')
    if edge_scores is not None:
        edge_scores=np.asarray(edge_scores,dtype=np.float64)
        if edge_scores.shape!=(8,h,w) or not np.isfinite(edge_scores).all():raise ValueError('Invalid edges')
        edge_valid=np.ones((8,h,w),bool) if edge_valid is None else np.asarray(edge_valid,dtype=bool)
        if edge_valid.shape!=(8,h,w):raise ValueError('Invalid edge mask')
    start=tuple(map(int,start));end=tuple(map(int,end))
    for y,x in (start,end):
        if not (0<=y<h and 0<=x<w and allowed[y,x]):return None
    best=np.full((h,w),-math.inf);best[start]=scores[start] if edge_scores is None else 1.
    previous={};queue=[(-best[start],start)]
    while queue:
        neg,p=heapq.heappop(queue);value=-neg
        if value<best[p]:continue
        if p==end:
            path=[p]
            while p!=start:p=previous[p];path.append(p)
            return list(reversed(path))
        y,x=p
        for direction,(dy,dx) in enumerate(CONNECTIVITY_DIRECTIONS):
            yy,xx=y+dy,x+dx
            if not (0<=yy<h and 0<=xx<w and allowed[yy,xx]):continue
            if edge_scores is None: quality=scores[yy,xx]
            else:
                opposite=CONNECTIVITY_OPPOSITE[direction]
                if not (edge_valid[direction,y,x] and edge_valid[opposite,yy,xx]):continue
                quality=min(edge_scores[direction,y,x],edge_scores[opposite,yy,xx])
            candidate=min(value,quality)
            if candidate>best[yy,xx]:
                best[yy,xx]=candidate;previous[(yy,xx)]=p
                heapq.heappush(queue,(-candidate,(yy,xx)))
    return None


def selected_vertex_loss(logits, path, target, supervise):
    """BCE on selected path vertices, preserving gradients to original logits."""
    if logits.ndim!=2:raise ValueError('Expected 2-D logits')
    selected=[(y,x) for y,x in path if bool(supervise[y,x])]
    if not selected:return logits.sum()*0
    y,x=zip(*selected);values=logits[list(y),list(x)]
    return F.binary_cross_entropy_with_logits(values,torch.full_like(values,float(target)))


def selected_edge_loss(logits, path, target, supervise):
    """BCE on both directed logits of valid, supervised path edges."""
    if logits.ndim!=3 or logits.shape[0]!=8:raise ValueError('Expected 8-channel logits')
    lookup={tuple(offset):i for i,offset in enumerate(CONNECTIVITY_DIRECTIONS)}
    selected=[]
    for (y,x),(yy,xx) in zip(path,path[1:]):
        direction=lookup.get((yy-y,xx-x))
        if direction is None:raise ValueError('Non-neighbour path edge')
        opposite=CONNECTIVITY_OPPOSITE[direction]
        if bool(supervise[direction,y,x]) and bool(supervise[opposite,yy,xx]):
            selected.extend((logits[direction,y,x],logits[opposite,yy,xx]))
    if not selected:return logits.sum()*0
    values=torch.stack(selected)
    return F.binary_cross_entropy_with_logits(values,torch.full_like(values,float(target)))
