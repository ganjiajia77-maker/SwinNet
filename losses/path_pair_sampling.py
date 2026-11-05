"""Training-only local GT skeleton path-pair sampling.

The returned paths use native crop coordinates; no GT enters model forward.
"""
from collections import Counter
import heapq
import math
import cv2
import numpy as np
from topology_direction_constants import CONNECTIVITY_DIRECTIONS


def _walk(start, skeleton, min_length, max_length, rng):
    """Shortest skeleton graph path, never walking outside the crop."""
    h,w=skeleton.shape
    q=[(0.,start)]; parent={start:None}; distance={start:0.}; candidates=[]
    while q:
        d,(y,x)=heapq.heappop(q)
        if d>distance[(y,x)]:continue
        if min_length<=d<=max_length:candidates.append((y,x))
        if d>=max_length:continue
        for dy,dx in CONNECTIVITY_DIRECTIONS:
            n=(y+dy,x+dx)
            nd=d+math.hypot(dy,dx)
            if 0<=n[0]<h and 0<=n[1]<w and skeleton[n] and nd<=max_length and nd<distance.get(n,float('inf')):
                parent[n]=(y,x);distance[n]=nd;heapq.heappush(q,(nd,n))
    if not candidates:return None
    end=candidates[int(rng.integers(len(candidates)))];path=[]
    while end is not None:path.append(end);end=parent[end]
    return path[::-1]


def _roi(a,b,shape,margin,max_size):
    h,w=shape;y0=max(0,min(a[0],b[0])-margin);y1=min(h,max(a[0],b[0])+margin+1)
    x0=max(0,min(a[1],b[1])-margin);x1=min(w,max(a[1],b[1])+margin+1)
    return None if max(y1-y0,x1-x0)>max_size else (y0,y1,x0,x1)


def sample_pairs(road, skeleton, ambiguous, rng, *, max_pos=4, max_neg=4,
                 min_length=16,max_length=64,max_window=128,margin=16,attempts=64):
    road=np.asarray(road,bool); skeleton=np.asarray(skeleton,bool)
    if road.ndim!=2 or road.shape!=skeleton.shape or any(v%4 for v in road.shape):
        raise ValueError('Expected aligned native masks with stride-4 dimensions')
    if ambiguous.shape!=(road.shape[0]//4,road.shape[1]//4):raise ValueError('Invalid ambiguity grid')
    points=np.argwhere(skeleton & road);stats=Counter();positive=[];negative=[];seen=set()
    h,w=road.shape
    if not len(points):stats['no_skeleton']+=1;return positive,negative,stats
    for _ in range(attempts):
        if len(positive)>=max_pos and len(negative)>=max_neg:break
        y,x=map(int,points[int(rng.integers(len(points)))])
        if min(y,x,h-1-y,w-1-x)<margin:stats['crop_boundary']+=1;continue
        if len(positive)<max_pos:
            path=_walk((y,x),skeleton & road,min_length,max_length,rng)
            if path is None:stats['no_positive_path']+=1
            else:
                a,b=path[0],path[-1];box=_roi(a,b,road.shape,margin,max_window)
                if box is None:stats['positive_window']+=1
                elif box[0]==0 or box[1]==h or box[2]==0 or box[3]==w:stats['positive_crop_truncated']+=1
                elif a[0]//4==b[0]//4 and a[1]//4==b[1]//4:stats['same_cell']+=1
                elif any(ambiguous[py//4,px//4] for py,px in path):stats['positive_ambiguous']+=1
                elif not all(box[0]<=py<box[1] and box[2]<=px<box[3] for py,px in path):stats['positive_path_outside']+=1
                else:
                    key=tuple(sorted((a,b)))
                    if key not in seen:positive.append({'a':a,'b':b,'path':path,'roi':box});seen.add(key)
        if len(negative)<max_neg:
            diff=points-np.array((y,x));dist2=(diff*diff).sum(1)
            near=np.flatnonzero((dist2>=min_length**2)&(dist2<=max_length**2))
            if not len(near):stats['no_nearby_negative']+=1;continue
            b=tuple(map(int,points[int(near[int(rng.integers(len(near)))])]))
            a=(y,x);box=_roi(a,b,road.shape,margin,max_window)
            if box is None:stats['negative_window']+=1;continue
            if box[0]==0 or box[1]==h or box[2]==0 or box[3]==w:stats['negative_crop_truncated']+=1;continue
            if min(b[0],b[1],h-1-b[0],w-1-b[1])<margin:stats['crop_boundary']+=1;continue
            if (a[0]//4,a[1]//4)==(b[0]//4,b[1]//4):stats['same_cell']+=1;continue
            if ambiguous[a[0]//4,a[1]//4] or ambiguous[b[0]//4,b[1]//4]:stats['negative_ambiguous']+=1;continue
            y0,y1,x0,x1=box
            _,labels=cv2.connectedComponents(road[y0:y1,x0:x1].astype(np.uint8),connectivity=8)
            aa=labels[a[0]-y0,a[1]-x0];bb=labels[b[0]-y0,b[1]-x0]
            if not aa or not bb or aa==bb:stats['locally_connected']+=1;continue
            key=tuple(sorted((a,b)))
            if key not in seen:negative.append({'a':a,'b':b,'roi':box});seen.add(key)
    stats['positive_pairs']=len(positive);stats['negative_pairs']=len(negative)
    return positive,negative,stats
