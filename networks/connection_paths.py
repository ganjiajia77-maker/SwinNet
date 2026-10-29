"""Bounded prediction-only curved path proposals, with separate GT audit labels."""
from dataclasses import dataclass, asdict
import heapq
import math

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra

from evaluation.road_comparison.road_metrics import skeletonize, skeleton_graph
from networks.reliable_directional_repair import DIRECTIONS
from diagnostics.v92_connection_candidates import tangent, cap_pairs_by_length


@dataclass(frozen=True)
class PathConfig:
    thresholds: tuple = (.30, .45, .60)
    max_endpoints: int = 32
    max_pairs: int = 48
    max_paths: int = 128
    paths_per_pair: int = 2
    max_distance_work: float = 40.
    max_path_work: float = 80.
    search_margin_work: int = 6
    max_expansions: int = 12000
    min_width_work: float = 1.
    max_width_work: float = 3.
    corridor_tolerance_native: float = 2.
    audit_min_source_confidence: float = .05

    def validate(self):
        if not self.thresholds or any(not 0<t<1 for t in self.thresholds):
            raise ValueError('Candidate thresholds must lie in (0,1)')
        if min(self.max_endpoints,self.max_pairs,self.max_paths,self.paths_per_pair,self.max_expansions)<1:
            raise ValueError('Candidate budgets must be positive')
        if not 0<self.min_width_work<=self.max_width_work or self.max_distance_work<2 or self.max_path_work<self.max_distance_work:
            raise ValueError('Invalid width or path distance bounds')
        return self


def path_length(points, scale=(1.,1.)):
    diff=np.diff(np.asarray(points,dtype=float),axis=0)*np.asarray(scale)
    return float(np.linalg.norm(diff,axis=1).sum())


def soft_corridor(points, shape, width):
    """Continuous Gaussian support; truncated at two widths, never outside image."""
    seed=np.ones(shape,np.uint8)
    xy=np.asarray(points,dtype=int)
    if len(xy)==0: return np.zeros(shape,np.float32)
    xy[:,0]=xy[:,0].clip(0,shape[0]-1); xy[:,1]=xy[:,1].clip(0,shape[1]-1)
    seed[xy[:,0],xy[:,1]]=0
    distance=cv2.distanceTransform(seed,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    sigma=max(float(width)*.5,.5)
    return (np.exp(-.5*(distance/sigma)**2)*(distance<=max(float(width),1.))).astype(np.float32)


def _search(a,b,p0,q,w,semantic,config,penalty=None):
    """Symmetric A* cost from visual/road context and actual W channel order."""
    h,width=p0.shape; ay,ax=map(int,a); by,bx=map(int,b)
    margin=config.search_margin_work
    y0,y1=max(0,min(ay,by)-margin),min(h,max(ay,by)+margin+1)
    x0,x1=max(0,min(ax,bx)-margin),min(width,max(ax,bx)+margin+1)
    if semantic is not None:
        f=semantic; endpoint=.5*(f[:,ay,ax]+f[:,by,bx]); endpoint/=max(np.linalg.norm(endpoint),1e-6)
        norm=np.maximum(np.linalg.norm(f,axis=0),1e-6)
        semantic_cost=(1.-np.einsum('c,cyx->yx',endpoint,f)/norm).clip(0,2)*.2
    else: semantic_cost=np.zeros_like(p0)
    cost=1.+.7*(1.-p0)+.2*(1.-q)+semantic_cost
    if penalty is not None: cost=cost+penalty
    start=(ay,ax); end=(by,bx); previous={}; best={start:0.}
    queue=[(math.hypot(ay-by,ax-bx),0.,start)]; expanded=0
    while queue and expanded<config.max_expansions:
        _,g,current=heapq.heappop(queue)
        if g>best[current]+1e-8: continue
        expanded+=1
        if current==end:
            points=[current]
            while current!=start: current=previous[current]; points.append(current)
            points.reverse()
            return np.asarray(points,dtype=np.int32)
        cy,cx=current
        for d,(dy,dx) in enumerate(DIRECTIONS):
            ny,nx=cy+dy,cx+dx
            if not(y0<=ny<y1 and x0<=nx<x1): continue
            # Read-source convention: target p reads source p+delta[d].
            direction_support=.5*(w[d,cy,cx]+w[d^1,ny,nx])
            edge=math.hypot(dy,dx)*(.5*(cost[cy,cx]+cost[ny,nx])+.3*(1.-direction_support))
            nxt=(ny,nx); candidate=g+float(edge)
            if candidate+1e-8<best.get(nxt,float('inf')):
                best[nxt]=candidate; previous[nxt]=(cy,cx)
                heapq.heappush(queue,(candidate+math.hypot(ny-by,nx-bx),candidate,nxt))
    return None


def generate_paths(p0,r,w,q,semantic,native_shape,config=PathConfig()):
    """All inputs are predictions. Coordinates are native to this crop, no GT."""
    config.validate(); h,width=p0.shape
    if w.shape!=(8,h,width) or any(x.shape!=(h,width) for x in (r,q)):
        raise ValueError('Prediction map shapes or eight-direction channels differ')
    scale=(native_shape[0]/h,native_shape[1]/width)
    unique={}; counters={'pairs_considered':0,'search_failed':0,'overlong':0,'duplicate':0}
    searched=set()
    for threshold in config.thresholds:
        mask=p0>=threshold; sk=skeletonize(mask)
        degree=cv2.filter2D(sk.astype(np.uint8),-1,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT)-sk
        endpoints=np.argwhere(sk & (degree==1))
        # R ranks potential sources, with spatial quotas instead of a global hard cutoff.
        if len(endpoints):
            confidence=.5*(p0[endpoints[:,0],endpoints[:,1]]+r[endpoints[:,0],endpoints[:,1]])
            endpoints=endpoints[np.argsort(-confidence,kind='stable')]
        if len(endpoints)>config.max_endpoints:
            cells={}
            for point in endpoints:
                cell=(min(3,int(point[0])*4//h),min(3,int(point[1])*4//width))
                cells.setdefault(cell,[]).append(point)
            kept=[]; cursor=0
            while len(kept)<config.max_endpoints and any(cursor<len(items) for items in cells.values()):
                for cell in sorted(cells):
                    if cursor<len(cells[cell]): kept.append(cells[cell][cursor])
                    if len(kept)>=config.max_endpoints: break
                cursor+=1
            endpoints=np.asarray(kept)
        _,component,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
        tangents=[tangent(sk,point) for point in endpoints]
        distance=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,3)
        pairs=[]
        for i,a in enumerate(endpoints):
            for j in range(i+1,len(endpoints)):
                b=endpoints[j]; length=float(np.linalg.norm(a-b))
                if 2<=length<=config.max_distance_work:
                    pairs.append(('pair',i,j,a,b,length))
        pairs=cap_pairs_by_length(pairs,config.max_pairs)
        extensions=[]
        # Single-side predicted extensions supply hard negatives, not invented GT endpoints.
        for i,a in enumerate(endpoints):
            if tangents[i] is None: continue
            for length in (4.,8.,16.,32.):
                b=np.rint(a+tangents[i]*length).astype(int)
                if 0<=b[0]<h and 0<=b[1]<width and not mask[tuple(b)]:
                    extensions.append(('extension',i,None,a,b,length))
        # Spend the finite search budget across candidate types, lengths and thresholds.
        # Avoid generating hundreds of paths merely to discard them after expensive searches.
        ordered=[]
        for i in range(max(len(pairs),len(extensions))):
            if i<len(pairs): ordered.append(pairs[i])
            if i<len(extensions): ordered.append(extensions[i])
        before=len(unique); quota=max(1,config.max_paths//len(config.thresholds))
        for kind,i,j,a,b,_ in ordered:
            if len(unique)-before>=quota: break
            signature=(kind,tuple(a),tuple(b))
            if signature in searched: continue
            searched.add(signature)
            counters['pairs_considered']+=1
            penalty=np.zeros_like(p0)
            for alternative in range(config.paths_per_pair if kind=='pair' else 1):
                if len(unique)-before>=quota: break
                if np.linalg.norm(a-b)<4:
                    points=np.rint(np.linspace(a,b,int(np.ceil(np.linalg.norm(a-b)*2))+1)).astype(np.int32)
                    points=points[np.r_[True,np.any(np.diff(points,axis=0),axis=1)]]
                else: points=_search(a,b,p0,q,w,semantic,config,penalty)
                if points is None: counters['search_failed']+=1; break
                length=path_length(points)
                if length>config.max_path_work: counters['overlong']+=1; break
                # Alternate searches repel the interior, preserving the same sources.
                band=soft_corridor(points,p0.shape,2.)
                penalty=penalty+2.*band
                signature=tuple(map(tuple,points)); reverse=signature[::-1]
                key=min(signature,reverse)
                if key in unique: counters['duplicate']+=1; continue
                ys,xs=points[:,0],points[:,1]
                if (p0[ys,xs]<min(config.thresholds)).mean()<.05: continue
                radius=float(np.clip(2.*.5*(distance[tuple(a)]+distance[tuple(b)]),config.min_width_work,config.max_width_work))
                ta=tangents[i]; tb=tangents[j] if j is not None else None
                unique[key]={'endpoint_a':np.asarray(a,dtype=np.int32),'endpoint_b':np.asarray(b,dtype=np.int32),
                    'path_points':points,'path_length_work':length,'path_length_native':path_length(points,scale),
                    'scale_y':scale[0],'scale_x':scale[1],'native_shape':tuple(native_shape),
                    'endpoint_direction':(ta,tb),'source_confidence':(float(r[tuple(a)]),float(r[tuple(b)])),
                    'endpoint_p0':(float(p0[tuple(a)]),float(p0[tuple(b)])),
                    'source_component_area':(int(stats[component[tuple(a)],cv2.CC_STAT_AREA]),int(stats[component[tuple(b)],cv2.CC_STAT_AREA])),
                    'kind':kind,'threshold':float(threshold),'alternative':alternative,
                    'candidate_corridor_mask':soft_corridor(points,p0.shape,radius),
                    'width_work':radius,'width_native':radius*math.sqrt(scale[0]*scale[1])}
    paths=list(unique.values()); counters['paths_before_cap']=len(paths)
    if len(paths)>config.max_paths:
        groups={}
        for path in paths:
            key=(int(np.digitize(path['path_length_native'],[8,16,32])),path['kind'])
            groups.setdefault(key,[]).append(path)
        paths=[]; cursor=0
        while len(paths)<config.max_paths and any(cursor<len(items) for items in groups.values()):
            for key in sorted(groups):
                if cursor<len(groups[key]): paths.append(groups[key][cursor])
                if len(paths)>=config.max_paths: break
            cursor+=1
    counters['paths']=len(paths)
    return paths,counters


def label_paths(paths, gt_native, skeleton_native, p0_native, reference_threshold=.45, config=PathConfig()):
    """GT enters only here, after the prediction-only proposal list is frozen."""
    # Evaluate geometry on native pixels; never build inference paths from a GT path.
    gt=gt_native.astype(bool); sk=skeleton_native.astype(bool); h,width=gt.shape
    nodes,graph=skeleton_graph(sk)
    tree=cKDTree(nodes) if len(nodes) else None
    distance=cv2.distanceTransform((~sk).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    p0mask=p0_native>=reference_threshold
    missing=sk & ~(cv2.dilate(p0mask.astype(np.uint8),np.ones((5,5),np.uint8))>0)
    near_missing=cv2.dilate(missing.astype(np.uint8),np.ones((5,5),np.uint8))>0
    count,gap_labels,gap_stats,_=cv2.connectedComponentsWithStats(missing.astype(np.uint8),connectivity=8)
    coverage={i:0. for i in range(1,count) if gap_stats[i,cv2.CC_STAT_AREA]>=3}
    all_geometry=coverage.copy()
    results=[]; graph_cache={}
    for path in paths:
        scale=np.array((path['scale_y'],path['scale_x']))
        pts=np.rint((path['path_points']+.5)*scale-.5).astype(int)
        # Native line rasterization fills intervals between adjacent work-grid samples.
        line=np.zeros((h,width),np.uint8)
        cv2.polylines(line,[pts[:,::-1].reshape(-1,1,2)],False,1,1)
        py,px=np.nonzero(line); background=float((~gt[py,px]).mean()) if len(py) else 1.
        deviation=float(distance[py,px].mean()) if len(py) else float(max(h,width))
        near=float((distance[py,px]<=config.corridor_tolerance_native).mean()) if len(py) else 0.
        a=np.clip(np.rint((path['endpoint_a']+.5)*scale-.5).astype(int),(0,0),(h-1,width-1))
        b=np.clip(np.rint((path['endpoint_b']+.5)*scale-.5).astype(int),(0,0),(h-1,width-1))
        row={'label':-1,'subtype':'uncertain','uncertainty':'no_certified_gt_relation',
             'background_fraction':background,'gt_path_deviation_native':deviation,'gt_near_path_fraction':near}
        if tree is not None:
            snap,indices=tree.query(np.array((a,b)))
            row['endpoint_gt_snap_native']=list(map(float,snap))
            if max(snap)<=max(3.,path['width_native']):
                start,end=map(int,indices)
                if start not in graph_cache:
                    graph_cache[start]=dijkstra(graph,indices=start,directed=False,limit=max(20.,config.max_path_work*max(scale)*2.))
                gt_length=float(graph_cache[start][end]); row['gt_graph_length_native']=gt_length if np.isfinite(gt_length) else None
                plausible=start!=end and np.isfinite(gt_length) and gt_length<=max(6.,1.8*path['path_length_native'])
                if plausible and near>=.85 and background<=.2 and near_missing[py,px].any():
                    row.update(label=1,subtype='true_connection' if path['kind']=='pair' else 'true_extension',uncertainty=None)
                elif background>=.25 and (not plausible or near<.5):
                    ta,tb=path['endpoint_direction']; delta=b-a; delta=delta/max(np.linalg.norm(delta),1e-6)
                    parallel=ta is not None and tb is not None and abs(float(np.dot(ta,tb)))>.75 and abs(float(np.dot(ta,delta)))<.5
                    row.update(label=0,subtype='parallel_gap' if parallel else ('shortcut' if np.isfinite(gt_length) and not plausible else 'lateral_bridge'),uncertainty=None)
            if row['label']==-1 and background>=.85 and near<=.2:
                small=min(path['source_component_area'])<20
                row.update(label=0,subtype='isolated_extension' if small else ('false_extension' if path['kind']=='extension' else 'similar_linear_background'),uncertainty=None)
        elif background>=.95:
            row.update(label=0,subtype='similar_linear_background',uncertainty=None)
        # Only verified positive predicted paths contribute to discovery recall.
        if row['label']==1:
            tolerance=cv2.dilate(line,np.ones((5,5),np.uint8))>0
            for gap in coverage:
                gap_mask=gap_labels==gap
                fraction=float(tolerance[gap_mask].mean())
                all_geometry[gap]=max(all_geometry[gap],fraction)
                # Do not count source-free single-sided extensions as two-source repair reach.
                reliable_pair=path['kind']=='pair' and math.sqrt(max(np.prod(path['source_confidence']),0.))>=config.audit_min_source_confidence
                if reliable_pair: coverage[gap]=max(coverage[gap],fraction)
        results.append(row)
    gaps=[]
    for gap,covered in coverage.items():
        ys,xs=np.nonzero(gap_labels==gap)
        boundary=bool((ys<8).any() or (xs<8).any() or (ys>=h-8).any() or (xs>=width-8).any())
        length=int(gap_stats[gap,cv2.CC_STAT_AREA])
        gaps.append({'gap_id':gap,'length_native':length,'length_bin':'3-7' if length<8 else '8-15' if length<16 else '16-31' if length<32 else '32+',
                     'boundary_excluded':boundary,'any_hit':covered>0,'coverage_fraction':covered,'full_coverage':covered>=.99,
                     'any_geometry_coverage_fraction':all_geometry[gap],
                     'coverage_definition':'verified predicted pair path with sqrt(Ra*Rb)>=audit_min_source_confidence; not actual binary closure'})
    return results,gaps
