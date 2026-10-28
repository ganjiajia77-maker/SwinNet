"""GT-only training audit of local path coverage and bounded-step reachability."""
import cv2
import numpy as np

from networks.reliable_directional_repair import DIRECTIONS


def ordered_simple_path(component):
    points = {tuple(p) for p in np.argwhere(component)}
    neighbors = {p: [q for dy,dx in DIRECTIONS if (q := (p[0]+dy,p[1]+dx)) in points] for p in points}
    ends = [p for p in points if len(neighbors[p]) == 1]
    if len(ends) != 2 or any(len(v) > 2 for v in neighbors.values()):
        return None
    path = [min(ends)]; previous = None
    while True:
        choices = [p for p in neighbors[path[-1]] if p != previous]
        if not choices: break
        previous = path[-1]; path.append(choices[0])
        if len(path) > len(points): return None
    return path if len(path) == len(points) else None


def shift_read(array, dy, dx):
    h,w=array.shape; result=np.zeros_like(array)
    y0,y1=max(0,-dy),min(h,h-dy); x0,x1=max(0,-dx),min(w,w-dx)
    result[y0:y1,x0:x1]=array[y0+dy:y1+dy,x0+dx:x1+dx]
    return result


def reach_three_steps(path_work, anchors, candidates, supports, weights, min_weight, min_support):
    """Directed work-grid reach through original-run candidate maps; not a rerun."""
    h,w=anchors.shape; corridor=np.zeros((h,w),np.uint8)
    for y,x in path_work: corridor[y,x]=1
    corridor=cv2.dilate(corridor,np.ones((3,3),np.uint8))>0
    reach=anchors.copy(); steps=[reach.copy()]
    for k in range(3):
        neighbors=np.zeros((h,w),bool)
        src=reach & (supports[k]>=min_support)
        for direction,(dy,dx) in enumerate(DIRECTIONS):
            neighbors |= shift_read(src,dy,dx) & (weights[direction]>=min_weight)
        reach = reach | (corridor & candidates[k] & neighbors)
        steps.append(reach.copy())
    return steps


def longest_false_run(flags):
    longest=run=0
    for flag in flags:
        run=0 if flag else run+1
        longest=max(longest,run)
    return longest


def near_anchor(point, p0, reliable, gt_work, reference_threshold, reliable_threshold, radius=2):
    h,w=p0.shape; y,x=point; result=np.zeros((h,w),bool)
    y0,y1=max(0,y-radius),min(h,y+radius+1); x0,x1=max(0,x-radius),min(w,x+radius+1)
    yy,xx=np.ogrid[y0:y1,x0:x1]
    result[y0:y1,x0:x1]=(yy-y)**2+(xx-x)**2<=radius**2
    return result & (p0>=reference_threshold) & (reliable>=reliable_threshold) & gt_work


def length_bin(n):
    return '3-7' if n<8 else '8-15' if n<16 else '16-31' if n<32 else '32+'


def inspect_gaps(result, gt, skeleton, case, p0_threshold, p1_threshold,
                 min_effect=1e-5, reliable_threshold=.1, min_weight=.01, min_support=1e-6):
    """One predicted 512 crop. GT is used only for audit region definition."""
    def n(key): return result[key].detach().cpu().numpy()[0]
    p0full=n('p0')[0]; p1full=n('p1')[0]; p0=n('u0')[0]; reliable=n('r')[0]
    weights=n('w'); rate=n('transport_rate')[:,0]; m=n('m')[:,0]
    states=np.stack([n(f'u{k}')[0] for k in range(3)])
    available=n('source_available')[:,0]
    potential=rate*(m-states)
    candidates=(potential>min_effect)&(available>min_support)
    support=np.stack([reliable+(1-reliable)*np.clip((states[k]-p0)/np.maximum(1-p0,1e-5),0,1) for k in range(3)])
    gt=gt.astype(bool); skeleton=skeleton.astype(bool)
    reference=p0full>=p0_threshold; prediction=p1full>=p1_threshold
    missing=skeleton & ~(cv2.dilate(reference.astype(np.uint8),np.ones((5,5),np.uint8))>0)
    num,labels,stats,_=cv2.connectedComponentsWithStats(missing.astype(np.uint8),connectivity=8)
    _, old_labels=cv2.connectedComponents(reference.astype(np.uint8),connectivity=8)
    _, new_labels=cv2.connectedComponents(prediction.astype(np.uint8),connectivity=8)
    h,w=p0.shape; gh,gw=gt.shape
    gt_work=cv2.resize(gt.astype(np.float32),(w,h),interpolation=cv2.INTER_AREA)>0
    rows=[]; gap_work={}
    for i in range(1,num):
        length=int(stats[i,cv2.CC_STAT_AREA])
        if length<3: continue
        component=labels==i; points=np.argwhere(component)
        work=np.column_stack((np.minimum(points[:,0]*h//gh,h-1),np.minimum(points[:,1]*w//gw,w-1)))
        gap_work[i]=set(map(tuple,work))
        order=ordered_simple_path(component)
        flags=candidates[:,work[:,0],work[:,1]].any(axis=0)
        row={'case':case,'gap_id':i,'length_pixels':length,'length_bin':length_bin(length),
             'centroid_y':float(points[:,0].mean()),'centroid_x':float(points[:,1].mean()),
             'touches_tile_margin':bool((points[:,0]<7).any() or (points[:,1]<7).any() or
                                         (points[:,0]>=gh-7).any() or (points[:,1]>=gw-7).any()),
             'positive_candidate_fraction':float(flags.mean()),'at_least_one_candidate':bool(flags.any()),
             'candidate_full_coverage':bool(flags.all()),'simple_path':order is not None,
             'longest_consecutive_no_candidate_pixels':None,'center_reachable_3_steps':None,
             'both_flanks_reach_center_3_steps':None,'full_path_reachable_3_steps':None,
             'actual_binary_closed':None,'actual_closure_eligible':False,'anchor_left_count':0,'anchor_right_count':0}
        if order is not None:
            path_work=[(min(y*h//gh,h-1),min(x*w//gw,w-1)) for y,x in order]
            path_flags=[bool(candidates[:,y,x].any()) for y,x in path_work]
            row['longest_consecutive_no_candidate_pixels']=longest_false_run(path_flags)
            left=near_anchor(path_work[0],p0,reliable,gt_work,p0_threshold,reliable_threshold)
            right=near_anchor(path_work[-1],p0,reliable,gt_work,p0_threshold,reliable_threshold)
            row['anchor_left_count']=int(left.sum()); row['anchor_right_count']=int(right.sum())
            lhs=reach_three_steps(path_work,left,candidates,support,weights,min_weight,min_support)
            rhs=reach_three_steps(path_work,right,candidates,support,weights,min_weight,min_support)
            center=path_work[len(path_work)//2]
            row['center_reachable_3_steps']=bool(lhs[-1][center] or rhs[-1][center])
            row['both_flanks_reach_center_3_steps']=bool(lhs[-1][center] and rhs[-1][center])
            row['full_path_reachable_3_steps']=bool(all(lhs[-1][pt] or rhs[-1][pt] for pt in path_work))
            for k in range(1,4): row[f'path_reachable_fraction_step{k}']=float(np.mean([lhs[k][pt] or rhs[k][pt] for pt in path_work]))
            # A closure proxy requires two distinct old-road flank components and
            # the same new component touching both, plus 80% gap coverage.
            flank=[]
            for y,x in (order[0],order[-1]):
                y0,y1=max(0,y-7),min(gh,y+8); x0,x1=max(0,x-7),min(gw,x+8)
                old=old_labels[y0:y1,x0:x1]; target=gt[y0:y1,x0:x1]
                vals,cnt=np.unique(old[target & (old>0)],return_counts=True)
                flank.append(int(vals[np.argmax(cnt)]) if len(vals) else 0)
            if flank[0] and flank[1] and flank[0]!=flank[1]:
                row['actual_closure_eligible']=True
                touching=[set(np.unique(new_labels[(old_labels==label)&gt]))-{0} for label in flank]
                row['actual_binary_closed']=bool(prediction[component].mean()>=.8 and touching[0].intersection(touching[1]))
        rows.append(row)
    return rows,gap_work
