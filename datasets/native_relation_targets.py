"""Native-pixel local road adjacency and branch tangent targets for H0.

Input is the already augmented and cropped road/skeleton pair.  Direction
channels follow topology_direction_constants and GT is never used in forward.
"""
import cv2
import numpy as np
from topology_direction_constants import CONNECTIVITY_DIRECTIONS, CONNECTIVITY_OPPOSITE


def native_targets(mask, skeleton, stride=4, branch_steps=8, junction_radius=3, min_resultant=.7):
    road = np.asarray(mask, dtype=bool); skel = np.asarray(skeleton, dtype=bool)
    if road.ndim != 2 or road.shape != skel.shape or any(x % stride for x in road.shape):
        raise ValueError('Expected equally sized native masks divisible by the work stride')
    height, width = road.shape; h, w = height//stride, width//stride
    occupied = road.reshape(h, stride, w, stride).any(axis=(1, 3))
    ambiguous = np.zeros((h, w), bool)
    for cy, cx in np.argwhere(occupied):
        cell = road[cy*stride:(cy+1)*stride, cx*stride:(cx+1)*stride]
        count, _ = cv2.connectedComponents(cell.astype(np.uint8), connectivity=8)
        ambiguous[cy, cx] = count > 2
    target = np.zeros((8, h, w), np.float32)
    valid = np.zeros((8, h, w), bool)
    for d, (dy, dx) in enumerate(CONNECTIVITY_DIRECTIONS):
        sy=slice(max(0,-dy),min(h,h-dy)); sx=slice(max(0,-dx),min(w,w-dx))
        ty=slice(max(0,dy),min(h,h+dy)); tx=slice(max(0,dx),min(w,w+dx))
        valid[d, sy, sx] = ~(ambiguous[sy, sx] | ambiguous[ty, tx])
    direction_index={tuple(value):i for i,value in enumerate(CONNECTIVITY_DIRECTIONS)}
    # Enumerate genuine 8-neighbour road-pixel edges, not coarse occupancy pairs.
    for dy, dx in CONNECTIVITY_DIRECTIONS:
        sy=slice(max(0,-dy),min(height,height-dy));sx=slice(max(0,-dx),min(width,width-dx))
        ty=slice(max(0,dy),min(height,height+dy));tx=slice(max(0,dx),min(width,width+dx))
        yy,xx=np.nonzero(road[sy,sx] & road[ty,tx]);yy+=max(0,-dy);xx+=max(0,-dx)
        cy,cx=yy//stride,xx//stride;ny,nx=(yy+dy)//stride,(xx+dx)//stride
        delta=np.stack((ny-cy,nx-cx),axis=1)
        for offset,index in direction_index.items():
            select=(delta[:,0]==offset[0])&(delta[:,1]==offset[1])
            target[index,cy[select],cx[select]]=1.
    target *= valid

    # Trace only unambiguous two-neighbour skeleton branches at native scale.
    points={tuple(p) for p in np.argwhere(skel)}
    neighbours={p:[(p[0]+dy,p[1]+dx) for dy,dx in CONNECTIVITY_DIRECTIONS
                    if (p[0]+dy,p[1]+dx) in points] for p in points}
    junction=np.zeros_like(skel,np.uint8)
    for p,ns in neighbours.items():
        if len(ns)!=2: junction[p]=1
    if junction_radius:
        k=2*junction_radius+1
        junction=cv2.dilate(junction,np.ones((k,k),np.uint8))
    vectors=np.zeros((2,h,w),np.float64);counts=np.zeros((h,w),np.int32)
    def trace(origin,first):
        previous,current=origin,first
        for _ in range(max(0,branch_steps-1)):
            ns=neighbours.get(current,[])
            if len(ns)!=2: break
            nxt=ns[0] if ns[1]==previous else ns[1]
            previous,current=current,nxt
        return current
    for p,ns in neighbours.items():
        cy,cx=p[0]//stride,p[1]//stride
        if junction[p] or len(ns)!=2 or ambiguous[cy,cx]:continue
        a,b=trace(p,ns[0]),trace(p,ns[1]);dy,dx=b[0]-a[0],b[1]-a[1]
        length2=dx*dx+dy*dy
        if length2<4:continue
        vectors[0,cy,cx]+=(dx*dx-dy*dy)/length2
        vectors[1,cy,cx]+=2.*dx*dy/length2
        counts[cy,cx]+=1
    norm=np.sqrt((vectors*vectors).sum(axis=0));direction_valid=(counts>0)&(norm>=min_resultant*counts)&~ambiguous
    direction=np.zeros((2,h,w),np.float32)
    direction[:,direction_valid]=(vectors[:,direction_valid]/norm[direction_valid]).astype(np.float32)
    return {'connectivity':target, 'connectivity_valid':valid, 'direction':direction,
            'direction_valid':direction_valid, 'ambiguous_cells':ambiguous}
