"""Native cross-cell contact supervision; no path search or sampling.

Only four oriented representatives of undirected edges are used. Pixel pair
geometry matches native_targets' eight-neighbour adjacency exactly.
"""
import torch
import torch.nn.functional as F
from topology_direction_constants import CONNECTIVITY_DIRECTIONS


UNDIRECTED = tuple((d, dy, dx) for d, (dy, dx) in enumerate(CONNECTIVITY_DIRECTIONS)
                   if dy > 0 or (dy == 0 and dx > 0))
_INDEX_CACHE = {}


def contact_indices(height, width, stride, device):
    key = (height, width, stride, str(device))
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]
    h, w = height // stride, width // stride
    geometry = []
    for d, dy, dx in UNDIRECTED:
        # At most ten pairs per cardinal edge (stride=4), one per diagonal.
        # This small static loop constructs geometry once, never loops over pixels.
        axis = torch.arange(stride, device=device)
        off = torch.stack(torch.meshgrid(axis, axis, axis, axis, indexing='ij'), -1).reshape(-1, 4)
        contact = torch.maximum((dy*stride+off[:, 2]-off[:, 0]).abs(),
                                (dx*stride+off[:, 3]-off[:, 1]).abs()) == 1
        off = off[contact]
        yy, xx = torch.meshgrid(torch.arange(max(0, -dy), min(h, h-dy), device=device),
                                torch.arange(max(0, -dx), min(w, w-dx), device=device), indexing='ij')
        cy, cx = yy.reshape(-1), xx.reshape(-1)
        first = (cy[:, None]*stride+off[:, 0])*width+cx[:, None]*stride+off[:, 1]
        second = ((cy[:, None]+dy)*stride+off[:, 2])*width+(cx[:, None]+dx)*stride+off[:, 3]
        geometry.append((d, cy*w+cx, first, second))
    _INDEX_CACHE[key] = geometry
    return geometry


def local_cs_loss(surface_logits, c_logits, road_gt, relations, mode='c_guided'):
    if mode not in ('surface_only', 'c_guided'):
        raise ValueError(mode)
    n, _, height, width = surface_logits.shape
    h, w = c_logits.shape[-2:]
    stride = height // h
    if height != h*stride or width != w*stride or road_gt.shape != surface_logits.shape:
        raise ValueError('Surface, native GT and work grid must align exactly')
    if c_logits.shape != (n, 8, h, w):
        raise ValueError('Expected eight native direction channels')
    logits = surface_logits[:, 0].flatten(1)
    with torch.no_grad():
        probability = logits.sigmoid()
        road = road_gt[:, 0].flatten(1) > .5
        labels = relations['connectivity'].detach().reshape(n, 8, -1) > .5
        valid = relations['connectivity_valid'].detach().reshape(n, 8, -1).bool()
        confidence = c_logits.detach().sigmoid().reshape(n, 8, -1)
    counts = {'positive': 0, 'negative': 0, 'positive_without_contact': 0,
              'negative_with_gt_contact': 0, 'positive_empty_images': 0, 'negative_empty_images': 0}
    # Per-image normalization, including empty classes as zero.
    pos_sum = logits.sum(1)*0
    neg_sum = logits.sum(1)*0
    pos_count = torch.zeros(n, device=logits.device, dtype=torch.long)
    neg_count = torch.zeros_like(pos_count)
    pos_weights = torch.zeros(n, device=logits.device)
    neg_weights = torch.zeros_like(pos_weights)
    positive_mismatch = torch.zeros((), device=logits.device, dtype=torch.long)
    negative_mismatch = torch.zeros_like(positive_mismatch)
    for d, cells, first, second in contact_indices(height, width, stride, logits.device):
        with torch.no_grad():
            y = labels[:, d, cells]
            v = valid[:, d, cells]
            left_road, right_road = road[:, first], road[:, second]
            gt_contact = left_road & right_road
            pair_score = torch.minimum(probability[:, first], probability[:, second])
            positive_mismatch += (v & y & ~gt_contact.any(-1)).sum()
            negative_mismatch += (v & ~y & gt_contact.any(-1)).sum()
            # Positive: select among genuine GT contacts; negative: all pairs.
            eligible = ~y[..., None] | gt_contact
            selected = pair_score.masked_fill(~eligible, -1).argmax(-1, keepdim=True)
            a = first[None].expand(n, -1, -1).gather(-1, selected).squeeze(-1)
            b = second[None].expand(n, -1, -1).gather(-1, selected).squeeze(-1)
            ra, rb = road.gather(1, a), road.gather(1, b)
            ca = confidence[:, d, cells]
            weight = torch.where(y, ca, 1-ca) if mode == 'c_guided' else torch.ones_like(ca)
            pos, neg = v & y, v & ~y
        za, zb = logits.gather(1, a), logits.gather(1, b)
        pl = (F.softplus(-za)+F.softplus(-zb))*.5
        na, nb = (~ra).float(), (~rb).float()
        nl = (F.softplus(za)*na+F.softplus(zb)*nb)/(na+nb).clamp_min(1)
        pos_sum = pos_sum+(pl*weight*pos).sum(1)
        neg_sum = neg_sum+(nl*weight*neg).sum(1)
        pos_count += pos.sum(1)
        neg_count += neg.sum(1)
        pos_weights += (weight*pos).sum(1)
        neg_weights += (weight*neg).sum(1)
    counts['positive_without_contact'] = int(positive_mismatch.item())
    counts['negative_with_gt_contact'] = int(negative_mismatch.item())
    if counts['positive_without_contact'] or counts['negative_with_gt_contact']:
        raise RuntimeError(f'Native label/contact mismatch: {counts}')
    lp = (pos_sum/pos_count.clamp_min(1)).mean()
    ln = (neg_sum/neg_count.clamp_min(1)).mean()
    counts.update(positive=int(pos_count.sum().item()), negative=int(neg_count.sum().item()),
                  positive_empty_images=int((pos_count == 0).sum().item()),
                  negative_empty_images=int((neg_count == 0).sum().item()),
                  pos_loss=float(lp.detach()), neg_loss=float(ln.detach()),
                  positive_weight_mean=float((pos_weights/pos_count.clamp_min(1)).mean()),
                  negative_weight_mean=float((neg_weights/neg_count.clamp_min(1)).mean()))
    return .5*(lp+ln), counts
