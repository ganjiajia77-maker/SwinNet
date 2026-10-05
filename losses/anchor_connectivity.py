import numpy as np
import torch
import torch.nn.functional as F

from datasets.anchor_reachability import lookup_distance


def connection_targets(output, graphs, snap_radius=3.0, max_geodesic=96.0,
                       max_detour=2.0):
    coords = output['anchor_coords'].detach().cpu().numpy()
    pairs = output['edge_pairs'].detach().cpu().numpy()
    valid = output['edge_valid'].detach().cpu().numpy().copy()
    labels = np.zeros(valid.shape, np.float32)
    for batch_index, graph in enumerate(graphs):
        y, x = coords[batch_index].T
        nearest = graph['nearest'][y, x]
        snapped = ((nearest >= 0) &
                   (graph['snap_distance'][y, x] <= snap_radius) &
                   ~graph['ambiguous'][y, x])
        a, b = pairs[batch_index, ..., 0], pairs[batch_index, ..., 1]
        valid[batch_index] &= snapped[a] & snapped[b] & (nearest[a] != nearest[b])
        geodesic = lookup_distance(graph, nearest[a], nearest[b])
        if int(graph['node_count']) > 0:
            snapped_coords = graph['coords'][nearest.clip(0)]
            distance = np.linalg.norm(snapped_coords[a] - snapped_coords[b], axis=-1)
            labels[batch_index] = ((geodesic <= max_geodesic) &
                                   (geodesic <= max_detour * distance + 2.0))
    device = output['edge_logits'].device
    return torch.as_tensor(labels, device=device), torch.as_tensor(valid, device=device)


def supervised_connection_loss(output, graphs, **kwargs):
    labels, valid = connection_targets(output, graphs, **kwargs)
    values = F.binary_cross_entropy_with_logits(output['edge_logits'].float(),
                                               labels, reduction='none')
    positive = valid & (labels > 0.5)
    negative = valid & ~positive
    # Balance present classes without an unstable ratio on empty/one-class batches.
    pos_count, neg_count = positive.sum(), negative.sum()
    pos_loss = (values * positive).sum() / pos_count.clamp_min(1)
    neg_loss = (values * negative).sum() / neg_count.clamp_min(1)
    classes = (pos_count > 0).float() + (neg_count > 0).float()
    loss = (pos_loss + neg_loss) / classes.clamp_min(1)
    stats = dict(connection_loss=loss.detach(), connection_positive=pos_count.detach(),
                 connection_negative=neg_count.detach(), connection_valid=valid.sum().detach())
    return loss, stats
