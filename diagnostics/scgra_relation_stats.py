"""Descriptive statistics of attention over the exact valid anchors."""
import numpy as np


MODES = ('base', 'conn', 'pos', 'full')


def matrices(data, field, valid):
    return np.asarray(data[field], dtype=np.float64)[:, valid][:, :, valid]


def centered_rms(values):
    # Softmax is invariant to adding a constant to every key of a query.
    centered = values - values.mean(-1, keepdims=True)
    return float(np.sqrt(np.square(centered).mean()))


def entropy(weights):
    if weights.shape[-1] < 2:
        return np.zeros(weights.shape[:-1])
    return -(weights * np.log(np.clip(weights, 1e-300, 1.))).sum(-1)/np.log(weights.shape[-1])


def window_statistics(data):
    valid = np.flatnonzero(data['anchor_valid'])
    if len(valid) < 2:
        return []
    parts = {
        name: matrices(data, field, valid)
        for name, field in (
            ('qk', 'base_qk_logits'), ('conn', 'connectivity_bias_effective'),
            ('pos', 'spatial_bias_effective'),
        )
    }
    attention = {name: matrices(data, 'attention_'+name, valid) for name in MODES}
    base = attention['base']
    results = []
    for head in range(base.shape[0]):
        scores = np.sort(parts['qk'][head], axis=-1)
        stats = dict(head=head+1, valid_anchors=len(valid),
                     qk_top1_margin_median=float(np.median(scores[:,-1]-scores[:,-2])))
        for name, values in parts.items():
            stats[name+'_row_centered_rms'] = centered_rms(values[head])
            stats[name+'_mean_row_range'] = float(np.ptp(values[head],axis=-1).mean())
        for name, values in attention.items():
            a = values[head]
            abs_change = np.abs(a-base[head])
            stats[name+'_normalized_entropy'] = float(entropy(a).mean())
            stats[name+'_dominant_key_mass'] = float(a.mean(axis=0).max())
            stats[name+'_mean_abs_change'] = float(abs_change.mean())
            stats[name+'_max_abs_change'] = float(abs_change.max())
            stats[name+'_top1_changed_queries'] = int((a.argmax(-1) != base[head].argmax(-1)).sum())
        results.append(stats)
    return results


def query_statistics(data, query_index):
    valid = np.flatnonzero(data['anchor_valid'])
    q = int(np.flatnonzero(valid == query_index)[0])
    coordinates = np.asarray(data['anchor_yx'])[valid]
    descriptor = np.asarray(data['anchor_connectivity_features'],dtype=np.float64)[valid]
    norms = np.linalg.norm(descriptor,axis=-1)
    cosine = (descriptor @ descriptor[q])/np.maximum(norms,1e-6)/max(norms[q],1e-6)
    distance = np.linalg.norm(coordinates-coordinates[q],axis=-1)
    nonself = np.arange(len(valid)) != q
    attention = {name:matrices(data,'attention_'+name,valid)[:,q] for name in MODES}
    results = []
    n_heads = attention['base'].shape[0]
    for head in range(n_heads+1):
        values = {k:(v.mean(0) if head == n_heads else v[head]) for k,v in attention.items()}
        for name, weight in values.items():
            winner = int(weight.argmax())
            other = int(np.argmax(np.where(nonself,weight,-1))) if nonself.any() else None
            mass = float(weight[nonself].sum())
            results.append(dict(
                head='mean' if head == n_heads else head+1, mode=name, query_id=query_index+1,
                top_key_id=int(valid[winner]+1), top_key_weight=float(weight[winner]),
                top_key_distance_px=float(distance[winner]), top_key_c_cosine=float(cosine[winner]),
                top_nonself_key_id=int(valid[other]+1) if other is not None and mass else None,
                top_nonself_key_weight=float(weight[other]) if other is not None and mass else None,
                top_nonself_key_distance_px=float(distance[other]) if other is not None and mass else None,
                top_nonself_key_c_cosine=float(cosine[other]) if other is not None and mass else None,
                nonself_weighted_c_cosine=float(np.dot(weight[nonself],cosine[nonself])/mass) if mass else None,
                normalized_entropy=float(entropy(weight)),
                max_abs_weight_change=float(np.abs(weight-values['base']).max()),
                total_variation_from_base=float(np.abs(weight-values['base']).sum()/2),
            ))
    return results
