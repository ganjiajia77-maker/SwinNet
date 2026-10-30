"""Read-only gate replay on identical actions, with tie-aware ROC working points."""
import numpy as np
import torch

from networks.v9_transport_benefit import forward_trace, _exact_auc


@torch.no_grad()
def capture_bank(model, image=None, frozen_override=None):
    """Capture actual gate inputs without changing the original three-step forward."""
    if model.training: raise ValueError('Read-only action capture requires eval mode')
    relation_inputs, state_inputs = [], []
    handles = [
        model.gate_relation.register_forward_pre_hook(
            lambda unused, args: relation_inputs.append(tuple(
                value.detach() if isinstance(value, torch.Tensor) else value for value in args))),
        model.state_gate.register_forward_pre_hook(
            lambda unused, args: state_inputs.append(args[0].detach())),
    ]
    try:
        result = forward_trace(model, image, 3, frozen_override=frozen_override)
    finally:
        for handle in handles: handle.remove()
    if len(relation_inputs) != 1 or len(state_inputs) != 3:
        raise RuntimeError('Expected one semantic gate call and three actual state gate calls')
    return {'result': result, 'relation_inputs': relation_inputs[0], 'state_inputs': state_inputs}


@torch.no_grad()
def replay_gate(model, bank, mode='full_gate'):
    """Only score G. Never recompute/replace bank states, M, rates or source mass.

    full_gate includes the scorer's learned visual projection/DirectionRelations,
    evaluated on the bank's identical raw H/H3/P0/R inputs. Its internal W is a
    gate feature only, and is NEVER used to change the bank's physical action.
    residual_only additionally fixes old-G logit, W/two/one gate features, so only
    gate_relation/state_gate weights differ. It isolates that subset of G.
    """
    if model.training: raise ValueError('Read-only gate replay requires eval mode')
    highres, decoder, visual, p0, q, reliable, weights, two, one, relation_mode = bank['relation_inputs']
    if mode == 'full_gate':
        # Keep each learned visual coordinate system consistent with its own G.
        # This projection is a scoring feature; no R/M/rate/state is recomputed.
        visual = model.visual(highres, decoder)
        relation = model.relations(visual, p0, reliable, return_evidence=True)
        weights, two, one, old_logit = relation[0], relation[1], relation[2], relation[6]
    elif mode == 'residual_only':
        old_logit = bank['result']['g_v7_logit']
    else: raise ValueError('Expected full_gate or residual_only')
    delta, _ = model.gate_relation(highres, decoder, visual, p0, q, reliable,
                                    weights, two, one, relation_mode)
    state = torch.stack([model.state_gate(value) for value in bank['state_inputs']], dim=1)
    logits = (old_logit + delta)[:, None] + state
    if logits.shape != bank['result']['gate_logits'].shape or not bool(torch.isfinite(logits).all()):
        raise RuntimeError('Invalid fixed-action gate replay')
    return logits


def matched_acceptance(logits, labels, helpful_target, weights=None):
    """Match helpful TPR exactly in expectation using one uniform boundary tie rule.

    The same tie fraction applies to helpful AND harmful actions. No GT-based
    tie breaking. Also return attainable strict/inclusive deterministic bounds.
    Thresholds are diagnostic working points, never deployed as new G gates.
    """
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    weight = np.ones(len(logits), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if len(labels) != len(logits) or weight.shape != logits.shape:
        raise ValueError('Score/label/weight shapes differ')
    if not np.isfinite(logits).all() or not np.isfinite(weight).all() or (weight < 0).any():
        raise ValueError('Finite logits and nonnegative weights required')
    target = float(helpful_target)
    if not 0 <= target <= 1: raise ValueError('Helpful target must be in [0,1]')
    positive, negative = float(weight[labels].sum()), float(weight[~labels].sum())
    if positive == 0 or negative == 0:
        return {'available': False, 'helpful_actions': positive, 'harmful_actions': negative}
    if target == 0:
        return {'available': True, 'helpful_target': 0., 'helpful_acceptance': 0., 'harmful_acceptance': 0.,
                'threshold_logit': None, 'tie_fraction': 0., 'helpful_strict': 0., 'helpful_inclusive': 0.,
                'harmful_strict': 0., 'harmful_inclusive': 0., 'helpful_actions': positive, 'harmful_actions': negative}
    order = np.argsort(-logits[labels], kind='stable')
    scores, positive_weight = logits[labels][order], weight[labels][order]
    cutoff = min(int(np.searchsorted(positive_weight.cumsum(), target * positive, side='left')), len(scores)-1)
    boundary = float(scores[cutoff])
    above, equal = logits > boundary, logits == boundary
    pos_above, pos_equal = float(weight[above & labels].sum()), float(weight[equal & labels].sum())
    fraction = float(np.clip((target * positive - pos_above) / pos_equal, 0., 1.))
    neg_above, neg_equal = float(weight[above & ~labels].sum()), float(weight[equal & ~labels].sum())
    return {'available': True, 'helpful_target': target,
            'helpful_acceptance': (pos_above + fraction * pos_equal) / positive,
            'harmful_acceptance': (neg_above + fraction * neg_equal) / negative,
            'threshold_logit': boundary, 'tie_fraction': fraction,
            'helpful_strict': pos_above / positive, 'helpful_inclusive': (pos_above + pos_equal) / positive,
            'harmful_strict': neg_above / negative, 'harmful_inclusive': (neg_above + neg_equal) / negative,
            'helpful_actions': positive, 'harmful_actions': negative}


def compare_scores(labels, scores_b, scores_c, targets=(.5, .7, .8, .9, .95)):
    labels = np.asarray(labels, bool)
    b, c = np.asarray(scores_b), np.asarray(scores_c)
    if not (labels.shape == b.shape == c.shape): raise ValueError('Both scorers must judge exactly the same rows')
    raw = {}
    for name, value in (('B_control', b), ('C_gate3', c)):
        raw[name] = {'helpful': float((value[labels] >= 0).mean()) if labels.any() else None,
                     'harmful': float((value[~labels] >= 0).mean()) if (~labels).any() else None,
                     'auc': _exact_auc(value[labels], value[~labels])}
    targets = sorted(set(float(value) for value in targets) | (
        {raw['B_control']['helpful']} if raw['B_control']['helpful'] is not None else set()))
    rows = []
    for target in targets:
        left, right = matched_acceptance(b, labels, target), matched_acceptance(c, labels, target)
        rows.append({'helpful_target': target, 'B': left, 'C': right,
                     'harmful_C_minus_B': right['harmful_acceptance'] - left['harmful_acceptance']
                        if left['available'] and right['available'] else None,
                     'matches_B_default_helpful_rate': raw['B_control']['helpful'] == target})
    return {'actions': len(labels), 'helpful': int(labels.sum()), 'harmful': int((~labels).sum()),
            'raw_at_G_0_5': raw, 'matched_helpful': rows,
            'tie_rule': 'One uniform random acceptance fraction within the threshold tie; rates are exact expectations, not new model outputs. Strict/inclusive deterministic bounds are included.'}


def acceptance_points(logits, labels, targets):
    """Full curve from one sort, with the same boundary tie rule as above."""
    logits, labels = np.asarray(logits), np.asarray(labels, bool)
    positive, negative = int(labels.sum()), int((~labels).sum())
    if not positive or not negative: return []
    order = np.argsort(-logits, kind='stable'); values = logits[order]
    ends = np.r_[np.nonzero(np.diff(values))[0], len(order)-1]
    tp = labels[order].cumsum()[ends]; fp = ends+1-tp
    result = []
    for target in targets:
        if target == 0: result.append({'helpful_acceptance': 0., 'harmful_acceptance': 0.}); continue
        index = min(int(np.searchsorted(tp, target * positive, side='left')), len(tp)-1)
        before_tp, before_fp = (tp[index-1], fp[index-1]) if index else (0,0)
        fraction = (target * positive - before_tp) / (tp[index] - before_tp)
        result.append({'helpful_acceptance': float(target),
                       'harmful_acceptance': float((before_fp + fraction * (fp[index] - before_fp)) / negative)})
    return result


def bootstrap_difference(labels, scores_b, scores_c, image_ids, targets, repetitions=100, seed=1234):
    """Paired IMAGE bootstrap, keeping the same action rows and target TPR.

    Sorting once per scorer avoids repeated O(n log n) sorting of full val maps.
    """
    labels = np.asarray(labels, bool); ids = np.asarray(image_ids)
    images, inverse = np.unique(ids, return_inverse=True)
    if repetitions < 1 or len(images) < 2 or not labels.any() or labels.all(): return []
    prepared = []
    for scores in (scores_b, scores_c):
        order = np.argsort(-np.asarray(scores), kind='stable')
        sorted_scores = np.asarray(scores)[order]
        ends = np.r_[np.nonzero(np.diff(sorted_scores))[0], len(order)-1]
        prepared.append((order, ends))
    rng = np.random.default_rng(seed); values = {float(t): [] for t in targets}
    for _ in range(repetitions):
        multiplier = np.bincount(rng.integers(0, len(images), len(images)), minlength=len(images))
        weight = multiplier[inverse]
        positive, negative = weight[labels].sum(), weight[~labels].sum()
        if positive == 0 or negative == 0: continue
        estimates = []
        for order, ends in prepared:
            ordered_weight = weight[order]
            tp = (ordered_weight * labels[order]).cumsum()[ends]
            fp = (ordered_weight * ~labels[order]).cumsum()[ends]
            current = {}
            for target in values:
                if target == 0: current[target] = 0.; continue
                index = min(int(np.searchsorted(tp, target * positive, side='left')), len(tp)-1)
                before_tp = tp[index-1] if index else 0
                before_fp = fp[index-1] if index else 0
                fraction = np.clip((target * positive - before_tp) / max(tp[index] - before_tp, 1), 0., 1.)
                current[target] = float((before_fp + fraction * (fp[index] - before_fp)) / negative)
            estimates.append(current)
        for target in values: values[target].append(estimates[1][target] - estimates[0][target])
    return [{'helpful_target': target, 'repetitions': len(value), 'image_clusters': len(images),
             'delta_harmful_C_minus_B_ci95': np.quantile(value, [.025,.975]).tolist()}
            for target, value in values.items() if value]
