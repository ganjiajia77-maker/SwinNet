"""Penalise actual wrong-source probability mass, without edge matching.

Keep the original joint-training files byte-identical: saved S_selector
checkpoints bind their hashes and must remain strictly loadable.
"""
import torch


def late_choice_loss(result, relation_info):
    """Live nine-way LSE loss; native labels/eligibility are supervision only.

    A supervised position has at least one eligible, known, valid relation
    under the EXISTING road-evidence rule. At that position ALL confirmed
    wrong valid neighbours enter the forbidden set, including unmatched
    edges. Unknown sources remain allowed. Positions without wrong sources
    contribute zero (and are counted explicitly). Empty images are skipped.
    """
    edge = result['source_edge_logits']
    stay = result['source_stay_logit']
    valid = result['source_valid'].detach().bool()
    labels = relation_info['native']['labels'].detach()
    eligible = relation_info['rows']['valid'].detach().bool()
    if edge.ndim != 5 or edge.shape[1:3] != (3, 8):
        raise ValueError('Expected live edge logits [B,3,8,H,W]')
    if stay.shape != (edge.shape[0], 3, 1, *edge.shape[-2:]):
        raise ValueError('Stay logits do not match the actual three-step choices')
    if valid.shape != edge.shape or eligible.shape != edge.shape:
        raise ValueError('Neighbour validity/eligibility shape differs')
    if labels.shape != (edge.shape[0], 8, *edge.shape[-2:]):
        raise ValueError('Native labels must describe all eight neighbours')
    losses = []; steps = []
    for k in (1, 2):
        logits = torch.cat((stay[:, k], edge[:, k].masked_fill(~valid[:, k], -torch.inf)), 1)
        wrong = (labels == 0) & valid[:, k]
        forbidden = torch.cat((torch.zeros_like(stay[:, k], dtype=torch.bool), wrong), 1)
        # stay is never forbidden, so the permitted LSE is always finite.
        permitted = logits.masked_fill(forbidden, -torch.inf)
        per_position = torch.logsumexp(logits, 1) - torch.logsumexp(permitted, 1)
        position = (eligible[:, k] & valid[:, k] & (labels >= 0)).any(1)
        image_losses = []; images = []
        for b in range(edge.shape[0]):
            take = position[b]; count = int(take.sum())
            if count:
                image_losses.append(per_position[b][take].mean())
            with torch.no_grad():
                wrong_mass = (result['source_choice'][b, k, 1:] * wrong[b]).sum(0)
                images.append({'image': b, 'positions': count, 'skipped': count == 0,
                    'wrong_bearing_positions': int((take & wrong[b].any(0)).sum()),
                    'all_valid_sources_wrong_positions': int((take & (wrong[b] | ~valid[b,k]).all(0) & valid[b,k].any(0)).sum()),
                    'wrong_mass_sum': float(wrong_mass[take].sum()),
                    'loss_sum': float(per_position[b][take].sum())})
        # The two specified steps retain equal weight; an entirely empty
        # step contributes graph-connected zero, and is explicitly reported.
        loss = torch.stack(image_losses).mean() if image_losses else (stay[:, k].sum() + edge[:, k].sum()) * 0.
        losses.append(loss)
        steps.append({'step': k + 1, 'images': images, 'valid_images': len(image_losses),
                      'skipped_images': edge.shape[0] - len(image_losses), 'loss': float(loss.detach())})
    return torch.stack(losses).mean(), {'steps': steps,
        'normalization': 'eligible positions -> nonempty images -> equal steps 2/3; no matching',
        'no_wrong_position_policy': 'eligible positions with no wrong source contribute zero'}
