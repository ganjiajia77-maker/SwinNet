import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from datasets.native_relation_targets import native_targets
from losses.local_cs_loss import contact_indices, local_cs_loss, UNDIRECTED
from topology_direction_constants import CONNECTIVITY_DIRECTIONS


def targets(road):
    native = native_targets(road, np.zeros_like(road))
    return {k: torch.from_numpy(v)[None] for k, v in native.items()}


def edge(road, offset, cell=(0, 0), positive=True):
    t = targets(road)
    t['connectivity_valid'].zero_()
    d = CONNECTIVITY_DIRECTIONS.index(offset)
    t['connectivity_valid'][0, d, *cell] = True
    t['connectivity'][0, d, *cell] = float(positive)
    return t


@pytest.mark.parametrize('scene', ['horizontal', 'vertical', 'diagonal', 'parallel', 'gap', 'ambiguous', 'crop'])
def test_geometry_equals_native_labels(scene):
    road = np.zeros((16, 16), bool)
    if scene == 'horizontal': road[5, :] = True
    if scene == 'vertical': road[:, 5] = True
    if scene == 'diagonal': np.fill_diagonal(road, True)
    if scene == 'parallel': road[:, 2] = road[:, 5] = True
    if scene == 'gap': road[5, :7] = True; road[5, 9:] = True
    if scene == 'ambiguous': road[4, 4] = road[7, 7] = True
    if scene == 'crop':
        full = np.zeros((23, 23), bool); full[9, :] = True
        road = full[3:19, 1:17]
    for augmented in (road, np.flip(road, 0).copy(), np.rot90(road).copy()):
        t = targets(augmented)
        flat = torch.from_numpy(augmented.reshape(-1))
        for d, cells, a, b in contact_indices(16, 16, 4, torch.device('cpu')):
            observed = (flat[a] & flat[b]).any(-1)
            valid = t['connectivity_valid'][0, d].flatten()[cells]
            expected = t['connectivity'][0, d].flatten()[cells] > .5
            assert torch.equal(observed[valid], expected[valid])
            assert ((a % 16-b % 16).abs() <= 1).all()  # no wrap
            assert ((a//16-b//16).abs() <= 1).all()


def test_positive_never_selects_background_and_c_detached():
    road = np.zeros((8, 8), bool); road[1, 3:5] = True
    t = edge(road, (0, 1))
    z = torch.full((1, 1, 8, 8), 8., requires_grad=True)
    with torch.no_grad(): z[0, 0, 1, 3:5] = -4
    c = torch.zeros((1, 8, 2, 2), requires_grad=True)
    loss, stats = local_cs_loss(z, c, torch.tensor(road)[None, None], t)
    loss.backward()
    assert c.grad is None
    expected = torch.zeros_like(z, dtype=torch.bool); expected[0, 0, 1, 3:5] = True
    assert torch.equal(z.grad != 0, expected)
    assert (z.grad[expected] < 0).all()
    assert stats['positive'] == 1
    assert torch.allclose(loss.detach(), .25*F.softplus(torch.tensor(4.)))


def test_negative_only_background_and_reselects():
    road = np.zeros((8, 8), bool); road[1, 3] = True
    t = edge(road, (0, 1), positive=False)
    z = torch.full((1, 1, 8, 8), -8., requires_grad=True)
    with torch.no_grad(): z[0, 0, 1, 3:5] = 6
    c = torch.zeros((1, 8, 2, 2), requires_grad=True)
    loss, _ = local_cs_loss(z, c, torch.tensor(road)[None, None], t)
    loss.backward()
    assert z.grad[0, 0, 1, 3] == 0
    assert z.grad[0, 0, 1, 4] > 0
    assert (z.grad != 0).sum() == 1
    with torch.no_grad():
        z.fill_(-8); z[0, 0, 2, 3:5] = 6
    z.grad = None
    local_cs_loss(z, c, torch.tensor(road)[None, None], t)[0].backward()
    assert (z.grad != 0).sum() == 2
    assert z.grad[0, 0, 2, 3:5].min() > 0


def test_empty_and_mismatch():
    road = np.zeros((8, 8), bool)
    t = edge(road, (0, 1), positive=True)
    z = torch.zeros((1, 1, 8, 8), requires_grad=True)
    c = torch.zeros((1, 8, 2, 2))
    with pytest.raises(RuntimeError, match='mismatch'):
        local_cs_loss(z, c, torch.tensor(road)[None, None], t)
    t['connectivity_valid'].zero_()
    loss, stats = local_cs_loss(z, c, torch.tensor(road)[None, None], t)
    loss.backward()
    assert loss == 0 and z.grad.abs().sum() == 0
    assert stats['positive_empty_images'] == stats['negative_empty_images'] == 1


def test_weight_not_renormalized_and_unit_weight_control():
    road = np.zeros((8, 8), bool)
    t = edge(road, (1, 1), positive=False)
    z = torch.zeros((1, 1, 8, 8), requires_grad=True)
    c = torch.full((1, 8, 2, 2), float(np.log(9)))
    guided, _ = local_cs_loss(z, c, torch.tensor(road)[None, None], t)
    direct, _ = local_cs_loss(z, c, torch.tensor(road)[None, None], t, 'surface_only')
    assert torch.allclose(guided, direct*.1)
    assert len(UNDIRECTED) == 4


def test_batch_normalization_matches_image_average():
    road = np.zeros((8, 8), bool)
    t = targets(road)
    z = torch.zeros((1, 1, 8, 8), requires_grad=True)
    c = torch.zeros((1, 8, 2, 2))
    one, _ = local_cs_loss(z, c, torch.tensor(road)[None, None], t)
    tb = {k: v.repeat((2,) + (1,)*(v.ndim-1)) for k, v in t.items()}
    tb['connectivity_valid'][1].zero_()
    two, _ = local_cs_loss(z.repeat(2, 1, 1, 1), c.repeat(2, 1, 1, 1),
                           torch.tensor(road)[None, None].repeat(2, 1, 1, 1), tb)
    assert torch.allclose(two, one/2)


def test_original_loss_integration_and_eval_disable():
    from losses.road_losses import SurfaceStructureLoss
    criterion = SurfaceStructureLoss(stage_structure_weights=(0., 0., .008, .012))
    criterion.native_relation_enabled = True
    mask = torch.zeros(1, 1, 16, 16); mask[:, :, 6:9, :] = 1
    z = torch.zeros_like(mask, requires_grad=True)
    c = torch.zeros(1, 8, 4, 4, requires_grad=True)
    stage = [dict(stage=3, connectivity=c, skeleton=torch.zeros(1, 1, 4, 4))]
    kwargs = dict(surface_gt=mask, skeleton_gt=mask, skeleton_dilate_gt=mask, stage_outputs=stage)
    original, _ = criterion(z, **kwargs)
    criterion.local_cs_mode = 'off'
    off, _ = criterion(z, **kwargs)
    torch.testing.assert_close(original, off, rtol=0, atol=0)
    criterion.local_cs_mode = 'c_guided'; criterion.local_cs_weight = .05
    guided, _ = criterion(z, **kwargs)
    weighted = criterion.last_local_cs['weighted']
    torch.testing.assert_close(guided, original+weighted)
    assert torch.autograd.grad(weighted, c, allow_unused=True, retain_graph=True)[0] is None
    assert torch.autograd.grad(weighted, z)[0].abs().sum() > 0
    with torch.no_grad():
        val, _ = criterion(z, surface_gt=mask, skeleton_gt=mask, skeleton_dilate_gt=mask)
        assert torch.isfinite(val) and criterion.last_local_cs is None
