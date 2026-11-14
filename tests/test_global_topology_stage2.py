import torch
from torch import nn

from h0_phase_training import tensor_digests
from networks.encoder_h_postmerge_phase import install_encoder_h_postmerge_phase
from scripts.diagnose_h0_global_anchor_gaps import capture_image
from test_encoder_h_postmerge_phase import make_core


def test_stage2_uses_its_outputs_before_stage3_skip_and_matches_disabled_at_zero(monkeypatch):
    core = make_core().eval()
    baseline_state = tensor_digests(core)
    model = nn.Module()
    model.swin_unet = core
    model.forward = core.forward
    install_encoder_h_postmerge_phase(model)
    assert all(baseline_state[key] == value for key, value in tensor_digests(core).items()
               if key in baseline_state)

    events = []
    inputs = {}
    original = core.global_topology.forward_feature_anchors

    def observe(feature, h, prior, *, connectivity_feature):
        events.append('global')
        inputs.update(feature=feature, h=h, prior=prior, c=connectivity_feature)
        result = original(feature, h, prior, connectivity_feature=connectivity_feature)
        inputs['updated'] = result
        return result

    def observe_skip(*args, **kwargs):
        events.append('stage3_skip')
        deep = kwargs['deep']
        if 'updated' in inputs:
            assert torch.equal(deep, inputs['updated'])

    monkeypatch.setattr(core.global_topology, 'forward_feature_anchors', observe)
    handle = core.dca_blocks[1].register_forward_pre_hook(
        lambda module, args, kwargs: observe_skip(*args, **kwargs), with_kwargs=True)
    image = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        # Set alpha explicitly so the zero-residual case does not depend on initialization.
        core.global_topology.raw_alpha.zero_()
        relocated = model(image)
        assert events == ['global', 'stage3_skip']
        assert inputs['feature'].shape == (1, 96, 32, 32)
        assert inputs['prior'].shape == (1, 1, 32, 32) and not inputs['prior'].requires_grad
        assert inputs['h'] is core.last_stage_features['H1']
        c2 = next(item['connectivity'] for item in relocated[4] if item['stage'] == 2)
        assert torch.equal(inputs['c'], c2.sigmoid()) and not inputs['c'].requires_grad

        core.enable_global_topology = False
        inputs.clear()
        events.clear()
        baseline_disabled = model(image)
        assert events == ['stage3_skip']
        assert torch.equal(relocated[0], baseline_disabled[0])
        assert torch.equal(relocated[1], baseline_disabled[1])
        for stage in (2, 3):
            a = next(item for item in relocated[4] if item['stage'] == stage)
            b = next(item for item in baseline_disabled[4] if item['stage'] == stage)
            for name in ('skeleton', 'connectivity', 'direction'):
                assert torch.equal(a[name], b[name])
    handle.remove()

    core.enable_global_topology = True
    prediction = model(image)[0]
    gradient = torch.autograd.grad(prediction.square().mean(), core.global_topology.raw_alpha)[0]
    assert torch.isfinite(gradient) and gradient.abs() > 0
    with torch.no_grad():
        core.global_topology.raw_alpha.fill_(.3)
    prediction = model(image)[0]
    gradient = torch.autograd.grad(prediction.square().mean(), core.global_topology.grid_projection.weight)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    with torch.no_grad():
        assert not torch.equal(prediction, baseline_disabled[0])
    _, records, same = capture_image(model, image[0], torch.device('cpu'),
                                     check_first=True, tile=128, stride=64)
    assert same and records[0]['anchor_scale'] == 4
    for kind in ('candidate', 'selected'):
        assert (records[0][kind+'_coords'] == records[0][kind+'_grid_coords']*4+2).all()
