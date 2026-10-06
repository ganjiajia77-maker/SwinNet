import numpy as np
import torch
from torch import nn

from diagnose_h0_connectivity_fragments import ConnectivityProbe, audit_short_components


class DummyNet(nn.Module):
    def __init__(self):
        super().__init__()
        block = nn.Module()
        block.structure_gate = nn.Identity()
        self.decoder_structure_blocks = nn.ModuleDict({"3": block})

    @staticmethod
    def _latest_local_topology_features(outputs):
        return outputs[0]


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin_unet = DummyNet()

    def forward(self, x):
        gate = self.swin_unet.decoder_structure_blocks["3"].structure_gate(x)
        global_feature = self.swin_unet._latest_local_topology_features([x[:, 2:3]])
        if global_feature is None:
            global_feature = torch.zeros_like(gate[:, -1:])
        return (gate[:, -1:] + global_feature,)


def test_gate_and_global_modes_are_independent():
    model = DummyModel()
    probe = ConnectivityProbe(model)
    x = torch.zeros(1, 3, 4, 4)
    x[:, -1] = torch.arange(16).reshape(4, 4)
    skeleton = torch.ones(1, 1, 4, 4)
    try:
        base = probe.run(model, x, skeleton, "baseline")
        off = probe.run(model, x, skeleton, "stage3_gate_off")
        shifted = probe.run(model, x, skeleton, "stage3_gate_shift")
        oracle = probe.run(model, x, skeleton, "stage3_gate_gt")
        no_global = probe.run(model, x, skeleton, "global_no_c3")
        assert torch.equal(base, 2 * x[:, -1:])
        assert torch.equal(off, x[:, -1:])
        assert torch.equal(no_global, x[:, -1:])
        assert torch.equal(shifted, torch.roll(x[:, -1:], (2, 2), (-2, -1)) + x[:, -1:])
        assert not torch.equal(oracle, base)
    finally:
        probe.close()
    assert torch.equal(model(x)[0], base)


def test_short_components_are_separated_by_gt_support():
    gt = np.zeros((12, 12), dtype=bool)
    gt[2, 1:8] = True
    gt[6, 1:3] = True
    pred = np.zeros_like(gt)
    pred[2, 1:3] = True
    pred[2, 5:7] = True
    pred[6, 1:3] = True
    pred[10, 10] = True
    rows = audit_short_components("case", pred, gt, gt, area_limit=3, tolerance=1)
    categories = [row["category"] for row in rows]
    assert categories.count("supported_fragment") == 2
    assert categories.count("supported_independent") == 1
    assert categories.count("likely_background_fp") == 1
