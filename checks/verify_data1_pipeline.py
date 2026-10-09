"""Check historical crop failure, original target convention, and CUDA model gradients."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data1_common import (Data1Train, binary_prediction, connectivity_targets, create_model,
                          image_tensor, predict_full, source_loss, split_index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--pretrain_ckpt', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    index = split_index(args.root_path, 'train')[:4]
    crop_runs = []
    for epoch in (0, 1, 0):
        dataset = Data1Train(args.root_path, epoch=epoch, index=index)
        loader = DataLoader(dataset, batch_size=2, num_workers=2, persistent_workers=False)
        rows = []
        for batch in loader:
            assert (batch['crop_epoch'] == epoch).all()
            assert tuple(batch['image'].shape[1:]) == (3, 512, 512)
            assert tuple(batch['mask'].shape[1:]) == (512, 512)
            rows.extend(zip(batch['crop_top'].tolist(), batch['crop_left'].tolist()))
        crop_runs.append(rows)
    assert crop_runs[0] == crop_runs[2], 'same epoch must reproduce its crop'
    assert crop_runs[0] != crop_runs[1], 'worker crops repeated across epochs'
    print('WORKER-CROPS epochs0/1:', crop_runs[:2], flush=True)
    mask = torch.zeros(1, 13, 17, dtype=torch.long)
    mask[0, 2:11, 3:15] = 1
    mask[0, 0, 0] = 1
    for spacing in (2, 4):
        actual = connectivity_targets(mask, spacing).numpy()
        reference = np.zeros((1, 9, 13, 17), dtype=np.float32)
        for y in range(13):
            for x in range(17):
                for channel, (dy, dx) in enumerate((
                        (dy, dx) for dy in (-spacing, 0, spacing) for dx in (-spacing, 0, spacing))):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < 13 and 0 <= xx < 17:
                        reference[0, channel, y, x] = mask[0, y, x] * mask[0, yy, xx]
        assert np.array_equal(actual, reference)
    print('TARGETS spacing2/4 and boundary convention match reference', flush=True)
    class PointwiseModel(torch.nn.Module):
        def forward(self, x):
            logits = torch.cat((torch.zeros_like(x[:, :1]), x[:, :1]), dim=1)
            near = torch.full((x.shape[0], 9, *x.shape[-2:]), 0.25, device=x.device)
            far = torch.full_like(near, -0.25)
            return logits, near, far
    image = np.random.RandomState(3).randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
    components = predict_full(PointwiseModel(), image, 'cpu', tta=True)
    expected = image_tensor(image)[0].sigmoid().numpy()
    np.testing.assert_allclose(components[0], expected, atol=2e-7)
    np.testing.assert_allclose(components[1], 2.25, atol=1e-7)
    np.testing.assert_allclose(components[2], -2.25, atol=1e-7)
    assert np.array_equal(binary_prediction(components, 0.5, 'source_fusion'), expected >= 0.5)
    print('SLIDING native1024 tile512 stride256 four-flip restoration passed', flush=True)
    model = create_model('b2', args.pretrain_ckpt).cuda().eval()
    with torch.no_grad():
        outputs = model(torch.randn(1, 3, 512, 512, device='cuda'))
        assert [tuple(value.shape) for value in outputs] == [(1, 2, 512, 512),
                                                            (1, 9, 512, 512), (1, 9, 512, 512)]
        assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in outputs)
    del outputs
    model.train()
    outputs = model(torch.randn(1, 3, 128, 128, device='cuda'))
    target = torch.randint(0, 2, (1, 128, 128), device='cuda')
    loss, _ = source_loss(outputs, target, torch.tensor([1., 3.], device='cuda'))
    loss.backward()
    parameters = dict(model.named_parameters())
    names = ['backbone.patch_embed1.dcn.conv.weight', 'backbone.block1.0.attn.q_offset.weight',
             'decode_head.con.seg_branch.3.weight', 'decode_head.con.connect_branch.2.weight',
             'decode_head.con.connect_branch_d1.2.weight']
    for name in names:
        grad = parameters[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    print('MODEL full512 FP32 forward; 128 backward DCN/offset/surface/both-connection gradients passed', flush=True)
    print(json.dumps({'loss': float(loss.detach()), 'peak_cuda_memory_mb':
                      torch.cuda.max_memory_allocated() / 1024**2}), flush=True)


if __name__ == '__main__':
    main()
