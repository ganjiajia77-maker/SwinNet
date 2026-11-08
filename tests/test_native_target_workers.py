import random
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from datasets.dataset_road_skeleton import RoadSkeletonDataset
from datasets.native_relation_targets import native_targets
from datasets.native_relation_batch import pack_native_sample, verify_native_batch
from losses.road_losses import SurfaceStructureLoss


def synthetic():
    road = np.zeros((71, 75), np.float32)
    road[21:25, 3:70] = 1
    road[5:65, 38:41] = 1
    road[44:46, 6:34] = 1
    road[48:50, 6:34] = 1
    road[56, 56] = road[59, 59] = 1
    skel = np.zeros_like(road)
    skel[22, 3:70] = skel[5:65, 39] = 1
    skel[44, 6:34] = skel[48, 6:34] = 1
    return road[3:67, 5:69].copy(), skel[3:67, 5:69].copy()


@pytest.mark.parametrize('rotation', range(4))
@pytest.mark.parametrize('flip', [False, True])
def test_all_elements_and_rng_unchanged(rotation, flip):
    road, skel = synthetic()
    road, skel = np.rot90(road, rotation).copy(), np.rot90(skel, rotation).copy()
    if flip:
        road, skel = road[:, ::-1].copy(), skel[:, ::-1].copy()
    mask, skeleton = torch.from_numpy(road)[None], torch.from_numpy(skel)[None]
    random.seed(12); np.random.seed(12); torch.manual_seed(12)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    targets, stats = pack_native_sample(mask, skeleton)
    after = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    assert before[0] == after[0]
    assert all(np.array_equal(a, b) for a, b in zip(before[1], after[1]))
    assert torch.equal(before[2], after[2])
    ref = native_targets(road > .5, skel > .5)
    for key, value in targets.items():
        expected = ref[key][None] if key == 'direction_valid' else ref[key]
        assert np.array_equal(value.numpy(), expected)
    assert stats['ignored_edges'] == int((~ref['connectivity_valid']).sum())


class FinalFlip:
    def __call__(self, sample):
        for key in ('image', 'mask', 'skeleton', 'skeleton_dilate'):
            sample[key] = sample[key].flip(-1)
        return sample


def test_spawn_workers_use_final_augmented_crop_and_same_images(tmp_path):
    (tmp_path/'train'/'image').mkdir(parents=True)
    (tmp_path/'train'/'mask').mkdir()
    for i in range(4):
        road, _ = synthetic()
        rgb = np.random.RandomState(i).randint(0, 256, (64, 64, 3), dtype=np.uint8)
        assert cv2.imwrite(str(tmp_path/'train'/'image'/f'{i}.png'), rgb)
        assert cv2.imwrite(str(tmp_path/'train'/'mask'/f'{i}.png'), (road*255).astype(np.uint8))
    opts = dict(root_dir=str(tmp_path), split='train', image_size=32,
                source_patch_size=64, random_crop_train=True, augment=True,
                random_crop_seed=1234, transform=FinalFlip())
    old = RoadSkeletonDataset(**opts)
    new = RoadSkeletonDataset(**opts, native_relations=True)
    old.set_epoch(3); new.set_epoch(3)
    reference = [old[i] for i in range(len(old))]
    loader = DataLoader(new, batch_size=2, num_workers=2, multiprocessing_context='spawn')
    for i, batch in enumerate(loader):
        original = default_collate(reference[2*i:2*i+2])
        for key in ('image', 'mask', 'skeleton', 'skeleton_dilate', 'tile_top', 'tile_left'):
            assert torch.equal(batch[key], original[key]), key
        assert verify_native_batch(batch, original['mask'], original['skeleton'])['passed']


@pytest.mark.parametrize('mode', ['off', 'c_guided'])
def test_total_loss_and_all_prediction_gradients_are_bitwise_identical(mode):
    torch.set_num_threads(2)
    road, skel = synthetic()
    mask, skeleton = torch.from_numpy(road)[None, None], torch.from_numpy(skel)[None, None]
    targets, stats = pack_native_sample(mask[0], skeleton[0])
    batch = default_collate([dict(native_relations=targets, native_relation_stats=stats)])
    criterion = SurfaceStructureLoss(stage_structure_weights=(0., 0., .008, .012),
                                     stage_direction_factor=.2, stage_connectivity_factor=3.)
    criterion.native_relation_enabled = True
    criterion.local_cs_mode = mode; criterion.local_cs_weight = .05
    torch.manual_seed(13)
    values = [torch.randn(1, 1, 64, 64)]
    for _ in range(2):
        values.extend([torch.randn(1, 1, 16, 16), torch.randn(1, 8, 16, 16), torch.randn(1, 2, 16, 16)])

    def evaluate(precomputed):
        leaves = [v.clone().requires_grad_() for v in values]
        stages = [dict(stage=i+2, skeleton=leaves[1+3*i],
                       connectivity=leaves[2+3*i], direction=leaves[3+3*i]) for i in range(2)]
        kwargs = dict(native_relations=batch['native_relations'],
                      native_relation_stats=batch['native_relation_stats']) if precomputed else {}
        loss, info = criterion(leaves[0], surface_gt=mask, skeleton_gt=skeleton,
                               skeleton_dilate_gt=skeleton, stage_outputs=stages, **kwargs)
        gradients = torch.autograd.grad(loss, leaves)
        return loss.detach(), info, gradients, criterion.last_native_stats.copy()

    legacy = evaluate(False)
    # Optimized loss must not call the CPU label generator at all.
    with patch('datasets.native_relation_batch.native_targets', side_effect=AssertionError('CPU generation in loss')):
        worker = evaluate(True)
    assert torch.equal(legacy[0], worker[0])
    assert legacy[3] == worker[3]
    for key in legacy[1]:
        torch.testing.assert_close(legacy[1][key], worker[1][key], rtol=0, atol=0,
                                   equal_nan=True, msg=key)
    for a, b in zip(legacy[2], worker[2]):
        assert torch.equal(a, b)
