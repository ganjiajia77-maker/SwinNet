import argparse
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from datasets.dataset_road_skeleton import RoadSkeletonDataset
from h0_phase_training import inherit_h0_config, SAMPLING_PROTOCOL
from mass_protocol import crop_position
from scripts.check_h0_epoch_sampling import audit_workers, check_original_sources
from test_encoder_h_postmerge_phase import fake_reference


def make_data(tmp_path, mask_value=255):
    for folder in ('image', 'mask'):
        (tmp_path/'train'/folder).mkdir(parents=True)
    rng = np.random.RandomState(7)
    for index in range(2):
        image = rng.randint(0, 256, (80, 80, 3), dtype=np.uint8)
        mask = np.zeros((80, 80), np.uint8)
        mask[34:42, 4:76] = mask_value
        cv2.imwrite(str(tmp_path/'train'/'image'/f'{index}.png'), image)
        cv2.imwrite(str(tmp_path/'train'/'mask'/f'{index}.png'), mask)
    return RoadSkeletonDataset(str(tmp_path), 'train', image_size=32, source_patch_size=80,
        tile_size=32, tile_stride=16, augment=True, random_crop_train=True, random_crop_seed=1234)


def test_exact_original_sources_and_eighty_distinct_native_crop_positions():
    check_original_sources()
    for index in (0, 1, 553, 1107):
        points = [crop_position(1234, index, 0, epoch, 1500, 1500, 512) for epoch in range(80)]
        assert len(set(points)) == 80
        assert points == [crop_position(1234, index, 0, epoch, 1500, 1500, 512) for epoch in range(80)]


@pytest.mark.parametrize('persistent', [False, True])
def test_spawn_workers_match_original_direct_samples_and_change_each_epoch(tmp_path, persistent):
    torch.set_num_threads(1)
    dataset = make_data(tmp_path)
    rows = audit_workers(dataset, [0, 1], workers=2, persistent=persistent, context='spawn')
    assert len(rows) == 6
    for index in (0, 1):
        own = [row for row in rows if row['index'] == index]
        assert len({row['tensors_sha256'] for row in own}) == 3


def test_binary_zero_one_masks_match_zero_255_and_reproducible_augmentation(tmp_path):
    one = make_data(tmp_path/'one', 1)
    full = make_data(tmp_path/'full', 255)
    for epoch in (0, 1, 79):
        one.set_epoch(epoch); full.set_epoch(epoch)
        for key in ('image', 'mask', 'skeleton', 'skeleton_dilate'):
            assert torch.equal(one[0][key], full[0][key])
        assert one[0]['mask'].sum() > 0


def test_actual_reference_flag_is_overridden_and_explicitly_recorded(tmp_path):
    args, parser, path, saved = fake_reference(tmp_path)
    state = torch.load(path, weights_only=False)
    state['args']['disable_persistent_workers'] = False
    torch.save(state, path)
    parser.add_argument('--disable_persistent_workers', action='store_true')
    args.disable_persistent_workers = False
    report = inherit_h0_config(args, parser)
    assert args.disable_persistent_workers is True
    assert args.sampling_protocol == SAMPLING_PROTOCOL
    assert report['explicit_overrides']['disable_persistent_workers']['reference'] is False
    assert args.max_epochs == 80 and args.use_ema is False and args.warm_start_ckpt == ''
