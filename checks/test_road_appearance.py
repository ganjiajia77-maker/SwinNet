"""Geometry, donor safety, RNG isolation, label invariance and frozen forward."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from road_appearance import AppearanceConfig,apply_appearance,independent_rng,AppearanceStats
from road_appearance_dataset import RoadAppearanceDataset


def scene(size=512):
    yy,xx=np.indices((size,size));image=np.stack(((3*xx+yy)%256,(xx+2*yy)%256,(2*xx+3*yy)%256),2).astype(np.uint8)
    road=np.zeros((size,size),bool);road[size//2-3:size//2+4,10:size-10]=True
    skeleton=np.zeros_like(road);skeleton[size//2,10:size-10]=True
    return image,road,skeleton


class AppearanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_exclusive_request_probabilities_and_finite_skips(self):
        image=np.zeros((16,16,3),np.uint8);target=np.zeros((16,16),bool);rng=independent_rng(1234,1,'distribution')
        stats=AppearanceStats();config=AppearanceConfig(enabled=True)
        for _ in range(10000):
            output,info,_=apply_appearance(image,target,target,rng,config);stats.add(info)
            self.assertTrue(np.array_equal(output,image));self.assertLessEqual(info['placement_attempts'],10)
        report=stats.report();counts=report['counts']
        self.assertAlmostEqual(counts['clean:none:requested']/10000,.5,delta=.02)
        self.assertAlmostEqual(sum(counts.get('shadow:'+t+':requested',0) for t in ('road','background'))/10000,.25,delta=.02)
        self.assertAlmostEqual(report['road_requested_fraction_all_tiles'],.4,delta=.02)
        self.assertAlmostEqual(report['background_requested_fraction_all_tiles'],.1,delta=.02)
        self.assertEqual(report['successful_fraction_all_tiles'],0)

    def test_shadow_only_darkens_rectangle_and_preserves_labels(self):
        image,road,skeleton=scene();before=road.copy();skel_before=skeleton.copy()
        out,info,mask=apply_appearance(image,road,skeleton,independent_rng(2,1,'shadow'),AppearanceConfig(enabled=True),force=('shadow','road'),debug=True)
        self.assertTrue(info['success'],info);self.assertTrue(np.all(out<=image))
        self.assertTrue(np.array_equal(out[mask==0],image[mask==0]))
        self.assertTrue(np.array_equal(road,before));self.assertTrue(np.array_equal(skeleton,skel_before))
        self.assertGreaterEqual(info['alpha'],.35);self.assertLessEqual(info['alpha'],.75)
        self.assertGreaterEqual(info['length'],16);self.assertLessEqual(info['length'],64)
        self.assertGreaterEqual(info['width'],16);self.assertLessEqual(info['width'],96)
        self.assertGreaterEqual(info['length'],info['width'])
        self.assertGreater((mask==1).sum(),0)

    def test_texture_real_same_tile_donor_and_background_constraint(self):
        image,road,skeleton=scene();config=AppearanceConfig(enabled=True)
        for target in ('road','background'):
            out,info,mask=apply_appearance(image,road,skeleton,independent_rng(4,1,target),config,force=('texture',target),debug=True)
            self.assertTrue(info['success'],info)
            self.assertLessEqual(info['donor_road_fraction'],.01);self.assertLessEqual(info['donor_attempts'],10)
            if target=='background':self.assertLessEqual(info['target_road_fraction'],.05)
            x0,y0,x1,y1=info['donor_bbox'];dy,dx=np.nonzero(mask>0)
            ty0,tx0,ty1,tx1=dy.min(),dx.min(),dy.max()+1,dx.max()+1
            self.assertFalse(x0<tx1 and x1>tx0 and y0<ty1 and y1>ty0)
            local=mask[ty0:ty1,tx0:tx1];full=local>=1
            self.assertTrue(np.array_equal(out[ty0:ty1,tx0:tx1][full],image[y0:y1,x0:x1][full]))
            self.assertTrue(np.array_equal(out[mask==0],image[mask==0]))

    def test_no_black_or_constant_fallback_without_donor(self):
        image,road,skeleton=scene();image[:]=120
        out,info,mask=apply_appearance(image,road,skeleton,independent_rng(4,1,'empty-donor'),AppearanceConfig(enabled=True),force=('texture','road'),debug=True)
        self.assertFalse(info['success']);self.assertEqual(info['skip_reason'],'no_nonroad_texture_donor')
        self.assertTrue(np.array_equal(out,image));self.assertEqual(float(mask.sum()),0)

    def test_independent_seed_replays_without_global_rng_change(self):
        image,road,skeleton=scene();np.random.seed(77);before=np.random.get_state()
        outputs=[]
        for _ in range(2):outputs.append(apply_appearance(image,road,skeleton,independent_rng(5,2,'same'),AppearanceConfig(enabled=True),force=('shadow','road'),debug=True))
        after=np.random.get_state();self.assertTrue(np.array_equal(before[1],after[1]));self.assertEqual(before[2:],after[2:])
        self.assertTrue(np.array_equal(outputs[0][0],outputs[1][0]));self.assertTrue(np.array_equal(outputs[0][2],outputs[1][2]))
        self.assertNotEqual(independent_rng(5,2,'same').random(),independent_rng(5,3,'same').random())

    def test_crop_first_dataset_switch_cannot_change_base_input_or_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'train').mkdir();(root/'train_labels').mkdir()
            yy,xx=np.indices((1500,1500));image=np.stack(((xx+yy)%256,(2*xx+yy)%256,(xx+3*yy)%256),2).astype(np.uint8)
            road=np.zeros((1500,1500),np.uint8)
            for y in range(50,1450,100):road[y-3:y+4,20:1480]=255
            cv2.imwrite(str(root/'train/case.tif'),image);cv2.imwrite(str(root/'train_labels/case.tif'),road)
            off=RoadAppearanceDataset(root,1234,AppearanceConfig(enabled=False));on=RoadAppearanceDataset(root,1234,AppearanceConfig(enabled=True))
            off.set_epoch(3);on.set_epoch(3)
            a,da=off.sample(0,debug=True);b,db=on.sample(0,force=('shadow','road'),debug=True)
            for key in ('mask','skeleton','skeleton_dilate'):self.assertTrue(torch.equal(a[key],b[key]))
            for key in ('tile_top','tile_left'):self.assertEqual(a[key],b[key])
            for key in ('before_local_sha256','base_geometry','road_sha256','skeleton_sha256'):self.assertEqual(da['metadata'][key],db['metadata'][key])
            self.assertTrue(np.array_equal(da['before'],db['before']));self.assertEqual(tuple(b['image'].shape),(3,512,512))
            self.assertNotIn('appearance_mask',b)
            from export_road_appearance import export_examples
            report=export_examples(root,root/'preview',seed=1234,epoch=3,max_samples=200)
            self.assertEqual(len(report['samples']),24);self.assertEqual(report['counts']['road_shadow'],8)

    def test_frozen_forward_reads_perturbed_image_and_loss_has_no_choice(self):
        from checks.test_v9_state_gate import model as fake_v9
        from networks.v9_source_selector import SourceSelectedRepair
        from networks.v9_selector_transport_training import configure_joint
        from networks.v9_source_relation_labels import RelationLabels
        import run_v9_road_appearance as runner
        image,road,skeleton=scene(128)
        out,info,_=apply_appearance(image,road,skeleton,independent_rng(21,1,'model'),AppearanceConfig(enabled=True),force=('shadow','road'))
        self.assertTrue(info['success'],info)
        model=SourceSelectedRepair(fake_v9());optimizer,_=configure_joint(model,{k:1e-4 for k in ('visual','reliable_head','relations','proposal_head','accept_head','rates','gate_relation')},1e-4)
        model.train();inputs=[];hook=model.original.frozen.register_forward_pre_hook(lambda module,values:inputs.append(values[0]))
        tensor=torch.from_numpy(out.transpose(2,0,1).copy()).float()[None]/255.
        result=model(tensor,mode='source');hook.remove()
        self.assertTrue(torch.equal(inputs[0],tensor));self.assertFalse(model.original.frozen.training)
        args=SimpleNamespace(source_losses={'lambda_cont':.2},labels=RelationLabels(),config={'lambda_relation':.1,'lambda_choice':0})
        with patch('networks.v9_late_choice_loss.late_choice_loss',side_effect=AssertionError('L_choice must not run')):
            total,terms=runner.training_loss(args,result,torch.from_numpy(road.astype(np.float32))[None,None],torch.from_numpy(skeleton.astype(np.float32))[None,None],1234)
            total.backward()
        self.assertTrue(all(p.grad is None for p in model.original.frozen.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.selector.parameters() if p.grad is not None),0)


if __name__=='__main__':unittest.main()
