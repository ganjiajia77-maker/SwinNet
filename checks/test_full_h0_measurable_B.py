"""Production budget initialization and reuse of an immutable completed A."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np
import torch

from checks.test_full_h0_joint import arguments, make_source
from networks.full_h0_joint import FullH0Joint, build_h0, original_criterion
import compare_full_h0_joint as comparison
import run_full_h0_joint as runner


ROOT=Path(__file__).resolve().parents[1]


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def completed_A(pair):
    config={'variant':'A','h0_sha256':'base','epochs':20,'seed':1234,'root_path':'dataset',
            'batch_size':2,'num_workers':4,'val_interval':5,'h0_lr':5e-5,'selection_rule':'same rule',
            'reference_sha256':'reference','code_sha256':{'reference_h0_599a410/model.py':'original'}}
    write_json(pair/'A/experiment.json',config)
    write_json(pair/'A/training_complete.json',{'epochs':20})
    (pair/'A/best.pth').write_bytes(b'preserve original A')
    write_json(pair/'A/eval/val_selection.json',{'threshold':.5,
        'checkpoint_sha256':runner.digest(pair/'A/best.pth')})
    for epoch in range(1,21):
        write_json(pair/'A'/f'epoch_{epoch:03d}.json',
                   {'epoch':epoch,'case_order_sha256':str(epoch),'augmented_images_sha256':str(epoch*2)})
    (pair/'A/eval/val').mkdir(parents=True,exist_ok=True)
    for index in range(14):
        gt=np.zeros((4,4),dtype=bool);gt[1,1]=True
        p=np.where(gt,.7,.1).astype(np.float32)
        np.savez_compressed(pair/'A/eval'/f'val/{index}.npz',gt=gt,p0=p,p1=p,current_p0=p)
    return dict(config,variant='B')


class ReuseTests(unittest.TestCase):
    def test_completed_A_verified_without_changes_and_mismatches_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='.reuse_A_') as folder:
            pair=Path(folder);config=completed_A(pair)
            before={str(path.relative_to(pair)):path.read_bytes() for path in (pair/'A').rglob('*') if path.is_file()}
            runner.check_reusable_A(pair,config)
            with self.assertRaises(ValueError):runner.check_reusable_A(pair,dict(config,seed=7))
            bad=dict(config,code_sha256={'reference_h0_599a410/model.py':'changed'})
            with self.assertRaises(ValueError):runner.check_reusable_A(pair,bad)
            after={str(path.relative_to(pair)):path.read_bytes() for path in (pair/'A').rglob('*') if path.is_file()}
            self.assertEqual(before,after)

    def test_comparison_reads_new_B_and_preserves_old_results(self):
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='.compare_new_B_') as folder:
            pair=Path(folder);config=completed_A(pair);tag='B_rates01'
            new=pair/tag
            write_json(new/'experiment.json',dict(config,rate_initialization='explicit_softmax_budgets',
                initial_transport_rate=.01,initial_proposal_rate=.001))
            write_json(pair/'reference/fixed_gaps_and_anchors.json',{'count':0})
            (new/'best.pth').write_bytes(b'new B')
            write_json(new/'eval/val_selection.json',{'threshold':.5,'checkpoint_sha256':runner.digest(new/'best.pth')})
            for epoch in range(1,21):
                write_json(new/f'epoch_{epoch:03d}.json',json.loads((pair/'A'/f'epoch_{epoch:03d}.json').read_text()))
            (new/'eval/val').mkdir(parents=True,exist_ok=True)
            for path in (pair/'A/eval/val').glob('*.npz'):
                (new/'eval/val'/path.name).write_bytes(path.read_bytes())
            (pair/'comparison').mkdir();(pair/'comparison/REPORT.md').write_text('old comparison')
            (pair/'B').mkdir();(pair/'B/best.pth').write_bytes(b'old B')
            def point(cases,name,threshold,manifest):
                metrics={key:.5 for key in ('iou','f1','precision','recall','cldice',
                    'apls_gt_to_pred_approx','apls_pred_to_gt_approx','apls_bidirectional_approx','short_pred_components')}
                return {'threshold':threshold,'metrics':metrics,'closed':0,'bridge_proxy':0}
            with patch.object(comparison,'point_metrics',side_effect=point):
                comparison.compare(pair,tag)
            self.assertTrue((pair/('comparison_'+tag)/'REPORT.md').is_file())
            self.assertEqual((pair/'comparison/REPORT.md').read_text(),'old comparison')
            self.assertEqual((pair/'B/best.pth').read_bytes(),b'old B')
            self.assertEqual((pair/'A/best.pth').read_bytes(),b'preserve original A')


class MeasurableSmokeTests(unittest.TestCase):
    def test_real_model_production_initialization_and_training_smoke(self):
        torch.set_num_threads(2);torch.manual_seed(1234)
        base,saved=build_h0(arguments(),128)
        source,q=make_source()
        model=FullH0Joint(base,q,source,transport_rate_init=.01,proposal_rate_init=.001)
        rates=model.original.dynamics.rates[-1].bias.detach().softmax(0)
        self.assertAlmostEqual(float(rates[1]),.01,places=7)
        self.assertAlmostEqual(float(rates[2]),.001,places=7)
        for name,value in model.selector.state_dict().items():
            torch.testing.assert_close(value,source['selector'][name],atol=0,rtol=0)
        image=torch.randn(2,3,128,128)
        mask=torch.zeros(2,1,128,128);mask[:,:,47:79,4:124]=1
        skeleton=torch.zeros_like(mask);skeleton[:,:,63,4:124]=1
        batch={'image':image,'mask':mask,'skeleton':skeleton,
               'skeleton_dilate':torch.nn.functional.max_pool2d(skeleton,7,1,3)}
        class Dataset(torch.utils.data.Dataset):
            def __len__(self):return 2
            def set_epoch(self,epoch):pass
            def __getitem__(self,index):return {key:value[index] for key,value in batch.items()}
        from dataclasses import asdict
        from networks.v9_source_relation_labels import RelationLabels
        config={'rate_initialization':'explicit_softmax_budgets','initial_transport_rate':.01,
                'initial_proposal_rate':.001,'lambda_relation':.1,
                'source_training':{'source_losses':{'lambda_cont':.2},'labels':asdict(RelationLabels())}}
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='.measurable_smoke_') as folder:
            args=SimpleNamespace(variant='B',seed=1234,device=torch.device('cpu'),output_dir=Path(folder),config=config)
            criterion=original_criterion(saved,'cpu')
            with patch.object(runner,'training_dataset',return_value=Dataset()),patch('builtins.print'):
                runner.smoke(args,model,criterion)
            report=json.loads((Path(folder)/'smoke.json').read_text())
            self.assertTrue(report['passed'])
            self.assertGreater(report['initial_transport_abs_mean'],0.)
            self.assertGreater(report['initial_P1_minus_P0']['mean_abs'],0.)
            self.assertGreater(report['final_segmentation_gradients']['selector'],0.)
            self.assertGreater(report['final_segmentation_gradients']['rates_output'],0.)


if __name__=='__main__':unittest.main()
