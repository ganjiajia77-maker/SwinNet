import csv
import json
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn
from unittest.mock import patch
from checks.test_full_h0_joint import arguments,make_source
from networks.full_h0_joint import FullH0Joint,build_h0

import diagnostics.probe_h0_spatial_features as probe


class SpatialProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_full_channel_patch_preserves_spatial_order_and_boundary_padding(self):
        h=torch.zeros(1,64,128,128);g=torch.zeros(1,96,128,128)
        h[0,0,3:6,4:7]=torch.arange(9).reshape(3,3)
        g[0,0,3:6,4:7]=torch.arange(9).reshape(3,3)+20
        out=probe.patch_features(h,g,np.array([4,0]),np.array([5,0]))
        self.assertEqual(out.shape,(2,1440))
        np.testing.assert_array_equal(out[0,:9],np.arange(9))
        np.testing.assert_array_equal(out[0,64*9:64*9+9],np.arange(9)+20)
        self.assertEqual(float(out[1,0]),0.)

    def test_actual_joint_model_exposes_raw_h2_and_g3(self):
        base,_=build_h0(arguments(),128);source,q=make_source()
        model=FullH0Joint(base,q,source,transport_rate_init=.01,proposal_rate_init=.001).eval()
        captured=[]
        hook=model.original.upstream.register_forward_hook(lambda m,i,o:captured.append(o))
        try:
            with torch.no_grad():model(torch.randn(1,3,128,128))
        finally:hook.remove()
        self.assertEqual(len(captured),1)
        patches=probe.patch_features(captured[0]['detail'],captured[0]['semantic'],np.array([5]),np.array([7]))
        self.assertEqual(patches.shape,(1,1440));self.assertTrue(np.isfinite(patches).all())

    def test_crops_are_deterministic_and_extra_dedup_keeps_original(self):
        self.assertEqual(probe.extra_coordinates('case',123,0),probe.extra_coordinates('case',123,0))
        def row(ys):
            n=len(ys);item={key:np.zeros(n) for key in probe.FIELDS}
            item['features']=np.zeros((n,311));item['raw_patch']=np.zeros((n,1440))
            item['label']=np.ones(n);item['native_y']=np.asarray(ys);item['native_x']=np.zeros(n)
            return item
        out,removed=probe.append_nonduplicate(row([4]),[row([5,20])])
        self.assertEqual(removed,1);self.assertEqual(len(out['label']),2)
        np.testing.assert_array_equal(out['native_y'],[4,20])

    def test_cached_sample_reconstruction_uses_raw_upstream_and_original_projection(self):
        class Upstream(nn.Module):
            def forward(self,image):
                return {'detail':torch.ones(1,64,128,128),
                        'semantic':torch.full((1,96,128,128),2.)}
        class Toy(nn.Module):
            def __init__(self):
                super().__init__();self.param=nn.Parameter(torch.zeros(1))
                self.original=nn.Module();self.original.upstream=Upstream()
            def forward(self,image):
                self.original.upstream(image)
                return {'q':torch.full((1,1,128,128),.4),
                        'selector_context':{'feature':torch.full((1,32,128,128),.25)}}
        features=np.zeros((2,311),np.float32);features[:,4]=.4;features[:,5:37]=.25
        record={'label':np.ones(2),'tile_index':np.zeros(2),'gy':np.array([4,7]),
                'gx':np.array([5,8]),'features':features}
        sample={'case_name':'toy','image':torch.zeros(3,1500,1500)}
        patches,check=probe.extract_original(Toy(),sample,record,'fit',123)
        self.assertEqual(patches.shape,(2,1440));self.assertEqual(check['max_abs'],0.)
        self.assertEqual(float(patches[0,0]),1.);self.assertEqual(float(patches[0,64*9]),2.)
        record['features'][:,4]=.9
        with self.assertRaisesRegex(ValueError,'reconstruction mismatch'):
            probe.extract_original(Toy(),sample,record,'fit',123)

    def test_two_probes_and_unmatched_fp_are_reported(self):
        rng=np.random.default_rng(8);records=[]
        for name,split in [('a','fit'),('b','fit'),('c','hold'),('d','val'),('e','val')]:
            n=12;y=np.tile([1,0],6).astype(np.int8)
            nuisance=np.tile(np.array([.2,10,.3,65],np.float32),(n,1))
            features=rng.normal(0,.1,(n,311)).astype(np.float32);features[:,:4]=nuisance;features[:,5]=y
            raw=rng.normal(0,.1,(n,1440)).astype(np.float32);raw[:,0]=y
            neg=np.where(y,0,3);neg[-1]=4
            records.append({'case':name,'split':split,'features':features,'raw_patch':raw,'label':y,
                'nuisance':nuisance,'weight':np.ones(n,np.float32),'native_y':np.arange(n),
                'native_x':np.arange(n),'component':np.arange(n)+1,'length':np.full(n,8),
                'flank_proxy':np.zeros(n),'fixed_gap_id':y,'negative_kind':neg,
                'tile_index':np.zeros(n),'corridor_gt_fraction':np.full(n,.5,np.float32),
                'corridor_length':np.full(n,8,np.float32)})
        base,matching,results,images,fp,inc,positive=probe.evaluate_comparison(records,0)
        self.assertEqual(len(results),8);self.assertEqual(len(inc),4)
        self.assertTrue(any(r['category']=='bridge_proxy' and r['natural_sampled']>0 for r in fp))
        self.assertTrue(any(r['category']=='8_15' for r in positive))
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root,prefix='.spatial_probe_') as temp:
            out=Path(temp)
            identity={'extra_train_crops':3}
            probe.write_output(out,identity,records,matching,results,images,fp,inc,{'max_abs':0},base,positive)
            self.assertTrue((out/'REPORT.md').is_file())
            self.assertTrue((out/'false_positive_subgroups.csv').is_file())
            self.assertTrue((out/'matching_smd.csv').is_file())
            matched=np.zeros(len(base['label']),bool);matched[base['matched_indices']]=True
            path=out/'val_candidate_scores.csv'
            with path.open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['case','y','x','matched','label','sampling_weight','p0']);writer.writeheader()
                for i in np.flatnonzero(base['split']=='val'):
                    writer.writerow({'case':base['case'][i],'y':base['native_y'][i],
                                     'x':base['native_x'][i],'matched':matched[i],'label':base['label'][i],
                                     'sampling_weight':base['weight'][i],'p0':base['nuisance'][i,0]})
            self.assertEqual(probe.verify_original_validation_matches(out,base),24)
            with path.open('a',encoding='utf-8',newline='') as f:f.write('bogus,0,0,False,0,1,0.2\n')
            with self.assertRaisesRegex(ValueError,'differ'):
                probe.verify_original_validation_matches(out,base)

    def test_main_uses_previous_image_splits_and_reuses_cache(self):
        class Dataset:
            def __init__(self,names):self.image_files=[name+'.tif' for name in names]
            def __len__(self):return len(self.image_files)
            def __getitem__(self,i):
                name=Path(self.image_files[i]).stem
                value=(probe.stable_seed(1,name)%100000)/100000.
                return {'case_name':name,'image':torch.full((3,1500,1500),value),
                        'mask':torch.zeros(1,1500,1500)}
        train=Dataset(['a','b','c','d','e']);val=Dataset(['x','y'])
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root,prefix='.spatial_main_') as temp:
            pair=Path(temp)/'pair';reference=pair/'reference';reference.mkdir(parents=True)
            ref=reference/'fixed_gaps_and_anchors.json';ref.write_text('{}',encoding='utf-8')
            run=pair/'B_transport_0p01_proposal_0p001';run.mkdir()
            ckpt=run/'best.pth';torch.save({'config':{'reference_sha256':probe.digest(ref)}},ckpt)
            previous=Path(temp)/'previous';(previous/'cases').mkdir(parents=True)
            splits={'a':'fit','b':'fit','c':'fit','d':'fit','e':'hold'}
            identity={'script_sha256':probe.digest(root/'diagnostics/probe_full_h0_gap_features.py'),
                      'checkpoint_sha256':probe.digest(ckpt),'reference_sha256':probe.digest(ref),
                      'root_path':temp,'train_image_splits':splits,'val_image_names':['x','y'],'seed':1234}
            (previous/'identity.json').write_text(json.dumps(identity),encoding='utf-8')
            (previous/'summary.json').write_text('{"complete": true}',encoding='utf-8')
            all_records=[]
            for ds,split_names in ((train,splits),(val,{'x':'val','y':'val'})):
                for index,namefile in enumerate(ds.image_files):
                    name=Path(namefile).stem;split=split_names[name];sample=ds[index]
                    n=8;y=np.tile([1,0],4).astype(np.int8)
                    nuisance=np.tile(np.array([.2,10,.3,65],np.float32),(n,1))
                    feature=np.zeros((n,311),np.float32);feature[:,:4]=nuisance;feature[:,5]=y
                    row={'features':feature,'label':y,'weight':np.ones(n,np.float32),'nuisance':nuisance,
                         'native_y':np.arange(n),'native_x':np.arange(n),'component':np.arange(n)+1,
                         'length':np.full(n,8),'flank_proxy':np.zeros(n),'fixed_gap_id':y,
                         'negative_kind':1-y,'tile_index':np.zeros(n),'gy':np.arange(n),'gx':np.arange(n),
                         'corridor_gt_fraction':np.full(n,.5,np.float32),'corridor_length':np.full(n,8,np.float32),
                         'input_sha256':np.asarray(probe.signature(sample))}
                    np.savez_compressed(previous/'cases'/(split+'_'+name+'.npz'),**row)
                    all_records.append({'case':name,'split':split,'raw_patch':np.zeros((n,1440),np.float32),**row})
            combined,_=probe.combine(all_records);matched=np.zeros(len(combined['label']),bool)
            matched[combined['matched_indices']]=True
            with (previous/'val_candidate_scores.csv').open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['case','y','x','matched','label','sampling_weight','p0']);writer.writeheader()
                for i in np.flatnonzero(combined['split']=='val'):
                    writer.writerow({'case':combined['case'][i],'y':combined['native_y'][i],
                                     'x':combined['native_x'][i],'matched':matched[i],'label':combined['label'][i],
                                     'sampling_weight':combined['weight'][i],'p0':combined['nuisance'][i,0]})
            def fake_original(model,sample,record,split,seed):
                return np.zeros((len(record['label']),1440),np.float32),{'sampled_values':1,'max_abs':0.,'mean_abs':0.}
            def fake_extra(model,sample,name,seed,crop):
                row={key:all_records[0][key].copy() for key in probe.FIELDS}
                row['native_y']+=500;row['native_x']+=500
                return row,{'positive':4,'negative':4}
            out=Path(temp)/'output';argv=['audit','--pair_dir',str(pair),'--previous_probe',str(previous),
                 '--output_dir',str(out),'--device','cpu','--extra_train_crops','1','--bootstrap','0']
            with patch.object(sys,'argv',argv),patch.object(probe,'verify_recorded_code'), \
                 patch.object(probe,'full_dataset',side_effect=lambda root,split:train if split=='train' else val), \
                 patch.object(probe,'rebuild_model',return_value=nn.Linear(1,1)), \
                 patch.object(probe,'extract_original',side_effect=fake_original), \
                 patch.object(probe,'extra_crop',side_effect=fake_extra),patch('builtins.print'):
                probe.main()
            summary=json.loads((out/'summary.json').read_text())
            self.assertTrue(summary['complete']);self.assertEqual(summary['reconstruction']['previous_validation_candidates_verified'],16)
            with patch.object(sys,'argv',argv),patch.object(probe,'verify_recorded_code'), \
                 patch.object(probe,'full_dataset',side_effect=lambda root,split:train if split=='train' else val), \
                 patch.object(probe,'rebuild_model',return_value=nn.Linear(1,1)), \
                 patch.object(probe,'extract_original',side_effect=AssertionError('cache not reused')), \
                 patch.object(probe,'extra_crop',side_effect=AssertionError('cache not reused')),patch('builtins.print'):
                probe.main()


if __name__=='__main__':unittest.main()
