import unittest
import tempfile
from pathlib import Path
import json
import sys
from unittest.mock import patch
import numpy as np
import torch
from checks.test_full_h0_joint import arguments, make_source
from networks.full_h0_joint import FullH0Joint, build_h0

from diagnostics.probe_full_h0_gap_features import (binary_metrics,component_distances,features_from_result,
    make_candidates,match_image,tile_positions,fit_probe,GROUPS)
import diagnostics.probe_full_h0_gap_features as audit


class FrozenGapProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_two_component_distances_are_distinct_and_censored(self):
        mask=np.zeros((64,64),bool);mask[10:20,10:12]=True;mask[10:20,30:32]=True
        d1,d2,i1,i2,labels=component_distances(mask,limit=16,min_area=8)
        self.assertAlmostEqual(float(d1[15,20]),9.,places=3)
        self.assertAlmostEqual(float(d2[15,20]),10.,places=3)
        self.assertNotEqual(int(i1[15,20]),int(i2[15,20]))
        self.assertEqual(float(d2[50,50]),17.)

    def test_prediction_candidates_and_ground_truth_labels_are_separate(self):
        p0=np.full((512,512),.1,np.float32);p0[40:470,180:188]=.9;p0[40:470,300:308]=.9
        p1=p0.copy();p1[250:255,220:225]=.8
        gt=np.zeros((512,512),bool);gt[252:261,188:300]=True
        skeleton=np.zeros_like(gt);skeleton[256,188:300]=True
        result=make_candidates(p0,p1,gt,skeleton,seed=9)
        empty=make_candidates(p0,p1,np.zeros_like(gt),np.zeros_like(gt),seed=9)
        self.assertEqual(result['counts']['prediction_candidates'],empty['counts']['prediction_candidates'])
        self.assertGreater(result['counts']['positive_candidates'],0)
        self.assertGreater(result['counts']['negative_candidates'],0)
        self.assertEqual(empty['counts']['positive_candidates'],0)
        self.assertTrue(np.isfinite(result['corridor_gt_fraction']).any())
        self.assertTrue((result['weight']>=1).all())

    def test_matching_respects_all_calipers_without_reuse(self):
        nuisance=np.array([[.1,10,.2,20],[.2,15,.4,65],[.12,14,.25,30],[.21,16,.4,65]],np.float32)
        y=np.array([1,1,0,0])
        pairs=match_image(y,nuisance,seed=1)
        self.assertEqual(len(pairs),2)
        self.assertEqual(len(set(pairs[:,1])),2)
        self.assertTrue((np.abs(nuisance[pairs[:,0]]-nuisance[pairs[:,1]])<=np.array([.05,8,.15,16])+1e-6).all())
        self.assertTrue(((nuisance[pairs[:,0],3]<=64)==(nuisance[pairs[:,1],3]<=64)).all())

    def test_metrics_ties_and_weighted_prevalence(self):
        y=np.array([1,0,1,0]);score=np.array([1.,1.,1.,1.]);weights=np.array([2.,1.,2.,1.])
        m=binary_metrics(y,score,weights)
        self.assertAlmostEqual(m['roc_auc'],.5)
        self.assertAlmostEqual(m['pr_auc'],2/3)
        self.assertAlmostEqual(m['weighted_positive_fraction'],2/3)

    def test_feature_shapes_and_non_overlapping_cores(self):
        sample={'gy':np.array([5,10]),'gx':np.array([7,11]),'nuisance':np.zeros((2,4),np.float32)}
        result={'selector_context':{'feature':torch.zeros(1,32,128,128),'pair':torch.zeros(1,8,32,128,128)},
                'q':torch.zeros(1,1,128,128),'r':torch.zeros(1,1,128,128),'w':torch.zeros(1,8,128,128),
                'source_choice':torch.zeros(1,3,9,128,128)}
        x=features_from_result(result,sample)
        self.assertEqual(x.shape,(2,311))
        self.assertEqual([len(GROUPS[k]) for k in GROUPS],[4,5,21,37,279,311])
        owners=np.zeros((1500,1500),np.uint8)
        for top,left,y0,y1,x0,x1 in tile_positions():owners[y0:y1,x0:x1]+=1
        self.assertTrue((owners==1).all())

    def test_probe_fits_without_validation_data(self):
        rng=np.random.default_rng(3)
        y=np.tile([0,1],40).astype(np.int8)
        x=np.column_stack((y+rng.normal(0,.1,len(y)),rng.normal(size=len(y)))).astype(np.float32)
        for kind in ('linear','mlp'):
            pred,info=fit_probe(x[:60],y[:60],x[60:],y[60:],kind,1)
            self.assertGreater(binary_metrics(y[60:],pred(x[60:]))['roc_auc'],.95)
            self.assertGreater(info['best_step'],0)

    def test_reports_use_same_samples_and_image_disjoint_holdout(self):
        rng=np.random.default_rng(7);records=[]
        for name,split in [('a','fit'),('b','fit'),('c','hold'),('d','val'),('e','val')]:
            n=12;y=np.tile([1,0],6).astype(np.int8)
            nuisance=np.tile(np.array([.2,10,.3,65],np.float32),(n,1))
            features=rng.normal(0,.1,(n,311)).astype(np.float32);features[:,:4]=nuisance
            features[:,5]=y+rng.normal(0,.05,n)
            records.append({'case':name,'split':split,'features':features,'label':y,
                'nuisance':nuisance,'weight':np.ones(n,np.float32),'native_y':np.arange(n),
                'native_x':np.arange(n),'component':np.arange(n)+1,'length':np.full(n,8),
                'flank_proxy':np.zeros(n),'fixed_gap_id':y,'negative_kind':1-y,
                'tile_index':np.zeros(n),'corridor_gt_fraction':np.full(n,.5,np.float32),
                'corridor_length':np.full(n,8,np.float32),'counts_json':np.asarray('[]')})
        data,matching=audit.combine(records)
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root,prefix='.gap_probe_') as temp:
            out=Path(temp)
            results,per_image,subgroups,scores,predictions,masks,gaps=audit.evaluate(data,out,0)
            identity={'train_images_used':3,'val_images':2,'fixed_gaps':198,'bootstrap':0}
            audit.write_report(out,data,records,matching,results,per_image,subgroups,scores,identity,predictions,masks,gaps)
            self.assertTrue((out/'REPORT.md').is_file())
            self.assertTrue((out/'increment_vs_base.csv').is_file())
            self.assertTrue((out/'gap_detection_proxy.csv').is_file())
            self.assertEqual(len([r for r in results if r['classifier']=='linear']),12)

    def test_actual_joint_forward_feature_layout(self):
        torch.manual_seed(12)
        base,_=build_h0(arguments(),128);source,q=make_source()
        model=FullH0Joint(base,q,source,transport_rate_init=.01,proposal_rate_init=.001).eval()
        with torch.no_grad():result=model(torch.randn(1,3,128,128))
        sample={'gy':np.array([5,10]),'gx':np.array([7,11]),'nuisance':np.zeros((2,4),np.float32)}
        features=features_from_result(result,sample)
        self.assertEqual(features.shape,(2,311))
        self.assertTrue(np.isfinite(features).all())

    def test_main_recovers_from_saved_image_cache(self):
        class Dataset:
            def __init__(self,names):self.image_files=[name+'.tif' for name in names]
            def __len__(self):return len(self.image_files)
            def __getitem__(self,index):
                name=Path(self.image_files[index]).stem
                value=(audit.stable_seed(1,name)%100000)/100000.
                return {'case_name':name,'image':torch.full((3,1500,1500),value),
                        'mask':torch.zeros(1,1500,1500)}
        train=Dataset(['train_a','train_b','train_c','train_d','train_e'])
        val=Dataset(['val_x','val_y'])
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root,prefix='.gap_main_') as temp:
            pair=Path(temp)/'pair';reference=pair/'reference';reference.mkdir(parents=True)
            run=pair/'B_transport_0p01_proposal_0p001';run.mkdir()
            manifest={'count':2,'images':{name:{'shape':[1500,1500],'gaps':[{'id':1,'pixels':[0]}]}
                                         for name in ['val_x','val_y']}}
            path=reference/'fixed_gaps_and_anchors.json';path.write_text(json.dumps(manifest),encoding='utf-8')
            for name in manifest['images']:np.savez_compressed(reference/(name+'.npz'),gt=np.zeros((1500,1500),bool))
            torch.save({'config':{'reference_sha256':audit.digest(path),'root_path':temp,'seed':1}},run/'best.pth')
            def fake_extract(model,sample,split,seed,reference):
                n=12;y=np.tile([1,0],6).astype(np.int8);nuisance=np.tile(np.array([.2,10,.3,65],np.float32),(n,1))
                features=np.zeros((n,311),np.float32);features[:,:4]=nuisance;features[:,5]=y
                return {'features':features,'label':y,'weight':np.ones(n,np.float32),'nuisance':nuisance,
                        'native_y':np.arange(n),'native_x':np.arange(n),'component':np.arange(n)+1,
                        'length':np.full(n,8),'flank_proxy':np.zeros(n),'fixed_gap_id':y,
                        'negative_kind':1-y,'tile_index':np.zeros(n),
                        'corridor_gt_fraction':np.full(n,.5,np.float32),'corridor_length':np.full(n,8,np.float32),
                        'counts_json':np.asarray('[]'),'input_sha256':np.asarray(audit.signature(sample)),
                        'p0_map':np.empty((0,0),np.float16)}
            out=Path(temp)/'out'
            argv=['audit','--pair_dir',str(pair),'--output_dir',str(out),'--device','cpu',
                  '--train_images','5','--expected_images','2','--expected_gaps','2','--bootstrap','0']
            with patch.object(sys,'argv',argv),patch.object(audit,'verify_recorded_code'), \
                 patch.object(audit,'full_dataset',side_effect=lambda root,split:train if split=='train' else val), \
                 patch.object(audit,'rebuild_model',return_value=torch.nn.Linear(1,1)), \
                 patch.object(audit,'extract_image',side_effect=fake_extract), \
                 patch.object(audit,'make_visualizations'),patch('builtins.print'):
                audit.main()
            self.assertTrue(json.loads((out/'summary.json').read_text())['complete'])
            with patch.object(sys,'argv',argv),patch.object(audit,'verify_recorded_code'), \
                 patch.object(audit,'full_dataset',side_effect=lambda root,split:train if split=='train' else val), \
                 patch.object(audit,'rebuild_model',return_value=torch.nn.Linear(1,1)), \
                 patch.object(audit,'extract_image',side_effect=AssertionError('cache was not used')), \
                 patch.object(audit,'make_visualizations'),patch('builtins.print'):
                audit.main()


if __name__=='__main__':unittest.main()
