"""Paired prediction audit against known synthetic cases and frozen anchors."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

from diagnostics import audit_full_h0_saved_predictions as audit
from selector_transport_reference import bridge_proxy, closed_gaps


ROOT=Path(__file__).resolve().parents[1]


def json_file(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data),encoding='utf-8')


def fixture(pair):
    tag='B_transport_0p01_proposal_0p001'
    shape=(64,64)
    gt=np.zeros(shape,dtype=bool);gt[32,5:55]=True
    old=gt.copy();old[32,20:30]=False
    # Two old background pieces allow a genuine strict bridge proxy test.
    old[43,5:14]=True;old[43,25:34]=True
    b0=old.copy();b0[8,8]=True
    b1=b0.copy();b1[32,20:30]=True;b1[43,14:25]=True
    skeleton_pixels=[32*64+x for x in range(20,30)]
    anchors=[[32*64+x for x in range(17,20)],[32*64+x for x in range(30,33)]]
    gap={'id':7,'length':10,'length_bin':'8-15','pixels':skeleton_pixels,'anchors':anchors}
    manifest={'count':1,'p0_reference_threshold':.45,'images':{'image_0':{'shape':list(shape),'gaps':[gap]},
                                                                    'image_1':{'shape':list(shape),'gaps':[]}}}
    manifest_path=pair/'reference/fixed_gaps_and_anchors.json'
    json_file(manifest_path,manifest);sha=audit.digest(manifest_path)
    config={'h0_sha256':'h0','epochs':2,'seed':1234,'root_path':'fixture','batch_size':2,'num_workers':4,
            'h0_lr':5e-5,'val_interval':5,'selection_rule':'rule','reference_sha256':sha,
            'code_sha256':{'reference_h0_599a410/a.py':'same'}}
    for variant in ('A',tag):
        run=pair/variant
        json_file(run/'experiment.json',dict(config,variant='A' if variant=='A' else 'B'))
        (run/'best.pth').write_bytes((variant+'weights').encode())
        json_file(run/'eval/val_selection.json',{'threshold':.5,'reference_sha256':sha,
                                                  'checkpoint_sha256':audit.digest(run/'best.pth')})
        for epoch in (1,2):
            json_file(run/f'epoch_{epoch:03d}.json',{'case_order_sha256':str(epoch),
                                                      'augmented_images_sha256':str(epoch+100)})
        (run/'eval/val').mkdir()
        for index in (0,1):
            prob=lambda value: np.where(value,.9,.1).astype(np.float32)
            np.savez_compressed(run/'eval/val'/f'image_{index}.npz',gt=gt,p0=prob(old),
                current_p0=prob(old if variant=='A' else b0),p1=prob(old if variant=='A' else b1))
    return tag,gt,old,b0,b1,gap


class PairAuditTests(unittest.TestCase):
    def test_inherited_global_and_local_closure_distinguish_distant_detour(self):
        shape=(64,64);pred=np.zeros(shape,dtype=bool)
        pred[32,5:55]=True;pred[32,25]=False
        pred[5:33,5]=True;pred[5:33,54]=True;pred[5,5:55]=True
        gap={'id':1,'pixels':[32*64+x for x in range(20,30)],
             'anchors':[[32*64+x for x in range(17,20)],[32*64+x for x in range(30,33)]]}
        entry={'shape':list(shape),'gaps':[dict(gap,length_bin='8-15')]}
        _,labels,_,_=audit.component_data(pred)
        state=audit.gap_state(pred,labels,gap,shape)
        self.assertTrue(state['global_closed_proxy'])
        self.assertFalse(state['local_closed_proxy'])
        self.assertEqual(closed_gaps(pred,entry)[1],[1])

    def test_strict_bridge_matches_existing_definition(self):
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='.pair_bridge_') as temp:
            _,gt,_,b0,b1,_=fixture(Path(temp))
            rows=audit.bridge_candidates(b0,b1,gt,np.zeros_like(gt),'image','same','repair',True)
            self.assertGreater(len(rows),0)
            self.assertEqual(len(rows),bridge_proxy(b0,b1,gt))

    def test_short_components_track_background_and_thin_gt(self):
        old=np.zeros((24,24),dtype=bool);new=old.copy();gt=old.copy();thin=old.copy()
        new[2,2:5]=True
        new[10,10:14]=True;gt[10,10:14]=True;thin[10,10:14]=True
        rows,counts=audit.new_short_components(old,new,gt,thin,20,'case','same','transition')
        self.assertEqual(counts['short_component_net_change'],2)
        self.assertEqual({row['category'] for row in rows},{'mostly_GT_background','mostly_GT_road_thin_proxy'})
        self.assertEqual(sum(row['area_pixels'] for row in rows),7)

    def test_existing_component_that_shrinks_into_short_is_recorded(self):
        old=np.zeros((32,32),dtype=bool);old[10:15,10:15]=True
        new=np.zeros_like(old);new[10:12,10:12]=True
        rows,counts=audit.new_short_components(old,new,old,old,20,'case','same','transition')
        self.assertEqual(counts['before_short_components'],0)
        self.assertEqual(counts['after_short_components'],1)
        self.assertEqual(rows[0]['origin'],'became_short_after_shrink')

    def test_disjoint_regions_account_for_all_tp_fp_changes(self):
        gt=np.zeros((32,32),dtype=bool);gt[16,4:28]=True
        old=np.zeros_like(gt);old[16,4:12]=True
        new=old.copy();new[16,12:20]=True;new[4:6,4:6]=True
        oldp=np.where(old,.9,.1).astype(np.float32);newp=np.where(new,.9,.1).astype(np.float32)
        entry={'gaps':[{'pixels':[16*32+x for x in range(12,20)]}]}
        prepared=audit.prepare_gt(gt,entry,8.)
        rows=audit.region_rows(oldp,newp,gt,prepared['regions'],'case','same','repair',.5,.5)
        self.assertEqual(sum(row['region_pixels'] for row in rows),gt.size)
        self.assertEqual(sum(row['added_tp'] for row in rows),8)
        self.assertEqual(sum(row['added_fp'] for row in rows),4)

    def test_all_additions_partition_into_gap_island_edge_extension(self):
        gt=np.zeros((32,32),dtype=bool);gt[16,4:28]=True
        old=np.zeros_like(gt);old[16,4:12]=True
        new=old.copy();new[16,12:20]=True;new[4:6,4:6]=True
        gap=np.zeros_like(gt);gap[16,16:20]=True
        _,labels,_,_=audit.component_data(new)
        rows=audit.addition_spatial_rows(old,new,gt,gap,labels,'case','same','repair')
        counts={row['category']:row['added_pixels'] for row in rows}
        self.assertEqual(counts,{'fixed_gap_corridor_2px':4,'isolated_after_component':4,
                                 'previous_prediction_edge_2px':2,'attached_extension_beyond_2px':2})
        self.assertEqual(sum(row['added_pixels'] for row in rows),int((new&~old).sum()))

    def test_complete_read_only_pair_and_198_count_guard(self):
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='.pair_audit_') as temp:
            pair=Path(temp);tag,_,_,_,_,_=fixture(pair)
            old_a=(pair/'A/best.pth').read_bytes();old_b=(pair/tag/'best.pth').read_bytes()
            with self.assertRaises(ValueError):audit.load_inputs(pair,tag,198,2)
            cases,manifest,_,selection,_=audit.load_inputs(pair,tag,1,2)
            out=pair/'audit'
            summary=audit.analyze(cases,manifest,selection,out,8.,20,2)
            same=summary['results']['same_threshold']
            self.assertEqual(same['fixed_gaps']['eligible'],1)
            self.assertEqual(same['fixed_gaps']['repair_newly_closed_global'],1)
            self.assertEqual(same['fixed_gaps']['repair_newly_closed_local'],1)
            self.assertGreater(same['transitions']['B_P1_vs_B_P0']['added_tp'],0)
            self.assertGreater(same['transitions']['B_P1_vs_B_P0']['strict_bridges'],0)
            with (out/'fixed_gaps_common.csv').open(newline='',encoding='utf-8') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['gap_id'],'7')
            self.assertGreater(int(rows[0]['B_P1_vs_B_P0_strict_bridges_within_16px']),0)
            self.assertEqual(len(list((out/'overviews').glob('*.png'))),2)
            self.assertEqual((pair/'A/best.pth').read_bytes(),old_a)
            self.assertEqual((pair/tag/'best.pth').read_bytes(),old_b)
            path=pair/tag/'epoch_002.json'
            record=json.loads(path.read_text());record['case_order_sha256']='different';json_file(path,record)
            with self.assertRaises(ValueError):audit.load_inputs(pair,tag,1,2)

    def test_user_command_runs_with_only_saved_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='.pair_cli_') as temp:
            pair=Path(temp);tag,_,_,_,_,_=fixture(pair);out=pair/'audit_cli'
            process=subprocess.run([sys.executable,str(ROOT/'diagnostics/audit_full_h0_saved_predictions.py'),
                '--pair_dir',str(pair),'--b_run_name',tag,'--output_dir',str(out),
                '--expected_gaps','1','--expected_images','2','--visualizations','0'],
                cwd=ROOT,text=True,capture_output=True)
            self.assertEqual(process.returncode,0,process.stderr)
            self.assertIn('[VERIFIED] 2 saved val images; 1 frozen gaps',process.stdout)
            self.assertTrue((out/'REPORT.md').is_file())
            with (out/'fixed_gaps_common.csv').open(newline='',encoding='utf-8') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))),1)
            saved=json.loads((out/'summary.json').read_text())
            self.assertEqual(saved['provenance']['B_directory'],tag)

    def test_saved_predictions_work_without_model_weights_but_warn(self):
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='.pair_no_weights_') as temp:
            pair=Path(temp);tag,_,_,_,_,_=fixture(pair)
            (pair/'A/best.pth').unlink()
            cases,manifest,cfg,selection,sha=audit.load_inputs(pair,tag,1,2)
            self.assertEqual(len(cases),2)
            self.assertIn('missing',selection['_verification']['checkpoint_hash_status']['A'])


if __name__=='__main__':unittest.main()
