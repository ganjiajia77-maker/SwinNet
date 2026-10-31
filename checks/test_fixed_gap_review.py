"""Read-only image atlas, cache identity, exact original predicate, HTML data."""
import argparse
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from diagnostics.review_fixed_gaps import build,assess,gap_geometry,cached_predictions,fingerprint,resolve_cases,read_json


def fixture(root):
    data=root/'data'; (data/'val').mkdir(parents=True); (data/'val_labels').mkdir()
    transport=root/'current_S';transport.mkdir(); choice=root/'choice';choice.mkdir()
    manifest={'count':2,'p0_reference_threshold':.45,'images':{}}
    for i in range(2):
        name=f'case{i}';y=20+i*12
        rgb=np.full((64,64,3),(20+i*20,100,170),np.uint8)
        gt=np.zeros((64,64),np.uint8);gt[y-2:y+3,4:60]=1
        p0=gt.astype(np.float32);p0[:,24:36]=0
        cv2.imwrite(str(data/'val'/f'{name}_sat.tif'),rgb)
        cv2.imwrite(str(data/'val_labels'/f'{name}_mask.tif'),gt*255)
        manifest['images'][name]={'shape':[64,64],'gaps':[{'id':10+i,'pixels':[y*64+x for x in range(27,33)],
            'anchors':[[y*64+18],[y*64+42]],'length':6,'length_bin':'3-7'}]}
        # Reverse numbered files to prove cache mapping does NOT assume order.
        np.savez_compressed(choice/f'val_probabilities_{1-i:02d}.npz',gt=gt.astype(bool),p0=p0,current_S=gt.astype(np.float32),choice=p0)
    manifest['sha256']='temporary'
    cases=resolve_cases(data,manifest);preds,_=cached_predictions(cases,transport,choice)
    manifest['sha256']=fingerprint(cases,preds)
    path=transport/'fixed_gaps_and_anchors.json';path.write_text(json.dumps(manifest))
    (choice/'comparison.json').write_text(json.dumps({'reports':{'current_S_common':{'threshold':.5}}}))
    args=argparse.Namespace(root_path=data,manifest=path,transport_run=transport,choice_run=choice,
        output_dir=root/'review',threshold=None,context=16,expected_count=2,expected_images=2)
    return args,manifest,cases,preds


class ReviewTests(unittest.TestCase):
    def test_full_atlas_native_coordinates_and_readonly_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            args,manifest,cases,preds=fixture(Path(directory));original=args.manifest.read_bytes()
            build(args)
            self.assertEqual(args.manifest.read_bytes(),original)
            rows=read_json(args.output_dir/'gaps.json');self.assertEqual(len(rows),2)
            self.assertTrue(read_json(args.output_dir/'summary.json')['reference_fingerprint_verified'])
            self.assertTrue(rows[0]['current_S_official_proxy_pass']);self.assertFalse(rows[0]['choice_official_proxy_pass'])
            marked=cv2.imread(str(args.output_dir/'overviews/case0_marked.png'))
            self.assertEqual(marked.shape,(64,64,3))
            ids=cv2.imread(str(args.output_dir/'masks/case0_gap_ids.tif'),cv2.IMREAD_UNCHANGED)
            self.assertEqual(ids.dtype,np.uint16);self.assertEqual(np.count_nonzero(ids),6)
            self.assertTrue(all(ids.reshape(-1)[p]==1 for p in manifest['images']['case0']['gaps'][0]['pixels']))
            rgb=cv2.imread(str(args.output_dir/rows[0]['rgb_crop']))
            self.assertEqual(rgb.shape[:2],(rows[0]['crop_y1_exclusive']-rows[0]['crop_y0'],rows[0]['crop_x1_exclusive']-rows[0]['crop_x0']))
            page=(args.output_dir/'index.html').read_text(encoding='utf-8')
            embedded=re.search(r'<script id="data" type="application/json">(.*?)</script>',page,re.S).group(1)
            self.assertEqual(len(json.loads(embedded)['rows']),2)
            self.assertTrue(all(not row['scene_type'] for row in rows))
            template=args.output_dir/'review_template.csv';template.write_text('manually reviewed')
            build(args);self.assertEqual(template.read_text(),'manually reviewed')

    def test_cache_mapping_is_content_based_and_gt_changes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args,manifest,cases,preds=fixture(Path(directory))
            self.assertTrue(preds['case0']['current_S'][20,30])
            self.assertFalse(preds['case0']['current_S'][32,30])
            mask=cases['case0']['gt'].copy();mask[0,0]=True
            cv2.imwrite(str(cases['case0']['gt_path']),mask.astype(np.uint8)*255)
            with self.assertRaisesRegex(ValueError,'Cache GT matches'): build(args)

    def test_original_global_and_extra_local_connectivity_distinguished(self):
        pred=np.zeros((64,64),bool)
        # Gap and anchor pixels all present, connected only by a distant detour.
        pred[30,20:24]=True;pred[30,40:44]=True
        pred[5:31,20]=True;pred[5:31,43]=True;pred[5,20:44]=True
        gap={'pixels':[30*64+21,30*64+22],'anchors':[[30*64+20],[30*64+43]]}
        result=assess(pred,gap,(15,20,49,41))
        self.assertTrue(result['official_proxy_pass'])
        self.assertTrue(result['diagnostic_global_only_connection'])

    def test_disappeared_anchor_and_exact_coverage_failure(self):
        pred=np.zeros((32,32),bool);pred[16,10:15]=True
        gap={'pixels':[16*32+10,16*32+11,16*32+18],'anchors':[[16*32+10],[16*32+22]]}
        result=assess(pred,gap,(5,5,28,25))
        self.assertFalse(result['coverage_pass']);self.assertEqual(result['missing_anchor_groups'],1)
        self.assertIn('anchor_group_disappeared',result['failure_reason'])

    def test_no_cache_still_exports_rgb_and_uncertain_not_fake_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            args,manifest,_,_=fixture(Path(directory));args.choice_run=None;args.transport_run=None
            build(args);summary=read_json(args.output_dir/'summary.json')
            self.assertFalse(summary['reference_fingerprint_verified'])
            self.assertIsNone(summary['common_threshold'])
            row=read_json(args.output_dir/'gaps.json')[0]
            self.assertEqual(row['choice_failure_reason'],'unavailable_cache_or_threshold')
            self.assertNotIn('choice_official_proxy_pass',row)


if __name__=='__main__': unittest.main()
