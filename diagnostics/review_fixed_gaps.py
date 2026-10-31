"""Native-image fixed-gap atlas. Read-only inputs; CPU, no checkpoints/inference."""
import argparse
import csv
import hashlib
import html
import json
import sys
from pathlib import Path

# Works both as `python diagnostics/review_fixed_gaps.py` and `python -m ...`.
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

import cv2
import numpy as np
from evaluation.road_comparison.road_metrics import skeletonize

VERSION='fixed_gap_scene_review_v1'
PALETTE=((255,70,220),(20,220,255),(220,200,30),(80,255,80),(255,160,60),(160,100,255)) # BGR
ANNOTATIONS=('confirmed_true_gap','road_type','background','scene_type','anchor_reliability','confidence','notes')


def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    value=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''): value.update(block)
    return value.hexdigest()


def dump(path,value): Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def write_csv(path,rows):
    if not rows: return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with open(path,'w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def save_image(path,image):
    if not cv2.imwrite(str(path),image): raise OSError(f'Cannot write image: {path}')


def case_id(path): return Path(path).stem.replace('_sat','')


def resolve_cases(root,manifest):
    image_dir=next((path for path in (root/'val/image',root/'val') if path.is_dir()),None)
    label_dir=next((path for path in (root/'val/mask',root/'val/label',root/'val_labels') if path.is_dir()),None)
    if image_dir is None or label_dir is None: raise FileNotFoundError(f'Missing native val images/labels under {root}')
    extensions={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}; available={}
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() in extensions:
            key=case_id(path)
            if key in available: raise ValueError(f'Ambiguous image case: {key}')
            available[key]=path
    if set(available)!=set(manifest['images']): raise ValueError('Image IDs differ from the fixed manifest; no silent subset matching')
    result={}
    for key,entry in manifest['images'].items():
        if Path(key).name!=key or '/' in key or '\\' in key: raise ValueError('Case IDs must be plain filenames')
        path=available[key]; base=path.stem
        candidates=[path.name]+[base+ext for ext in ('.png','.tif','.tiff')]
        for source in ('_sat','_image','_img'):
            candidates += [base.replace(source,'_mask')+ext for ext in ('.png','.tif','.tiff')]
        target=next((label_dir/name for name in candidates if (label_dir/name).is_file()),None)
        if target is None: raise FileNotFoundError(f'No native GT for {path}')
        rgb=cv2.imread(str(path),cv2.IMREAD_COLOR); gt=cv2.imread(str(target),cv2.IMREAD_GRAYSCALE)
        if rgb is None or gt is None: raise ValueError(f'Unreadable original image/GT: {path}, {target}')
        if list(rgb.shape[:2])!=entry['shape'] or list(gt.shape)!=entry['shape']: raise ValueError(f'Native shape mismatch: {key}; resizing is forbidden')
        values=np.unique(gt)
        if not np.isin(values,(0,1,255)).all() or (1 in values and 255 in values): raise ValueError(f'Nonbinary GT: {target}')
        result[key]={'image_path':path,'gt_path':target,'rgb':rgb,'gt':gt>0}
    return result


def gt_key(mask):
    return hashlib.sha256(np.asarray(mask.shape,dtype=np.int64).tobytes()+np.packbits(mask,axis=None).tobytes()).hexdigest()


def cached_predictions(cases,transport_run=None,choice_run=None):
    """Map numbered NPZs by exact, uniquely matching GT content, never filenames.

    Role semantics are explicit. control_predictions.p1 is B_control, NOT S.
    Probability mosaics/RGB screenshots are never accepted as numeric predictions.
    """
    lookup={}
    for key,case in cases.items(): lookup.setdefault(gt_key(case['gt']),[]).append(key)
    predictions={key:{} for key in cases}; sources=[]
    candidates=[]
    if choice_run and choice_run.is_dir(): candidates += [(path,'choice') for path in sorted(choice_run.glob('val_probabilities_*.npz'))]
    if transport_run and transport_run.is_dir():
        candidates += [(path,'transport') for path in sorted(transport_run.glob('val_probabilities_*.npz'))]
        candidates += [(path,'control_p0') for path in sorted((transport_run/'control_predictions').glob('*.npz'))]
    for path,kind in candidates:
        with np.load(path,allow_pickle=False) as cache:
            if not {'gt','p0'}<=set(cache.files): continue
            gt=np.asarray(cache['gt'])
            if gt.ndim!=2 or not np.isin(gt,(0,1)).all(): raise ValueError(f'Cache GT must be native 2D binary: {path}')
            keys=lookup.get(gt_key(gt.astype(bool)),[])
            if len(keys)!=1: raise ValueError(f'Cache GT matches {len(keys)} images, cannot safely identify {path}')
            key=keys[0]
            roles={'p0':'p0'}
            if kind=='choice': roles.update(current_S='current_S',choice='choice')
            elif kind=='transport': roles.update(current_S='selector')
            for role,source_key in roles.items():
                if source_key not in cache: continue
                value=np.asarray(cache[source_key])
                if value.shape!=gt.shape or value.dtype.kind!='f' or not np.isfinite(value).all() or value.min()<0 or value.max()>1:
                    raise ValueError(f'Invalid native probability map: {path}:{source_key}')
                if role in predictions[key]:
                    old=predictions[key][role]
                    if role=='p0' and not np.array_equal(old>=.45,value>=.45): raise ValueError(f'Conflicting frozen P0 caches for {key}')
                    continue # choice cache takes priority over older transport cache
                predictions[key][role]=value.copy()
                sources.append({'case':key,'role':role,'path':str(path.resolve()),'key':source_key,'mapping':'unique exact GT content'})
    return predictions,sources


def validate_manifest(manifest,count,images):
    if len(manifest['images'])!=images: raise ValueError(f'Expected {images} fixed images')
    actual=sum(len(row['gaps']) for row in manifest['images'].values())
    if actual!=count or manifest['count']!=actual: raise ValueError(f'Expected {count} fixed gaps, got {actual}')
    if manifest['p0_reference_threshold']!=.45: raise ValueError('This review requires original P0 reference0.45')
    for case,entry in manifest['images'].items():
        height,width=entry['shape']; seen=set(); ids=set()
        for gap in entry['gaps']:
            if gap['id'] in ids: raise ValueError(f'Duplicate gap ID in {case}')
            ids.add(gap['id']); pixels=gap['pixels']
            if not pixels or len(pixels)!=len(set(pixels)) or any(type(p) is not int or p<0 or p>=height*width for p in pixels): raise ValueError('Invalid frozen gap pixels')
            if seen.intersection(pixels): raise ValueError('Fixed gap components overlap')
            seen.update(pixels)
            if len(gap['anchors'])<2: raise ValueError('Original fixed candidates must have at least two anchor groups')
            for anchor in gap['anchors']:
                if not anchor or any(type(p) is not int or p<0 or p>=height*width for p in anchor): raise ValueError('Invalid frozen anchors')


def fingerprint(cases,predictions):
    if any('p0' not in predictions[key] for key in cases): return None
    digest=hashlib.sha256()
    for key,case in cases.items():
        gt=case['gt']; digest.update(key.encode()+b'\0'); digest.update(np.asarray(gt.shape,dtype=np.int64).tobytes())
        for mask in (predictions[key]['p0']>=.45,gt,skeletonize(gt)): digest.update(np.packbits(mask,axis=None).tobytes())
    return digest.hexdigest()


def gap_geometry(gap,shape,context):
    height,width=shape; pixels=np.asarray(gap['pixels'],np.int64)
    all_points=np.concatenate([pixels]+[np.asarray(a,np.int64) for a in gap['anchors']])
    yy,xx=np.divmod(all_points,width); gy,gx=np.divmod(pixels,width)
    box=(max(0,int(xx.min())-context),max(0,int(yy.min())-context),min(width,int(xx.max())+context+1),min(height,int(yy.max())+context+1))
    return {'crop':box,'centroid':(float(gx.mean()),float(gy.mean())),
            'missing_bbox':(int(gx.min()),int(gy.min()),int(gx.max())+1,int(gy.max())+1)}


def assess(pred,gap,box,components=None):
    """Original global predicate plus explicitly NON-official diagnostic checks."""
    height,width=pred.shape; flat=pred.reshape(-1); points=np.asarray(gap['pixels'],np.int64)
    if components is None: _,components=cv2.connectedComponents(pred.astype(np.uint8),connectivity=8)
    labels=components.reshape(-1)
    sets=[set(np.unique(labels[a]))-{0} for a in gap['anchors']]
    missing=sum(not values for values in sets)
    connected=bool(sets and set.intersection(*sets))
    coverage=float(flat[points].mean()); coverage_pass=coverage>=.8
    x0,y0,x1,y1=box; _,local=cv2.connectedComponents(pred[y0:y1,x0:x1].astype(np.uint8),connectivity=8)
    local_sets=[]
    for anchor in gap['anchors']:
        yy,xx=np.divmod(np.asarray(anchor,np.int64),width)
        local_sets.append(set(np.unique(local[yy-y0,xx-x0]))-{0})
    local_connected=bool(local_sets and set.intersection(*local_sets))
    tolerant=cv2.dilate(pred.astype(np.uint8),np.ones((5,5),np.uint8)).reshape(-1)
    failures=[]
    if not coverage_pass: failures.append('centerline_coverage_below_80pct')
    if missing: failures.append('anchor_group_disappeared')
    elif not connected: failures.append('anchor_groups_not_connected')
    return {'centerline_coverage':coverage,'coverage_pass':coverage_pass,'missing_anchor_groups':missing,
        'global_anchors_connected':connected,'official_proxy_pass':bool(coverage_pass and connected),
        'failure_reason':'|'.join(failures) or 'passed',
        'diagnostic_coverage_tolerance2px':float(tolerant[points].mean()),
        'diagnostic_local_anchors_connected':local_connected,
        'diagnostic_global_only_connection':bool(connected and not local_connected)}


def overlay(image,gt,missing,anchors,pred=None):
    out=image.copy()
    if pred is None:
        contour=cv2.morphologyEx(gt.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0
        out[contour]=(240,150,30)
    else:
        take=pred.astype(bool)
        out[take]=(out[take].astype(np.float32)*.55+np.array((30,230,40))*.45).astype(np.uint8)
    visible=cv2.dilate(missing.astype(np.uint8),np.ones((3,3),np.uint8))>0
    out[visible]=(0,40,255)
    for i,anchor in enumerate(anchors): out[anchor]=PALETTE[i%len(PALETTE)]
    return out


def labelled(image,text):
    # Pad narrow border crops for readable captions; never resample the RGB.
    width=max(256,image.shape[1]); canvas=np.full((image.shape[0],width,3),235,np.uint8)
    left=(width-image.shape[1])//2; canvas[:,left:left+image.shape[1]]=image
    panel=cv2.copyMakeBorder(canvas,26,0,0,0,cv2.BORDER_CONSTANT,value=(25,25,25))
    cv2.putText(panel,text,(5,18),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA)
    return panel


def render_gap(folder,key,gap,global_id,case,predictions,threshold):
    rgb,gt=case['rgb'],case['gt']; shape=gt.shape; height,width=shape
    geometry=gap_geometry(gap,shape,folder['context']); x0,y0,x1,y1=geometry['crop']
    missing=np.zeros(shape,bool); missing.reshape(-1)[gap['pixels']]=True
    anchors=[]
    for points in gap['anchors']:
        anchor=np.zeros(shape,bool); anchor.reshape(-1)[points]=True; anchors.append(anchor)
    cropped=rgb[y0:y1,x0:x1]; gt_crop=gt[y0:y1,x0:x1]; miss_crop=missing[y0:y1,x0:x1]
    anchor_crops=[a[y0:y1,x0:x1] for a in anchors]
    panels=[labelled(cropped,'RGB - unmodified'),labelled(overlay(cropped,gt_crop,miss_crop,anchor_crops),'GT / missing / anchors')]
    row={'gap_uid':global_id,'case':key,'source_gap_id':gap['id'],'gap_pixel_count':len(gap['pixels']),
         'source_length_field':gap['length'],'source_length_bin':gap['length_bin'],'anchor_groups':len(anchors),
         'centroid_x':geometry['centroid'][0],'centroid_y':geometry['centroid'][1],
         'crop_x0':x0,'crop_y0':y0,'crop_x1_exclusive':x1,'crop_y1_exclusive':y1,
         'missing_x0':geometry['missing_bbox'][0],'missing_y0':geometry['missing_bbox'][1],
         'missing_x1_exclusive':geometry['missing_bbox'][2],'missing_y1_exclusive':geometry['missing_bbox'][3],
         'native_image_path':str(case['image_path'].resolve()),'native_gt_path':str(case['gt_path'].resolve()),
         'case_overview':f'overviews/{key}_marked.png','rgb_crop':f'gaps/{global_id}_rgb.png','mosaic':f'gaps/{global_id}_review.jpg',
         **{field:'' for field in ANNOTATIONS}}
    for role in ('p0','current_S','choice'):
        used=.45 if role=='p0' else threshold
        values=predictions.get(role)
        if values is not None and used is not None:
            pred=values>=used
            assessment=assess(pred,gap,geometry['crop'],folder['components'][role])
            row.update({role+'_'+name:value for name,value in assessment.items()}); row[role+'_threshold']=used
            panel=overlay(cropped,gt_crop,miss_crop,anchor_crops,pred[y0:y1,x0:x1])
            panels.append(labelled(panel,f'{role} binary @ {used:.4f}'))
        else:
            row[role+'_failure_reason']='unavailable_cache_or_threshold'
            panel=np.full_like(cropped,235); panels.append(labelled(panel,role+' unavailable'))
    if 'p0' in predictions and 'choice' in predictions:
        delta=predictions['choice'][y0:y1,x0:x1]-predictions['p0'][y0:y1,x0:x1]
        heat=cv2.applyColorMap(np.uint8(np.clip((delta+.25)/.5,0,1)*255),cv2.COLORMAP_JET)
        panels.append(labelled(heat,'choice-P0 [-0.25,+0.25]'))
    else: panels.append(labelled(np.full_like(cropped,235),'Probability change unavailable'))
    mosaic=np.concatenate((np.concatenate(panels[:3],1),np.concatenate(panels[3:],1)),0)
    save_image(folder['root']/row['rgb_crop'],cropped); save_image(folder['root']/row['mosaic'],mosaic)
    return row,geometry


def threshold_from_reports(transport_run,choice_run,explicit):
    if explicit is not None:
        if not 0<explicit<1: raise ValueError('Common threshold must be in(0,1)')
        return explicit,'explicit common diagnostic threshold'
    if choice_run and (choice_run/'comparison.json').is_file():
        value=read_json(choice_run/'comparison.json')['reports']['current_S_common']['threshold']
        return float(value),str(choice_run/'comparison.json')+' : current_S_common'
    if transport_run and (transport_run/'comparison.json').is_file():
        value=read_json(transport_run/'comparison.json')['reports']['S_selector']['own_val_threshold']['threshold']
        return float(value),str(transport_run/'comparison.json')+' : S_selector own_val_threshold'
    return None,'missing metadata; only RGB/GT/P0 are scored, S/choice not thresholded'


def build(args):
    manifest=read_json(args.manifest); manifest_sha=sha(args.manifest)
    validate_manifest(manifest,args.expected_count,args.expected_images)
    binding=args.manifest.parent/'fixed_reference_binding.json'
    if binding.is_file() and read_json(binding)['manifest_sha256']!=manifest_sha: raise ValueError('Manifest differs from its fixed binding')
    if args.choice_run and (args.choice_run/'fixed_gaps_and_anchors.json').is_file():
        if sha(args.choice_run/'fixed_gaps_and_anchors.json')!=manifest_sha: raise ValueError('Choice and S runs use different fixed manifests')
    if args.choice_run and args.transport_run and (args.choice_run/'experiment.json').is_file() and (args.transport_run/'best.pth').is_file():
        expected=read_json(args.choice_run/'experiment.json').get('source_sha256')
        if expected and expected!=sha(args.transport_run/'best.pth'): raise ValueError('Choice continuation was initialized from a different current S checkpoint')
    cases=resolve_cases(args.root_path,manifest)
    predictions,sources=cached_predictions(cases,args.transport_run,args.choice_run)
    reference_sha=fingerprint(cases,predictions)
    if reference_sha is not None and reference_sha!=manifest['sha256']: raise ValueError('Cached P0/native GT fingerprint differs from the fixed original reference')
    threshold,origin=threshold_from_reports(args.transport_run,args.choice_run,args.threshold)
    out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    if out.resolve() in {path.resolve() for path in (args.transport_run,args.choice_run,args.root_path,args.manifest.parent) if path is not None}:
        raise ValueError('Output must be a separate review directory, not an input run/data root')
    identity={'version':VERSION,'manifest_sha256':manifest_sha,'reference_sha256':manifest['sha256'],
              'threshold':threshold,'context':args.context,'root_path':str(args.root_path.resolve()),
              'native_files':{key:{'image_sha256':sha(case['image_path']),'gt_sha256':sha(case['gt_path'])} for key,case in cases.items()},
              'source_files':[{**item,'file_sha256':sha(Path(item['path']))} for item in sources]}
    marker=out/'identity.json'
    if marker.is_file() and read_json(marker)!=identity: raise ValueError('Review identity/cache changed; use a new output directory')
    dump(marker,identity)
    for name in ('originals','overviews','gaps','masks'): (out/name).mkdir(exist_ok=True)
    rows=[]; locations=[]; number=0
    for key,entry in manifest['images'].items():
        case=cases[key]; rgb=case['rgb']; gt=case['gt']; marked=rgb.copy(); ids=np.zeros(gt.shape,np.uint16)
        save_image(out/'originals'/f'{key}.png',rgb)
        components={}
        for role,prob in predictions[key].items():
            used=.45 if role=='p0' else threshold
            if used is not None: _,components[role]=cv2.connectedComponents((prob>=used).astype(np.uint8),connectivity=8)
        folder={'root':out,'context':args.context,'components':components}
        image_locations=[]
        for gap in entry['gaps']:
            number+=1; uid=f'G{number:04d}'
            if number>65535: raise ValueError('More than65535 gaps; cannot encode native uint16 ID masks')
            row,geometry=render_gap(folder,key,gap,uid,case,predictions[key],threshold)
            rows.append(row); ids.reshape(-1)[gap['pixels']]=number
            points=np.asarray(gap['pixels'],np.int64); yy,xx=np.divmod(points,gt.shape[1])
            missing=np.zeros(gt.shape,np.uint8); missing[yy,xx]=1
            marked[cv2.dilate(missing,np.ones((3,3),np.uint8))>0]=(0,40,255)
            for index,anchor in enumerate(gap['anchors']):
                ay,ax=np.divmod(np.asarray(anchor,np.int64),gt.shape[1]); marked[ay,ax]=PALETTE[index%len(PALETTE)]
            bx0,by0,bx1,by1=geometry['missing_bbox']; cv2.rectangle(marked,(max(0,bx0-5),max(0,by0-5)),(min(gt.shape[1]-1,bx1+5),min(gt.shape[0]-1,by1+5)),(0,200,255),1)
            tx=max(1,min(gt.shape[1]-64,bx0)); ty=max(16,min(gt.shape[0]-3,by0-7))
            cv2.rectangle(marked,(tx-1,ty-13),(tx+64,ty+3),(0,0,0),-1)
            cv2.putText(marked,uid,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1,cv2.LINE_AA)
            cv2.line(marked,(tx,ty+3),(int(xx.mean()),int(yy.mean())),(0,200,255),1)
            image_locations.append({'gap_uid':uid,'source_gap_id':gap['id'],'centroid_xy':geometry['centroid'],
                'missing_bbox_xyxy_exclusive':geometry['missing_bbox'],'context_bbox_xyxy_exclusive':geometry['crop'],
                'missing_pixels_xy':np.stack((xx,yy),1).tolist(),
                'anchor_groups_xy':[np.stack((np.asarray(a)%gt.shape[1],np.asarray(a)//gt.shape[1]),1).tolist() for a in gap['anchors']]})
        save_image(out/'overviews'/f'{key}_marked.png',marked)
        save_image(out/'masks'/f'{key}_gap_ids.tif',ids)
        locations.append({'case':key,'native_shape_hw':list(gt.shape),'gaps':image_locations})
        print(f'[GAP REVIEW] {key}: {len(entry["gaps"])} fixed candidates',flush=True)
    write_csv(out/'gap_inventory.csv',rows)
    template=out/'review_template.csv'
    if not template.exists(): write_csv(template,rows) # never overwrite a manually edited template
    dump(out/'gap_locations.json',{'coordinate_system':'native image pixels x/y; zero-based; top-left origin; NOT geographic coordinates','images':locations})
    dump(out/'gaps.json',rows)
    summary={'version':VERSION,'images':len(cases),'case_ids':list(cases),'fixed_candidates':len(rows),'reviewed':0,'scene_labels':'not automatically inferred; manual review required',
        'manifest_path':str(args.manifest.resolve()),'manifest_sha256':manifest_sha,'reference_fingerprint_verified':reference_sha is not None,
        'common_threshold':threshold,'threshold_origin':origin,'prediction_sources':sources,
        'missing_predictions':{key:[role for role in ('p0','current_S','choice') if role not in predictions[key]] for key in cases},
        'limits':['Candidates are not manually confirmed true breaks.','length is missing-centerline component area, not endpoint distance.',
            'Original pass predicate is >=80% exact missing pixels AND common global anchor component; tolerance/local connection are diagnostic only.',
            'Road material, obstruction and background categories require RGB review; no color-threshold semantic labels are fabricated.']}
    dump(out/'summary.json',summary)
    from diagnostics.review_fixed_gaps_html import write_review_page
    write_review_page(out/'index.html',rows,summary)
    (out/'REPORT.md').write_text(f'# Fixed-gap native-image review\n\nImages: {len(cases)}; candidates: {len(rows)}.\n'
        f'P0/GT reference fingerprint verified: {reference_sha is not None}.\nCommon S/choice threshold: {threshold}; {origin}.\n\n'
        'Open index.html after downloading this COMPLETE folder. Original-sized marked images: overviews/. Unmodified RGB originals: originals/. '
        'Native uint16 candidate-ID masks: masks/ (G0001 = pixel value1). Pixel coordinates: gap_locations.json.\n\n'
        'Red=missing GT centerline; colored anchor pixels distinguish anchor groups; cyan-blue=GT boundary; green=predicted road. '
        'G IDs are review IDs; source_gap_id preserves original component ID.\n\n'
        'Scene labels initially remain unreviewed. Use the page to classify road/background, confirm/reject/mark uncertainty, and export annotations. '
        'Local browser saving is convenience only: export JSON/CSV to preserve work. Original manifest/checkpoints/predictions are never changed.\n',encoding='utf-8')
    if sha(args.manifest)!=manifest_sha: raise RuntimeError('Input manifest changed during review generation')
    print('[DONE] '+str(out/'index.html'),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root_path',type=Path,required=True); parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output_dir',type=Path,required=True); parser.add_argument('--transport_run',type=Path); parser.add_argument('--choice_run',type=Path)
    parser.add_argument('--threshold',type=float); parser.add_argument('--context',type=int,default=128)
    parser.add_argument('--expected_count',type=int,default=206); parser.add_argument('--expected_images',type=int,default=14)
    args=parser.parse_args()
    if args.context<8: parser.error('Context must be at least8 native pixels')
    build(args)


if __name__=='__main__': main()
