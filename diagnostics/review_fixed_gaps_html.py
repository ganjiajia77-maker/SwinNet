"""Offline visual scene review, annotation export; no network/model calls."""
import json


PAGE=r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>固定缺口 · 原影像与场景审查</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f6f8;color:#17232d;font:15px system-ui,"Microsoft YaHei",sans-serif}
header{background:#18394a;color:white;padding:18px 24px}h1{margin:0 0 7px;font-size:23px}header p{margin:5px 0;color:#d6e6eb}
button,input,select,textarea{font:inherit}button{padding:7px 12px;border:1px solid #9ab0bb;background:white;border-radius:5px;cursor:pointer}button:hover{background:#e6f1f6}
nav{padding:12px 20px;background:white;display:flex;flex-wrap:wrap;gap:8px;border-bottom:1px solid #cdd8df;align-items:center}
main{display:grid;grid-template-columns:250px 1fr;gap:16px;padding:16px}aside{background:white;border-radius:7px;padding:10px;max-height:90vh;overflow:auto;position:sticky;top:0;align-self:start}
.item{width:100%;text-align:left;display:block;margin:4px 0;font-size:13px}.active{background:#d7edf7;border-color:#315f75}.muted{color:#5f707b;font-size:13px}
.panel{background:white;border-radius:7px;padding:16px;margin-bottom:15px}h2{font-size:20px;margin:0 0 9px}h3{font-size:17px;margin:6px 0 10px}
img{max-width:100%;height:auto}#mosaic{border:1px solid #becbd3}.fields{display:grid;grid-template-columns:repeat(3,minmax(150px,1fr));gap:12px}
label{display:block;font-weight:600}select,textarea{display:block;width:100%;margin-top:5px;padding:6px;border:1px solid #b8c7cf;border-radius:4px}textarea{min-height:80px}
.wide{grid-column:1/-1}.tags{display:flex;flex-wrap:wrap;gap:9px;padding:8px 0}.tags label{font-weight:400;font-size:14px}#status{font-size:13px;color:#486677}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #d5dfe4;padding:7px;text-align:left}th{background:#edf3f6}
.notice{padding:10px;background:#fff4d8;border-left:4px solid #c59a35;margin:10px 0}.toplinks{display:flex;gap:18px;flex-wrap:wrap;margin:8px 0}
@media(max-width:850px){main{grid-template-columns:1fr}aside{position:static;max-height:220px}.fields{grid-template-columns:1fr 1fr}}
</style>
<header><h1>固定缺口：先看原图，再判断是否是真实断路</h1><p id="identity"></p><p>场景标签初始均未审查。树冠、阴影、窄路、邻路混入等，需要根据 RGB 与 GT 判断；不能从补通计数自动推断。</p></header>
<nav><button id="previous">← 上一个</button><button id="next">下一个 →</button>
<select id="caseFilter" style="width:auto;margin:0"></select><select id="reviewFilter" style="width:auto;margin:0"><option value="all">全部审查状态</option><option value="todo">未审查</option><option value="yes">确认真实断路</option><option value="no">不是真实断路</option><option value="uncertain">不确定</option></select>
<select id="lengthFilter" style="width:auto;margin:0"><option value="all">全部缺失像素数分档</option><option>3-7</option><option>8-15</option><option>16-31</option><option>32+</option></select>
<button id="exportJson">导出审查 JSON</button><button id="exportCsv">导出审查 CSV</button><button id="import">导入审查 JSON</button><input id="file" type="file" accept=".json" hidden><span id="status"></span></nav>
<main><aside><strong id="progress"></strong><div id="list"></div></aside><section>
<div class="panel"><h2 id="heading"></h2><div class="muted" id="geometry"></div><div class="toplinks"><a id="rgbLink" target="_blank">干净 RGB 裁剪</a><a id="overviewLink" target="_blank">原影像固定缺口总标记（原尺寸）</a><a id="originalLink" target="_blank">未标记原影像</a></div>
<div class="notice">红色＝原固定缺失中心线；彩色点＝不同锚点组；蓝青色＝GT 道路边界；绿色半透明＝预测道路。长度字段是中心线分量像素数，不是道路两端距离。局部图像没有缩放坐标。</div>
<a id="mosaicLink" target="_blank"><img id="mosaic" alt="RGB、GT、固定中心线、锚点及已有预测的对齐审查图"></a>
<h3>原判据失败原因（保持原评分不变）</h3><div id="scores"></div>
<p class="muted">原判据：缺失坐标直接覆盖 ≥80%，且全部锚点组具有共同全图预测分量。2像素容差覆盖和局部连通仅为辅助审查，不替代原计分。全图连通、局部不通提示连接可能离开裁剪范围，不自动判定误连接。</p></div>
<div class="panel"><h3>人工道路与背景审查</h3><div class="fields" id="fields"></div>
<label class="wide">背景因素（可多选）</label><div class="tags" id="backgrounds"></div>
<label>备注 / 标注疑点 / 道路走向<textarea id="notes"></textarea></label>
<p class="muted">修改会尝试保存在当前浏览器。关闭前请导出 JSON/CSV；只复制 index.html 会丢失图片，应下载完整目录。</p></div>
<details class="panel"><summary>查看 14 张影像的全图标记</summary><select id="overviewCase"></select><a id="fullLink" target="_blank"><img id="full" loading="lazy" alt="固定缺口在原影像上的编号标记"></a></details>
</section></main>
<script id="data" type="application/json">__DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('data').textContent), rows=data.rows, summary=data.summary;
const byId=Object.fromEntries(rows.map(r=>[r.gap_uid,r]));
const key='fixed-gap-review:'+summary.manifest_sha256; let annotations={},current=rows[0]?.gap_uid,saveWarning='';
try{annotations=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){saveWarning='浏览器本地保存不可用，请及时导出'}
const $=id=>document.getElementById(id), cases=summary.case_ids||[...new Set(rows.map(r=>r.case))];
const fields={
 confirmed_true_gap:['是否确认真实断路',[['','未审查'],['yes','是：可信的真实断路'],['no','否：筛选混入 / 不是断路'],['uncertain','不确定：证据不足']]],
 road_type:['道路类型',[['','未填写'],['wide_paved','较宽铺装道路'],['narrow_paved','窄铺装 / 居住区道路'],['unpaved_track','非铺装 / 小径'],['junction_branch','路口 / 支路'],['internal_parking','内部道路 / 停车场'],['uncertain','不确定']]],
 scene_type:['缺失场景',[['','未填写'],['canopy_occlusion','树冠遮挡疑似断路'],['shadow','阴影区域漏检'],['visible_narrow_missed','可见窄路漏检'],['junction_missed','路口 / 支路漏检'],['low_contrast','低对比道路漏检'],['neighbor_anchor_mix','邻路 / 其他碎片混入锚点'],['spatial_offset','道路位置偏移'],['annotation_questionable','GT 标注存疑'],['not_clear_break','不构成明确断路'],['uncertain','不确定']]],
 anchor_reliability:['锚点是否像同一道路两侧',[['','未填写'],['plausible_flanks','可信的两侧道路'],['multiple_roads','多道路 / 多锚点混入'],['unreliable','锚点不可靠'],['uncertain','不确定']]],
 confidence:['审查信心',[['','未填写'],['high','高'],['medium','中'],['low','低']]]};
const backgrounds=[['trees','树木 / 植被'],['shadow','投影 / 阴影'],['buildings','建筑 / 屋顶'],['open_field','田地 / 田埂'],['parallel_roads','邻近 / 平行道路'],['parking','停车场 / 硬化地面'],['water_bridge','水体 / 桥梁'],['low_contrast','低对比背景'],['clear_visible','无明显遮挡'],['uncertain','不确定']];
function option(parent,value,text){let o=document.createElement('option');o.value=value;o.textContent=text;parent.append(o)}
function values(id){return annotations[id]||{}}
function reason(value){const text={passed:'通过原代理判据',centerline_coverage_below_80pct:'中心线直接覆盖不足80%',anchor_group_disappeared:'有锚点组消失',anchor_groups_not_connected:'固定锚点组未连通',unavailable_cache_or_threshold:'无可用预测缓存 / 阈值，未评分'};return String(value||'').split('|').map(x=>text[x]||x).join('；')}
function save(){try{localStorage.setItem(key,JSON.stringify(annotations))}catch(e){saveWarning='本地保存失败，请导出 JSON'}$('status').textContent=saveWarning||'已保存在当前浏览器（仍建议导出）'}
function change(field,value){annotations[current]={...values(current),[field]:value};save();list()}
for(const [name,[title,options]] of Object.entries(fields)){
 let label=document.createElement('label');label.textContent=title;let select=document.createElement('select');select.id='field_'+name;for(const [v,t]of options)option(select,v,t);select.onchange=()=>change(name,select.value);label.append(select);$('fields').append(label);
}
for(const [name,title]of backgrounds){let label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.value=name;input.onchange=()=>change('background',[...$('backgrounds').querySelectorAll('input:checked')].map(x=>x.value).join('|'));label.append(input,document.createTextNode(title));$('backgrounds').append(label)}
$('notes').oninput=()=>change('notes',$('notes').value);
option($('caseFilter'),'all','全部影像');for(const name of cases){option($('caseFilter'),name,name);option($('overviewCase'),name,name)}
function visible(){return rows.filter(r=>($('caseFilter').value==='all'||r.case===$('caseFilter').value)&&($('lengthFilter').value==='all'||r.source_length_bin===$('lengthFilter').value)&&($('reviewFilter').value==='all'||($('reviewFilter').value==='todo'?!values(r.gap_uid).confirmed_true_gap:values(r.gap_uid).confirmed_true_gap===$('reviewFilter').value)))}
function list(){let available=visible();$('list').replaceChildren();for(const r of available){let b=document.createElement('button');b.className='item'+(r.gap_uid===current?' active':'');b.textContent=(values(r.gap_uid).confirmed_true_gap?'✓ ':'○ ')+r.gap_uid+' · '+r.case+' · '+r.gap_pixel_count+'px';b.onclick=()=>show(r.gap_uid);$('list').append(b)}$('progress').textContent='已审查 '+rows.filter(r=>values(r.gap_uid).confirmed_true_gap).length+'/'+rows.length+'；当前筛选 '+available.length}
function show(id){current=id;let r=byId[id];if(!r)return;
 $('heading').textContent=r.gap_uid+' · '+r.case+' · 原候选 '+r.source_gap_id;
 $('geometry').textContent='原生中心 (x,y)=('+r.centroid_x.toFixed(1)+', '+r.centroid_y.toFixed(1)+')；缺失 '+r.gap_pixel_count+' 像素；锚点 '+r.anchor_groups+' 组；裁剪 ['+r.crop_x0+','+r.crop_y0+']–['+r.crop_x1_exclusive+','+r.crop_y1_exclusive+')';
 $('rgbLink').href=r.rgb_crop;$('overviewLink').href=r.case_overview;$('originalLink').href='originals/'+r.case+'.png';$('mosaic').src=r.mosaic;$('mosaicLink').href=r.mosaic;
 $('scores').replaceChildren();let table=document.createElement('table');
 for(const [i,role]of ['header','p0','current_S','choice'].entries()){
  let tr=document.createElement('tr');let cells=i===0?['预测','直接覆盖','锚点消失组','全图连通','局部连通','原判据 / 未通过原因']:[role,r[role+'_centerline_coverage']===undefined?'无缓存':(100*r[role+'_centerline_coverage']).toFixed(1)+'%',r[role+'_missing_anchor_groups']??'—',r[role+'_global_anchors_connected']??'—',r[role+'_diagnostic_local_anchors_connected']??'—',reason(r[role+'_failure_reason'])];
  for(const cell of cells){let td=document.createElement(i===0?'th':'td');td.textContent=String(cell);tr.append(td)}table.append(tr)
 }$('scores').append(table);
 for(const name of Object.keys(fields))$('field_'+name).value=values(id)[name]||'';
 const selected=(values(id).background||'').split('|');for(const input of $('backgrounds').querySelectorAll('input'))input.checked=selected.includes(input.value);
 $('notes').value=values(id).notes||'';list();
}
for(const id of ['caseFilter','reviewFilter','lengthFilter'])$(id).onchange=()=>{list();let candidates=visible();if(!candidates.some(r=>r.gap_uid===current)&&candidates[0])show(candidates[0].gap_uid)};
function move(delta){let candidates=visible(),index=candidates.findIndex(r=>r.gap_uid===current);if(candidates.length)show(candidates[(index+delta+candidates.length)%candidates.length].gap_uid)}
$('previous').onclick=()=>move(-1);$('next').onclick=()=>move(1);
function download(name,text,type){const a=document.createElement('a'),url=URL.createObjectURL(new Blob([text],{type}));a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
function annotatedRows(){return rows.map(r=>({...r,...values(r.gap_uid)}))}
$('exportJson').onclick=()=>download('fixed_gap_annotations.json',JSON.stringify({manifest_sha256:summary.manifest_sha256,rows:annotatedRows()},null,2),'application/json');
$('exportCsv').onclick=()=>{const result=annotatedRows(),keys=[...new Set(result.flatMap(r=>Object.keys(r)))],quote=x=>'"'+String(x??'').replaceAll('"','""')+'"';download('fixed_gap_annotations.csv','\ufeff'+[keys,...result.map(r=>keys.map(k=>r[k]))].map(row=>row.map(quote).join(',')).join('\r\n'),'text/csv;charset=utf-8')};
$('import').onclick=()=>$('file').click();$('file').onchange=async()=>{try{let imported=JSON.parse(await $('file').files[0].text());if(imported.manifest_sha256!==summary.manifest_sha256)throw Error('清单 SHA 不同，不能混入另一批候选');let next={...annotations};for(const row of imported.rows){if(!byId[row.gap_uid])throw Error('未知缺口 ID '+row.gap_uid);next[row.gap_uid]=Object.fromEntries([...Object.keys(fields),'background','notes'].map(k=>[k,String(row[k]||'')]))}annotations=next;save();show(current)}catch(e){alert('导入失败：'+e.message)}};
function fullImage(){const path='overviews/'+$('overviewCase').value+'_marked.png';$('full').src=path;$('fullLink').href=path}$('overviewCase').onchange=fullImage;
$('identity').textContent=summary.images+' 张原影像 · '+summary.fixed_candidates+' 个固定候选 · 原 P0/GT 指纹'+(summary.reference_fingerprint_verified?'已核验':'未完整核验（缓存缺失）')+' · S/choice 公共阈值 '+(summary.common_threshold??'未取得');
fullImage();if(current)show(current);$('status').textContent=saveWarning||'浏览器本地保存 + 可导出 JSON/CSV';
</script></html>'''


def write_review_page(path,rows,summary):
    data=json.dumps({'rows':rows,'summary':summary},ensure_ascii=False).replace('</','<\\/')
    path.write_text(PAGE.replace('__DATA__',data),encoding='utf-8')
