"""Render M1/M2 from completed SCGRA exports. No model or GPU is needed."""
import argparse
from collections import Counter, defaultdict
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np


METHODS = ("structure_fps", "probability_fps", "random")
MODES = ("full", "no_connectivity_bias", "no_spatial_bias", "no_both_biases")
MODE_LABELS = ["Full", "Without\nconn. bias", "Without\nspatial bias", "Without\nboth biases"]
COLORS = ["#b66528", "#397da9", "#8a959e"]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 7, "pdf.fonttype": 42, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False})


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def image(path):
    return plt.imread(path)


def binary(path):
    result = image(path)
    if result.ndim != 2 or not np.isin(result, [0, 1]).all():
        raise ValueError(f"Expected a binary PNG: {path}")
    return result.astype(bool)


def save(fig, axes, output, stem):
    fig.canvas.draw()
    if np.asarray(axes).ndim == 1:
        axes = np.asarray(axes)[None, :]
    axes = np.asarray(axes)
    identifiers = np.asarray(list("abcdefghijkl")[:axes.size]).reshape(axes.shape)
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ModuleNotFoundError as error:
        if error.name != "audit_panel_alignment":
            raise
    else:
        require_matplotlib_panel_alignment(
            fig, axes=list(axes.flat), panel_ids=list(identifiers.flat),
            row_groups=[list(line) for line in identifiers if len(line) > 1],
            column_groups=[list(line) for line in identifiers.T if len(line) > 1],
            strict=True, json_out=output / f"{stem}.alignment.json",
        )
    fig.savefig(output / f"{stem}.png", dpi=240)
    fig.savefig(output / f"{stem}.pdf", dpi=400)
    plt.close(fig)
    return stem


def m1_summary(root, output, meta):
    image_ids = meta["evaluated_case_ids"]
    data = defaultdict(list)
    for row in rows(root / "anchor_statistics_per_image.csv"):
        data[row["case_id"], row["method"]].append(row)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))
    fig.subplots_adjust(left=.095, right=.97, bottom=.26, top=.76, wspace=.47)
    fields = (("road_hit_rate_exact", "Road hit rate (%)", 100),
              ("nearest_neighbor_mean_px", "Mean nearest-neighbor distance (px)", 1),
              ("component_coverage", "GT components hit (%)", 100))
    for ax, (field, label, scale) in zip(axes, fields):
        values = np.array([[np.mean([float(row[field]) for row in data[c, method] if row[field]])
                            for method in METHODS] for c in image_ids]) * scale
        for value in values:
            if np.isfinite(value).all():
                ax.plot(range(3), value, color="#c4cdd2", lw=.42, alpha=.68, zorder=1)
        for i, column in enumerate(values.T):
            valid = column[np.isfinite(column)]
            if not len(valid):
                continue
            ax.scatter(np.full(len(valid), i), valid, color=COLORS[i], s=9, alpha=.48)
            ax.errorbar(i, valid.mean(), yerr=valid.std(ddof=1) if len(valid) > 1 else 0,
                        fmt="o", color=COLORS[i], ecolor="#222222", markeredgecolor="#222222",
                        lw=.7, capsize=2, ms=4.5, zorder=4)
        ax.set_ylabel(label)
        ax.set_xticks(range(3), ["Structure × P\n+ FPS", "P only\n+ FPS", "Random"])
        ax.set_xlim(-.35, 2.35)
    fig.text(.095,.94, f"M1  |  Anchor placement  |  Massachusetts {meta['split']}, n={len(image_ids)} images",
             weight="bold", fontsize=8)
    fig.text(.095,.065, "Lines pair images. Random draws are averaged within image. Large markers: mean ± image SD.",
             fontsize=6)
    return save(fig, axes, output, "M1_anchor_statistics")


def m1_spatial(root, output, case, limit):
    directory = root / "cases" / case
    # Same fixed central 512-pixel window in every case.
    region = np.s_[512:1024, 512:1024]
    rgb = image(directory / "image.png")[region]
    with np.load(directory / "windows/y0512_x0512.npz", allow_pickle=False) as archive:
        valid = archive["selected_valid"]
        candidates = archive["candidate_coords"][archive["candidate_valid"]]
        anchors = archive["selected_coords"][valid]
        attention = archive["node_attention_mean"][valid][:, valid]
        prior = archive["road_prior"]
        residual = archive["effective_residual_rms"]
    with np.load(directory / "prediction_on_off.npz", allow_pickle=False) as archive:
        delta = archive["probability_delta"][region]
    changes = np.rint(image(directory / "change_categories.png")[region] * 255).astype(int)
    overlay = rgb.copy() * .55
    change_colors = ((.15,.55,.95),(.94,.63,.15),(.95,.36,.64),(.2,.81,.55))
    for label, color in enumerate(change_colors, 1):
        overlay[changes == label, :3] = color
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.9))
    fig.subplots_adjust(left=.075,right=.975,top=.90,bottom=.145,wspace=.24,hspace=.46)
    a, b, c, d, e, f = axes.flat
    images = [
        (a, a.imshow(prior, cmap="cividis", vmin=0, vmax=1), "Probability"),
        (c, c.imshow(attention, cmap="cividis", vmin=0, vmax=1) if len(anchors) else None, "Weight"),
        (d, d.imshow(residual, cmap="magma", vmin=0, vmax=limit[0]), "Feature RMS"),
        (e, e.imshow(delta, cmap="PuOr_r", norm=TwoSlopeNorm(0,-limit[1],limit[1])), "Probability difference"),
    ]
    b.imshow(rgb)
    b.scatter(candidates[:,1], candidates[:,0], s=2, color="#59c1ee", linewidths=0)
    b.scatter(anchors[:,1], anchors[:,0], s=9, facecolors="#e79c2d", edgecolors="#202020", linewidths=.35)
    b.set(xlim=(-.5,511.5),ylim=(511.5,-.5))
    f.imshow(overlay)
    # Independent colorbar axes leave all six plot rectangles aligned.
    for ax, artist, label in images:
        if artist is not None:
            box = ax.get_position()
            bar = fig.add_axes([box.x0,box.y0-.031,box.width,.009])
            cb = fig.colorbar(artist, cax=bar, orientation="horizontal")
            cb.set_label(label, fontsize=6, labelpad=1)
            cb.ax.tick_params(labelsize=5.5, length=1.4, pad=1)
    titles = ["Road prior","Candidates and anchors","Mean node attention",
              "Effective writeback","ON − OFF","Changed pixels"]
    for ax, title in zip(axes.flat,titles):
        ax.set_title(title, fontsize=7)
        ax.set_axis_off()
    fig.text(.075,.965, f"SCGRA spatial trace  |  {case}  |  Window (512, 512)", fontsize=8, weight="bold")
    fig.text(.075,.065,"Cyan dots: candidate points. Orange circles: selected anchors. Delta uses stitched probabilities.",
             fontsize=6)
    legend = [Patch(color=color,label=label) for color,label in zip(change_colors,
              ("New TP","New FP","Lost TP","Removed FP"))]
    fig.legend(handles=legend, ncol=4, loc="lower center", bbox_to_anchor=(.55,.007),
               fontsize=6, frameon=False)
    return save(fig, axes, output, "M1_case_" + case)


def m2_summary(root, output, meta):
    relation = {r["mode"]:r for r in rows(root/"relation_and_segmentation_summary.csv")}
    graph = {r["mode"]:r for r in rows(root/"graph_summary.csv")}
    fig, axes = plt.subplots(1,3,figsize=(7.2,2.8))
    fig.subplots_adjust(left=.08,right=.975,top=.76,bottom=.26,wspace=.55)
    a,b,c = axes
    if all(mode in relation for mode in MODES):
        a.bar(range(4),[100*float(relation[mode]["dominant_key_mass"]) for mode in MODES],
              color=["#b66528","#397da9","#397da9","#397da9"])
        a.set_xticks(range(4),MODE_LABELS,fontsize=5.5)
        a.set_ylim(0,105)
        a.set_title("Bias interventions",fontsize=7)
        a.set_ylabel("Dominant key mass (%)")
    else:
        a.text(.5,.5,"Bias interventions\nnot present in this export",ha="center",va="center",transform=a.transAxes)
        a.set_xticks([])
        a.set_yticks([])
    off,on = relation["alpha_zero"], relation["full"]
    values = [100*(float(on[key])-float(off[key])) for key in ("iou","precision","recall")]
    b.bar(range(3),values,color=["#397da9" if v>=0 else "#b25d81" for v in values])
    b.axhline(0,color="#3d454d",lw=.6)
    b.set_xticks(range(3),["IoU","Precision","Recall"],fontsize=6)
    b.set_ylabel("ON − OFF (percentage points)")
    b.set_title("Segmentation",fontsize=7)
    g = [graph["alpha_zero"],graph["full"]]
    for i, row in enumerate(g):
        c.bar(np.arange(2)+(.34*i-.17),[int(row["attached_unsupported"]),int(row["isolated_unsupported"])],
              width=.34,color=("#83949e","#b66528")[i],label=("OFF","ON")[i])
    c.set_xticks(range(2),["Attached","Isolated"],fontsize=6)
    c.set_ylabel("Predicted segments")
    c.set_title("Segments outside GT tolerance",fontsize=7)
    c.legend(frameon=False,fontsize=6)
    fig.text(.08,.94,f"M2  |  SCGRA interventions  |  Massachusetts {meta['split']}, n={len(meta['image_ids'])} images",
             weight="bold",fontsize=8)
    rates = [100*float(row["match_rate"]) for row in g]
    fig.text(.08,.07,f"Fixed threshold {meta['threshold']:.2f}. Local path match: OFF {rates[0]:.2f}%; ON {rates[1]:.2f}%.",
             fontsize=6)
    return save(fig, axes, output,"M2_intervention_summary")


def m2_gallery(root, output):
    paired = defaultdict(Counter)
    for row in rows(root / "paired_path_changes.csv"):
        if row["control"] == "alpha_zero":
            paired[row["case_id"]][row["outcome"]] += 1
    if not paired:
        return None
    # Outcome-dependent examples are explicitly descriptive, with a separate loss case.
    gain = min(paired, key=lambda case:(-paired[case]["gap_recovered"],case))
    loss = min(paired, key=lambda case:(-paired[case]["gap_lost"],case == gain,case))
    cases = [gain] if loss == gain else [gain,loss]
    fig, axs = plt.subplots(len(cases),4,figsize=(7.2, 2.28*len(cases)),squeeze=False)
    fig.subplots_adjust(left=.052,right=.976,bottom=.095,top=.83,wspace=.12,hspace=.33)
    for index, case in enumerate(cases):
        directory = root / "maps" / case
        rgb = image(directory / "rgb.png")
        gt = binary(directory / "gt.png")
        off = binary(directory / "alpha_zero_mask.png")
        on = binary(directory / "full_mask.png")
        overlay = rgb.copy()*.53
        overlay[off & ~on & gt,:3] = [.96,.36,.64]
        overlay[off & ~on & ~gt,:3] = [.24,.85,.53]
        overlay[on & ~off,:3] = [.98,.68,.17]
        for j, (ax, panel) in enumerate(zip(axs[index],(rgb,off,on,overlay))):
            ax.imshow(panel,cmap="gray",vmin=0,vmax=1,interpolation="nearest")
            ax.set_axis_off()
            if index == 0:
                ax.set_title(("RGB","Writeback OFF","Full model","Pixel changes")[j],fontsize=7)
        count = paired[case]
        axs[index,0].text(0,-.06,f"{case}  |  Probe gains {count['gap_recovered']} / losses {count['gap_lost']}",
                          transform=axs[index,0].transAxes,fontsize=6,va="top")
    fig.text(.052,.968,"M2  |  Paired road maps",weight="bold",fontsize=8)
    fig.text(.052,.926,"Examples with the most gap-probe gains and losses. A probe match can change without new road pixels.",
             fontsize=6)
    fig.legend(handles=[Patch(color="#f55ca3",label="Lost true road"),
                        Patch(color="#3dd987",label="Removed false positive"),
                        Patch(color="#faad2b",label="New foreground")],
               frameon=False,ncol=3,loc="lower center",bbox_to_anchor=(.5,.003),fontsize=6)
    return save(fig, axs, output,"M2_location_examples")


def topology_figure(path, output):
    scores = {r["name"]:r for r in rows(path)}
    off,on = scores["SCGRA_OFF"],scores["SCGRA_ON"]
    fields = [("cldice","clDice ↑"),("topo_precision","TOPO-P ↑"),("topo_recall","TOPO-R ↑"),
              ("topo_f1","TOPO-F1 ↑"),("apls_bidirectional","Sampled APLS ↑"),("tlts_infeasible","TLTS-Infeasible ↓")]
    fig, ax = plt.subplots(figsize=(7.2,3.1))
    fig.subplots_adjust(left=.20,right=.94,top=.79,bottom=.22)
    for i,(field,label) in enumerate(fields):
        x0,x1 = float(off[field])*100,float(on[field])*100
        ax.plot([x0,x1],[i,i],color="#8b969d",lw=.95)
        ax.scatter(x0,i,s=23,color="#83949e",marker="s",label="OFF" if i==0 else None)
        ax.scatter(x1,i,s=24,color="#b66528",label="ON" if i==0 else None)
    ax.set_yticks(range(len(fields)),[label for _,label in fields])
    ax.set_xlabel("Score (%)")
    ax.set_xlim(0,104)
    ax.invert_yaxis()
    ax.legend(frameon=False,ncol=2,loc="lower right",bbox_to_anchor=(1,1.02),fontsize=6)
    fig.text(.20,.94,"SCGRA ON/OFF  |  Six topology metrics",weight="bold",fontsize=8)
    fig.text(.20,.07,"Both masks use the validation-selected threshold. Arrows show the preferred direction.",
             fontsize=6)
    # A single plot has no comparable panel group.
    fig.savefig(output/"M2_topology_metrics.png",dpi=240)
    fig.savefig(output/"M2_topology_metrics.pdf",dpi=400)
    plt.close(fig)
    return "M2_topology_metrics"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path, help="Directory containing m1_anchors and m2_locations")
    args = parser.parse_args()
    root = args.results
    output = root / "figures"
    output.mkdir(parents=True,exist_ok=True)
    m1, m2 = root/"m1_anchors", root/"m2_locations"
    figures = []
    if (m1/"export_complete.json").is_file():
        meta = read_json(m1/"metadata.json")
        figures.append(m1_summary(m1,output,meta))
        cases = meta["export_case_ids"]
        residual_max, delta_max = 1e-12, 1e-12
        for case in cases:
            directory = m1/"cases"/case
            with np.load(directory/"windows/y0512_x0512.npz", allow_pickle=False) as z:
                residual_max = max(residual_max, float(z["effective_residual_rms"].max()))
            with np.load(directory/"prediction_on_off.npz", allow_pickle=False) as z:
                delta_max = max(delta_max, float(abs(z["probability_delta"][512:1024,512:1024]).max()))
        for case in cases:
            figures.append(m1_spatial(m1,output,case,(residual_max,delta_max)))
    if (m2/"report.html").is_file():
        figures.append(m2_summary(m2,output,read_json(m2/"metadata.json")))
        gallery = m2_gallery(m2,output)
        if gallery:
            figures.append(gallery)
    topology = root/"topology_v7/test_comparison.csv"
    if topology.is_file():
        figures.append(topology_figure(topology,output))
    if not figures:
        parser.error("No completed M1 or M2 result found beneath --results")
    links = "".join(f'<figure><a href="{f}.pdf">{html.escape(f)} · PDF</a>'
                    f'<img src="{f}.png" loading="lazy"></figure>' for f in figures)
    details = "".join(f'<a href="../{file}">{label}</a> · ' for file, label in (
        ("m1_anchors/anchor_summary.csv","M1 数据表"),
        ("m2_locations/report.html","M2 逐图报告"),
        ("topology_v7/test_comparison.csv","六项拓扑数据表"),
    ) if (root/file).is_file())
    (output/"index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>SCGRA figures</title>'
        '<style>body{max-width:1200px;margin:28px auto;padding:0 18px;font:16px system-ui}'
        'img{width:100%;display:block}figure{margin:24px 0;border-top:1px solid #ddd;padding-top:12px}</style>'
        '<h1>SCGRA M1 / M2 图件</h1>'+details+links,encoding="utf-8")
    print(output/"index.html")


if __name__ == "__main__":
    main()
